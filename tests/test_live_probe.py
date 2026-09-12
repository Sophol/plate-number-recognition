"""Tests for the RTSP pre-flight probe.

Run against a tiny in-process RTSP server that behaves like a real IP camera:
it answers an unauthenticated DESCRIBE with a 401 Digest challenge and only
returns 200 once the client presents a correct Digest response. The bug these
guard against is a probe that never sends credentials at all and then reports
the camera's routine 401 challenge as "rejected the username/password".
"""

import base64
import hashlib
import re
import socket
import threading

import pytest

from apps.api.live.probe import probe

REALM = "IP Camera(TEST)"
NONCE = "abc123nonce"


class FakeCamera:
    """Minimal RTSP responder. Handles one DESCRIBE exchange per connection."""

    def __init__(self, username: str, password: str, scheme: str = "Digest"):
        self.username, self.password, self.scheme = username, password, scheme
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.requests: list[str] = []
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                # Nonce is bound to this connection, like a real GD882: a Digest
                # response presented on a different socket is rejected.
                challenged = False
                for _ in range(2):
                    try:
                        data = conn.recv(4096).decode()
                    except OSError:
                        break
                    if not data:
                        break
                    self.requests.append(data)
                    conn.sendall(self._respond(data, challenged).encode())
                    challenged = True

    def _respond(self, request: str, challenged: bool) -> str:
        cseq = re.search(r"CSeq: (\d+)", request)
        cseq = cseq.group(1) if cseq else "1"
        auth = re.search(r"Authorization: (.+)\r\n", request)
        # Digest auth only counts on the socket that issued the nonce.
        if auth and challenged and self._valid(auth.group(1), request):
            return (
                f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nContent-Type: application/sdp\r\n"
                "Content-Length: 0\r\n\r\n"
            )
        challenge = (
            f'Digest realm="{REALM}", nonce="{NONCE}"'
            if self.scheme == "Digest"
            else f'Basic realm="{REALM}"'
        )
        return f"RTSP/1.0 401 Unauthorized\r\nCSeq: {cseq}\r\nWWW-Authenticate: {challenge}\r\n\r\n"

    def _valid(self, header: str, request: str) -> bool:
        if header.startswith("Basic "):
            expected = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
            return header == f"Basic {expected}"
        if header.startswith("Digest "):
            fields = dict(re.findall(r'(\w+)="([^"]*)"', header))
            uri = re.search(r"DESCRIBE (\S+)", request).group(1)
            ha1 = hashlib.md5(f"{self.username}:{REALM}:{self.password}".encode()).hexdigest()
            ha2 = hashlib.md5(f"DESCRIBE:{uri}".encode()).hexdigest()
            expected = hashlib.md5(f"{ha1}:{NONCE}:{ha2}".encode()).hexdigest()
            return (
                fields.get("username") == self.username
                and fields.get("uri") == uri
                and fields.get("response") == expected
            )
        return False

    def close(self) -> None:
        self.sock.close()


@pytest.fixture
def digest_camera():
    cam = FakeCamera("admin", "Pa$$w0rd", scheme="Digest")
    yield cam
    cam.close()


@pytest.fixture
def basic_camera():
    cam = FakeCamera("admin", "Pa$$w0rd", scheme="Basic")
    yield cam
    cam.close()


def test_correct_digest_credentials_are_accepted(digest_camera):
    """The password carries '$$' and must be URL-decoded before hashing."""
    ok, reason = probe(f"rtsp://admin:Pa%24%24w0rd@127.0.0.1:{digest_camera.port}/stream")
    assert ok, reason
    # The probe must have actually answered the challenge, not given up at 401.
    assert any("Authorization: Digest" in r for r in digest_camera.requests)


def test_correct_basic_credentials_are_accepted(basic_camera):
    ok, reason = probe(f"rtsp://admin:Pa%24%24w0rd@127.0.0.1:{basic_camera.port}/")
    assert ok, reason
    assert any("Authorization: Basic" in r for r in basic_camera.requests)


def test_wrong_password_is_reported_as_rejected(digest_camera):
    ok, reason = probe(f"rtsp://admin:nope@127.0.0.1:{digest_camera.port}/stream")
    assert not ok
    assert "rejected" in reason
    assert "admin" in reason
    assert REALM in reason


def test_missing_credentials_are_reported_as_required(digest_camera):
    ok, reason = probe(f"rtsp://127.0.0.1:{digest_camera.port}/stream")
    assert not ok
    assert "requires a username and password" in reason


def test_unreachable_host_is_reported():
    ok, reason = probe("rtsp://admin:x@127.0.0.1:1/stream")
    assert not ok
    assert "cannot reach" in reason
