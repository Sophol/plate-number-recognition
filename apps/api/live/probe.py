"""Explains *why* a stream will not open, so the UI can say something useful.

OpenCV only reports a boolean, and the RTSP retry loop hides the reason behind
an endless "connecting". A single RTSP DESCRIBE tells us whether the host is
unreachable, the credentials are refused, or the path is wrong.
"""

import base64
import hashlib
import os
import re
import secrets
import socket
from urllib.parse import unquote, urlparse

RTSP_DEFAULT_PORT = 554
TIMEOUT = 6.0


def probe(url: str) -> tuple[bool, str]:
    """Return (looks_ok, human readable reason)."""
    from apps.api.live.devicesource import parse_device

    index = parse_device(url)
    if index is not None:
        import cv2

        capture = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        opened = capture.isOpened()
        ok, frame = capture.read() if opened else (False, None)
        capture.release()
        if not opened:
            return False, f"no camera at device index {index}"
        if not ok or frame is None:
            return False, (
                f"camera {index} opened but delivered no frame "
                "(another app may be using it, or macOS camera permission is denied)"
            )
        return True, f"camera {index} ready"

    if url.startswith("device:"):
        return False, f"malformed device url {url!r}; expected device:<number>"

    parsed = urlparse(url)

    if parsed.scheme in ("", "file") or not parsed.netloc:
        # A local file path is a legitimate source for testing.
        import os

        if os.path.exists(url):
            return True, "local file"
        return False, f"no such file: {url}"

    if parsed.scheme != "rtsp":
        return False, f"unsupported scheme {parsed.scheme!r}"

    host = parsed.hostname
    if not host:
        return False, "no host in URL"
    port = parsed.port or RTSP_DEFAULT_PORT

    try:
        socket.gethostbyname(host)
    except OSError:
        return False, f"hostname {host!r} does not resolve"

    # The request URI carries no userinfo; credentials travel in a header.
    path = parsed.path or "/"
    target = f"rtsp://{host}:{port}{path}"

    # The challenge and the authenticated retry must share one connection: many
    # cameras (the GD882 at this gate among them) bind the Digest nonce to the
    # TCP socket, so a nonce reused on a fresh connection comes back 401. Opening
    # per request is what made a correct password look rejected.
    try:
        sock = socket.create_connection((host, port), timeout=TIMEOUT)
    except OSError as exc:
        return False, _connect_error(host, port, exc)

    try:
        reply = _describe(sock, target, cseq=1)
        status = _status(reply)
        if " 200 " in status:
            return True, "stream available"
        if " 404 " in status:
            return False, f"camera has no stream at path {path!r}"
        if " 401 " not in status:
            return False, status or "no response to RTSP DESCRIBE"

        # A 401 to an unauthenticated DESCRIBE is not a rejection - it is the
        # camera asking. Nearly every IP camera does this. Only a 401 to a
        # request that actually carried the credentials means they were refused.
        realm = re.search(r'realm="([^"]*)"', reply)
        who = f" ({realm.group(1)})" if realm else ""
        if not parsed.username:
            return False, f"camera{who} requires a username and password in the URL"

        username = unquote(parsed.username)
        password = unquote(parsed.password or "")
        authorization = _answer_challenge(reply, username, password, target)
        if authorization is None:
            return False, f"camera{who} uses an authentication scheme this probe does not support"

        reply = _describe(sock, target, cseq=2, authorization=authorization)
        status = _status(reply)
        if " 200 " in status:
            return True, "stream available"
        if " 401 " in status:
            return False, f"camera{who} rejected the username/password for {username!r}"
        if " 404 " in status:
            return False, f"camera has no stream at path {path!r}"
        return False, status or "no response to authenticated RTSP DESCRIBE"
    except OSError as exc:
        return False, f"RTSP handshake failed ({exc})"
    finally:
        sock.close()


def _describe(sock: socket.socket, target: str, cseq: int, authorization: str | None = None) -> str:
    """Send one DESCRIBE on an already-open socket and return the raw reply."""
    headers = f"CSeq: {cseq}\r\nAccept: application/sdp\r\n"
    if authorization:
        headers += f"Authorization: {authorization}\r\n"
    request = f"DESCRIBE {target} RTSP/1.0\r\n{headers}\r\n"
    sock.sendall(request.encode())
    return sock.recv(4096).decode("utf-8", "replace")


def _status(reply: str) -> str:
    return reply.splitlines()[0] if reply else ""


def _connect_error(host: str, port: int, exc: OSError) -> str:
    return f"cannot reach {host}:{port} ({exc.strerror or exc})"


def _answer_challenge(reply: str, username: str, password: str, uri: str) -> str | None:
    """Build an Authorization header for the camera's WWW-Authenticate challenge.

    Digest (RFC 2617) is what IP cameras overwhelmingly use; Basic is kept for
    the few that offer it. A camera may advertise both - Digest is preferred
    because Basic sends the password in the clear.
    """
    challenges = re.findall(r"WWW-Authenticate: (.+)", reply)
    digest = next((c for c in challenges if c.startswith("Digest ")), None)
    basic = next((c for c in challenges if c.startswith("Basic ")), None)

    if digest:
        fields = dict(re.findall(r'(\w+)="([^"]*)"', digest))
        realm, nonce = fields.get("realm", ""), fields.get("nonce", "")
        ha1 = hashlib.md5(f"{username}:{realm}:{password}".encode()).hexdigest()
        ha2 = hashlib.md5(f"DESCRIBE:{uri}".encode()).hexdigest()

        parts = [f'username="{username}"', f'realm="{realm}"', f'nonce="{nonce}"', f'uri="{uri}"']
        qop_match = re.search(r'qop="?([^",]+)"?', digest)
        if qop_match and "auth" in qop_match.group(1).split(","):
            nc, cnonce = "00000001", secrets.token_hex(8)
            response = hashlib.md5(f"{ha1}:{nonce}:{nc}:{cnonce}:auth:{ha2}".encode()).hexdigest()
            parts += [f"qop=auth", f"nc={nc}", f'cnonce="{cnonce}"']
        else:
            response = hashlib.md5(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()
        parts.append(f'response="{response}"')
        if "opaque" in fields:
            parts.append(f'opaque="{fields["opaque"]}"')
        return "Digest " + ", ".join(parts)

    if basic:
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        return f"Basic {token}"

    return None
