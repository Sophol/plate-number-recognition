"""Explains *why* a stream will not open, so the UI can say something useful.

OpenCV only reports a boolean, and the RTSP retry loop hides the reason behind
an endless "connecting". A single RTSP DESCRIBE tells us whether the host is
unreachable, the credentials are refused, or the path is wrong.
"""

import re
import socket
from urllib.parse import urlparse, unquote

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

    try:
        sock = socket.create_connection((host, port), timeout=TIMEOUT)
    except OSError as exc:
        return False, f"cannot reach {host}:{port} ({exc.strerror or exc})"

    try:
        path = parsed.path or "/"
        request = (
            f"DESCRIBE rtsp://{host}:{port}{path} RTSP/1.0\r\n"
            "CSeq: 1\r\nAccept: application/sdp\r\n\r\n"
        )
        sock.sendall(request.encode())
        reply = sock.recv(2048).decode("utf-8", "replace")
    except OSError as exc:
        return False, f"RTSP handshake failed ({exc})"
    finally:
        sock.close()

    status = reply.splitlines()[0] if reply else ""
    if " 200 " in status:
        return True, "stream available"

    if " 401 " in status:
        realm = re.search(r'realm="([^"]+)"', reply)
        who = f" ({realm.group(1)})" if realm else ""
        if parsed.username:
            return False, (
                f"camera{who} rejected the username/password for "
                f"{unquote(parsed.username)!r}"
            )
        return False, f"camera{who} requires a username and password in the URL"

    if " 404 " in status:
        return False, f"camera has no stream at path {path!r}"

    return False, status or "no response to RTSP DESCRIBE"
