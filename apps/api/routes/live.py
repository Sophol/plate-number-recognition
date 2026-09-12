import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.live.session import registry
from apps.api.security import Viewer, decode_token
from db.models import Camera
from db.session import get_session

router = APIRouter(prefix="/live", tags=["live"])

Session = Annotated[AsyncSession, Depends(get_session)]

BOUNDARY = "frame"


async def _camera_or_404(camera_id: uuid.UUID, session: AsyncSession) -> Camera:
    camera = await session.get(Camera, camera_id)
    if camera is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Camera not found")
    return camera


@router.get("/cameras")
async def live_cameras(session: Session, _: Viewer) -> list[dict]:
    """Active cameras plus the state of any running preview."""
    from sqlalchemy import select

    result = await session.execute(select(Camera).where(Camera.is_active.is_(True)))
    out = []
    for camera in result.scalars().all():
        existing = registry.existing(str(camera.id))
        out.append(
            {
                "id": str(camera.id),
                "name": camera.name,
                "rtsp_url": camera.rtsp_url,
                "preview": existing.status if existing else None,
            }
        )
    return out


@router.get("/{camera_id}/events")
async def live_events(camera_id: uuid.UUID, session: Session, _: Viewer, limit: int = 20) -> dict:
    """Per-detection pipeline trace for the explainer panel."""
    camera = await _camera_or_404(camera_id, session)
    existing = registry.existing(str(camera.id))
    if existing is None:
        return {"status": None, "events": []}
    return {"status": existing.status, "events": existing.recent_events(limit)}


@router.get("/{camera_id}/stream")
async def live_stream(
    camera_id: uuid.UUID,
    session: Session,
    token: Annotated[str | None, Query()] = None,
) -> StreamingResponse:
    """MJPEG preview.

    An <img> tag cannot send an Authorization header, so the token is accepted
    as a query parameter here and validated the same way.
    """
    if token is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing token")
    decode_token(token)  # raises 401 on a bad or expired token

    camera = await _camera_or_404(camera_id, session)
    preview = registry.get(str(camera.id), camera.name, camera.rtsp_url)
    preview.acquire()

    async def frames() -> AsyncIterator[bytes]:
        try:
            while True:
                jpeg = await preview.next_frame()
                yield (
                    b"--" + BOUNDARY.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                )
        finally:
            preview.release()

    return StreamingResponse(
        frames(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store", "Connection": "close"},
    )
