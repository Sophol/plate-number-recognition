import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.schemas import PlateReadCreate, PlateReadOut
from apps.api.security import ListEditor, Viewer
from db.models import PlateRead
from db.session import get_session

router = APIRouter(prefix="/plate-reads", tags=["plate-reads"])
Session = Annotated[AsyncSession, Depends(get_session)]


@router.get("", response_model=list[PlateReadOut])
async def list_plate_reads(
    session: Session,
    _: Viewer,
    camera_id: uuid.UUID | None = None,
    plate_text: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    only_active_model: bool = True,
    limit: Annotated[int, Query(le=500)] = 100,
    offset: int = 0,
):
    stmt = select(PlateRead).order_by(PlateRead.frame_ts.desc())
    if camera_id is not None:
        stmt = stmt.where(PlateRead.camera_id == camera_id)
    if plate_text is not None:
        stmt = stmt.where(PlateRead.plate_text.ilike(f"%{plate_text}%"))
    if since is not None:
        stmt = stmt.where(PlateRead.frame_ts >= since)
    if until is not None:
        stmt = stmt.where(PlateRead.frame_ts <= until)
    if only_active_model:
        stmt = stmt.where(PlateRead.is_active_model.is_(True))
    stmt = stmt.limit(limit).offset(offset)
    return (await session.execute(stmt)).scalars().all()


@router.post("", response_model=PlateReadOut, status_code=status.HTTP_201_CREATED)
async def create_plate_read(payload: PlateReadCreate, session: Session, _: ListEditor):
    read = PlateRead(**payload.model_dump())
    session.add(read)
    await session.commit()
    await session.refresh(read)
    return read


@router.post("/{plate_read_id}/verify", response_model=PlateReadOut)
async def verify_plate_read(plate_read_id: uuid.UUID, session: Session, _: ListEditor):
    # Composite PK (id, frame_ts) on the partitioned table rules out session.get().
    result = await session.execute(select(PlateRead).where(PlateRead.id == plate_read_id))
    read = result.scalar_one_or_none()
    if read is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Plate read not found")
    read.is_verified = True
    await session.commit()
    await session.refresh(read)
    return read
