import uuid
from datetime import datetime
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.schemas import ContainerLookupOut, ContainerReadCreate, ContainerReadOut
from apps.api.security import ListEditor, Viewer
from apps.inference_worker.container import load_known, normalise, parse
from config import get_settings
from db.models import ContainerRead
from db.session import get_session

router = APIRouter(tags=["containers"])
Session = Annotated[AsyncSession, Depends(get_session)]


@lru_cache
def known_containers():
    """The PAS list, loaded once per process. None when it has not been fetched."""
    return load_known(get_settings().container_known_list_path)


@router.get("/container-reads", response_model=list[ContainerReadOut])
async def list_container_reads(
    session: Session,
    _: Viewer,
    camera_id: uuid.UUID | None = None,
    container_number: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: Annotated[int, Query(le=500)] = 100,
    offset: int = 0,
):
    stmt = select(ContainerRead).order_by(ContainerRead.frame_ts.desc())
    if camera_id is not None:
        stmt = stmt.where(ContainerRead.camera_id == camera_id)
    if container_number:
        stmt = stmt.where(ContainerRead.container_number.ilike(f"%{normalise(container_number)}%"))
    if since is not None:
        stmt = stmt.where(ContainerRead.frame_ts >= since)
    if until is not None:
        stmt = stmt.where(ContainerRead.frame_ts <= until)
    return (await session.execute(stmt.limit(limit).offset(offset))).scalars().all()


@router.post("/container-reads", response_model=ContainerReadOut, status_code=status.HTTP_201_CREATED)
async def create_container_read(payload: ContainerReadCreate, session: Session, _: ListEditor):
    """Manual insert. The checksum and known flags are computed here, never trusted from the client."""
    number = parse(payload.container_number)
    canonical = number.canonical if number else normalise(payload.container_number)
    known = known_containers()
    read = ContainerRead(
        **payload.model_dump(exclude={"container_number"}),
        container_number=canonical,
        owner_code=canonical[:4],
        checksum_ok=bool(number and number.checksum_ok),
        is_known=(canonical in known) if known is not None else None,
    )
    session.add(read)
    await session.commit()
    await session.refresh(read)
    return read


@router.get("/containers/{number}", response_model=ContainerLookupOut)
async def lookup_container(number: str, session: Session, _: Viewer, recent: Annotated[int, Query(le=50)] = 10):
    """Everything known about one number: format, check digit, PAS membership, read history."""
    canonical = normalise(number)
    parsed = parse(canonical)
    known = known_containers()

    reads = (await session.execute(
        select(ContainerRead).where(ContainerRead.container_number == canonical)
        .order_by(ContainerRead.frame_ts.desc()).limit(recent)
    )).scalars().all()
    count = (await session.execute(
        select(func.count()).select_from(ContainerRead).where(ContainerRead.container_number == canonical)
    )).scalar_one()

    return ContainerLookupOut(
        number=canonical,
        well_formed=parsed is not None,
        checksum_ok=bool(parsed and parsed.checksum_ok),
        owner_code=canonical[:4] if parsed else None,
        is_known=(canonical in known) if known is not None else None,
        read_count=count,
        last_seen=reads[0].frame_ts if reads else None,
        recent_reads=reads,
    )
