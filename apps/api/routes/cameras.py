import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.schemas import CameraCreate, CameraOut, CameraUpdate
from apps.api.security import ListEditor, Viewer
from db.models import Camera
from db.session import get_session

router = APIRouter(prefix="/cameras", tags=["cameras"])
Session = Annotated[AsyncSession, Depends(get_session)]


@router.get("", response_model=list[CameraOut])
async def list_cameras(session: Session, _: Viewer, is_active: bool | None = None):
    stmt = select(Camera).order_by(Camera.name)
    if is_active is not None:
        stmt = stmt.where(Camera.is_active == is_active)
    return (await session.execute(stmt)).scalars().all()


@router.get("/{camera_id}", response_model=CameraOut)
async def get_camera(camera_id: uuid.UUID, session: Session, _: Viewer):
    camera = await session.get(Camera, camera_id)
    if camera is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Camera not found")
    return camera


@router.post("", response_model=CameraOut, status_code=status.HTTP_201_CREATED)
async def create_camera(payload: CameraCreate, session: Session, _: ListEditor):
    camera = Camera(**payload.model_dump())
    session.add(camera)
    await session.commit()
    await session.refresh(camera)
    return camera


@router.patch("/{camera_id}", response_model=CameraOut)
async def update_camera(
    camera_id: uuid.UUID, payload: CameraUpdate, session: Session, _: ListEditor
):
    camera = await session.get(Camera, camera_id)
    if camera is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Camera not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(camera, field, value)
    await session.commit()
    await session.refresh(camera)
    return camera
