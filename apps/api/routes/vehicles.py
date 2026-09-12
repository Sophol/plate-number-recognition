import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.schemas import VehicleCreate, VehicleOut, VehicleUpdate
from apps.api.security import ListEditor, Viewer
from apps.api.services import audit
from db.models import ListType, Vehicle
from db.session import get_session

router = APIRouter(prefix="/vehicles", tags=["vehicles"])
Session = Annotated[AsyncSession, Depends(get_session)]


def _snapshot(vehicle: Vehicle) -> dict:
    return {
        "plate_text": vehicle.plate_text,
        "owner_name": vehicle.owner_name,
        "phone": vehicle.phone,
        "list_type": vehicle.list_type.value,
        "notes": vehicle.notes,
    }


@router.get("", response_model=list[VehicleOut])
async def list_vehicles(
    session: Session,
    _: Viewer,
    list_type: ListType | None = None,
    plate_text: str | None = None,
    limit: Annotated[int, Query(le=500)] = 100,
    offset: int = 0,
):
    stmt = select(Vehicle).order_by(Vehicle.plate_text).limit(limit).offset(offset)
    if list_type is not None:
        stmt = stmt.where(Vehicle.list_type == list_type)
    if plate_text is not None:
        stmt = stmt.where(Vehicle.plate_text.ilike(f"%{plate_text}%"))
    return (await session.execute(stmt)).scalars().all()


@router.get("/{vehicle_id}", response_model=VehicleOut)
async def get_vehicle(vehicle_id: uuid.UUID, session: Session, _: Viewer):
    vehicle = await session.get(Vehicle, vehicle_id)
    if vehicle is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Vehicle not found")
    return vehicle


@router.post("", response_model=VehicleOut, status_code=status.HTTP_201_CREATED)
async def create_vehicle(payload: VehicleCreate, session: Session, user: ListEditor):
    existing = await session.execute(
        select(Vehicle).where(Vehicle.plate_text == payload.plate_text)
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "Vehicle with this plate already exists")

    vehicle = Vehicle(**payload.model_dump())
    session.add(vehicle)
    await session.flush()
    await audit.record(
        session,
        actor=user.username,
        action="create",
        entity="vehicle",
        entity_id=str(vehicle.id),
        after=_snapshot(vehicle),
    )
    await session.commit()
    await session.refresh(vehicle)
    return vehicle


@router.patch("/{vehicle_id}", response_model=VehicleOut)
async def update_vehicle(
    vehicle_id: uuid.UUID, payload: VehicleUpdate, session: Session, user: ListEditor
):
    vehicle = await session.get(Vehicle, vehicle_id)
    if vehicle is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Vehicle not found")

    before = _snapshot(vehicle)
    changes = payload.model_dump(exclude_unset=True, exclude={"reason"})
    for field, value in changes.items():
        setattr(vehicle, field, value)
    await session.flush()

    await audit.record(
        session,
        actor=user.username,
        action="update",
        entity="vehicle",
        entity_id=str(vehicle.id),
        before=before,
        after=_snapshot(vehicle),
        reason=payload.reason,
    )
    await session.commit()
    await session.refresh(vehicle)
    return vehicle


@router.delete("/{vehicle_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_vehicle(vehicle_id: uuid.UUID, session: Session, user: ListEditor):
    vehicle = await session.get(Vehicle, vehicle_id)
    if vehicle is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Vehicle not found")

    await audit.record(
        session,
        actor=user.username,
        action="delete",
        entity="vehicle",
        entity_id=str(vehicle.id),
        before=_snapshot(vehicle),
    )
    await session.delete(vehicle)
    await session.commit()
