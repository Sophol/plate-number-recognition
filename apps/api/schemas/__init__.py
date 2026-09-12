import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from db.models import GateAction, ListType


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# A blank name renders as an unidentifiable row in the UI, so require one
# here rather than letting whitespace through. strip_whitespace makes "  "
# fail min_length instead of being stored as-is.
CameraName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)
]
RtspUrl = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CameraCreate(BaseModel):
    name: CameraName
    rtsp_url: RtspUrl
    site_id: str | None = None
    direction: str | None = None
    is_active: bool = True


class CameraUpdate(BaseModel):
    name: CameraName | None = None
    rtsp_url: RtspUrl | None = None
    site_id: str | None = None
    direction: str | None = None
    is_active: bool | None = None


class CameraOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    rtsp_url: str
    site_id: str | None
    direction: str | None
    is_active: bool
    created_at: datetime


class PlateReadCreate(BaseModel):
    camera_id: uuid.UUID
    track_id: str | None = None
    plate_text: str
    province_code: str | None = None
    plate_type: str | None = None
    vehicle_type: str | None = None
    confidence: float
    detector_confidence: float | None = None
    ocr_confidence: float | None = None
    province_confidence: float | None = None
    frame_ts: datetime
    image_path: str | None = None
    plate_crop_path: str | None = None
    model_version: str
    is_active_model: bool = True
    is_valid: bool = False


class PlateReadOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    camera_id: uuid.UUID
    track_id: str | None
    plate_text: str
    province_code: str | None
    plate_type: str | None
    vehicle_type: str | None
    confidence: float
    detector_confidence: float | None
    ocr_confidence: float | None
    province_confidence: float | None
    frame_ts: datetime
    image_path: str | None
    plate_crop_path: str | None
    model_version: str
    is_active_model: bool
    is_valid: bool
    is_verified: bool
    created_at: datetime


class ContainerReadCreate(BaseModel):
    camera_id: uuid.UUID
    track_id: str | None = None
    container_number: Annotated[str, StringConstraints(min_length=11, max_length=11, to_upper=True)]
    confidence: float
    frame_ts: datetime
    model_version: str
    ocr_text: str | None = None
    plate_read_id: uuid.UUID | None = None


class ContainerReadOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    camera_id: uuid.UUID
    track_id: str | None
    container_number: str
    owner_code: str
    checksum_ok: bool
    is_known: bool | None
    ocr_text: str | None
    was_snapped: bool
    confidence: float
    frame_ts: datetime
    model_version: str
    plate_read_id: uuid.UUID | None
    created_at: datetime


class ContainerLookupOut(BaseModel):
    """What the system knows about one container number."""

    number: str
    well_formed: bool
    checksum_ok: bool
    owner_code: str | None
    is_known: bool | None      # None when the PAS list is not loaded on this server
    read_count: int
    last_seen: datetime | None
    recent_reads: list[ContainerReadOut]


class VehicleCreate(BaseModel):
    plate_text: str = Field(max_length=32)
    owner_name: str | None = None
    phone: str | None = None
    list_type: ListType = ListType.neutral
    notes: str | None = None


class VehicleUpdate(BaseModel):
    owner_name: str | None = None
    phone: str | None = None
    list_type: ListType | None = None
    notes: str | None = None
    reason: str | None = Field(default=None, description="Recorded in the audit log")


class VehicleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    plate_text: str
    owner_name: str | None
    phone: str | None
    list_type: ListType
    notes: str | None
    created_at: datetime
    updated_at: datetime


class GateEventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    plate_read_id: uuid.UUID | None
    action: GateAction
    actor: str
    reason: str | None
    created_at: datetime


class ProvinceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    code: str
    name_km: str
    name_en: str
