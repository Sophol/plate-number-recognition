import enum
import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Postgres is the production target; the generic JSON fallback lets the suite run
# against SQLite without a live server.
JsonType = JSONB().with_variant(JSON(), "sqlite")


class Base(DeclarativeBase):
    pass


class ListType(str, enum.Enum):
    whitelist = "whitelist"
    blacklist = "blacklist"
    neutral = "neutral"


class GateAction(str, enum.Enum):
    open = "open"
    deny = "deny"
    manual_override = "manual_override"


class Province(Base):
    __tablename__ = "provinces"

    code: Mapped[str] = mapped_column(String(8), primary_key=True)
    name_km: Mapped[str] = mapped_column(String(128))
    name_en: Mapped[str] = mapped_column(String(128))


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    rtsp_url: Mapped[str] = mapped_column(Text)
    site_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    direction: Mapped[str | None] = mapped_column(String(32), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    plate_reads: Mapped[list["PlateRead"]] = relationship(back_populates="camera")


class PlateRead(Base):
    __tablename__ = "plate_reads"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    camera_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"))
    # Groups frames belonging to one vehicle transit so voting and dedup are
    # scoped to a single vehicle rather than a time window.
    track_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    plate_text: Mapped[str] = mapped_column(String(32))
    province_code: Mapped[str | None] = mapped_column(
        ForeignKey("provinces.code", ondelete="SET NULL"), nullable=True
    )
    plate_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    vehicle_type: Mapped[str | None] = mapped_column(String(32), nullable=True)

    confidence: Mapped[float] = mapped_column(Float)
    detector_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    province_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Part of the primary key: Postgres requires the partition key in the PK of a
    # partitioned table. Lookups by id alone must therefore filter on frame_ts too.
    frame_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True, index=True
    )
    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    plate_crop_path: Mapped[str | None] = mapped_column(Text, nullable=True)

    model_version: Mapped[str] = mapped_column(String(64))
    # False for shadow-scored candidate models, which must not drive gates.
    is_active_model: Mapped[bool] = mapped_column(Boolean, default=True)
    is_valid: Mapped[bool] = mapped_column(Boolean, default=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    camera: Mapped["Camera"] = relationship(back_populates="plate_reads")

    __table_args__ = (
        Index("ix_plate_reads_plate_text", "plate_text"),
        Index("ix_plate_reads_camera_frame_ts", "camera_id", "frame_ts"),
        Index("ix_plate_reads_track_id", "track_id"),
        Index(
            "ix_plate_reads_plate_text_trgm",
            "plate_text",
            postgresql_using="gin",
            postgresql_ops={"plate_text": "gin_trgm_ops"},
        ),
    )


class Vehicle(Base):
    __tablename__ = "vehicles"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    plate_text: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    owner_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(32), nullable=True)
    list_type: Mapped[ListType] = mapped_column(
        Enum(ListType, name="list_type"), default=ListType.neutral
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class GateEvent(Base):
    __tablename__ = "gate_events"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    plate_read_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("plate_reads.id", ondelete="SET NULL"), nullable=True
    )
    action: Mapped[GateAction] = mapped_column(Enum(GateAction, name="gate_action"))
    actor: Mapped[str] = mapped_column(String(128))
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    """Records privileged mutations (list edits, gate overrides) for review."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(128), index=True)
    action: Mapped[str] = mapped_column(String(64))
    entity: Mapped[str] = mapped_column(String(64))
    entity_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    before: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    after: Mapped[dict | None] = mapped_column(JsonType, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class Outbox(Base):
    """Transactional outbox; the relay publishes unprocessed rows to the broker."""

    __tablename__ = "outbox"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    aggregate: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JsonType)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (Index("ix_outbox_unprocessed", "processed_at", "id"),)


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(256))
    # viewer | list_editor | gate_operator | admin
    role: Mapped[str] = mapped_column(String(32), default="viewer")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
