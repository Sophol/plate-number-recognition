"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-09-11
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    op.create_table(
        "provinces",
        sa.Column("code", sa.String(8), primary_key=True),
        sa.Column("name_km", sa.String(128), nullable=False),
        sa.Column("name_en", sa.String(128), nullable=False),
    )

    op.create_table(
        "cameras",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("rtsp_url", sa.Text(), nullable=False),
        sa.Column("site_id", sa.String(64)),
        sa.Column("direction", sa.String(32)),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("username", sa.String(64), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(256), nullable=False),
        sa.Column("role", sa.String(32), nullable=False, server_default="viewer"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_users_username", "users", ["username"])

    op.create_table(
        "vehicles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("plate_text", sa.String(32), nullable=False, unique=True),
        sa.Column("owner_name", sa.String(128)),
        sa.Column("phone", sa.String(32)),
        sa.Column(
            "list_type",
            sa.Enum("whitelist", "blacklist", "neutral", name="list_type"),
            nullable=False,
            server_default="neutral",
        ),
        sa.Column("notes", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_vehicles_plate_text", "vehicles", ["plate_text"])

    # Partitioned by month on frame_ts; the PK must include the partition key.
    op.execute(
        """
        CREATE TABLE plate_reads (
            id UUID NOT NULL,
            camera_id UUID NOT NULL REFERENCES cameras(id) ON DELETE CASCADE,
            track_id VARCHAR(64),
            plate_text VARCHAR(32) NOT NULL,
            province_code VARCHAR(8) REFERENCES provinces(code) ON DELETE SET NULL,
            plate_type VARCHAR(32),
            vehicle_type VARCHAR(32),
            confidence DOUBLE PRECISION NOT NULL,
            detector_confidence DOUBLE PRECISION,
            ocr_confidence DOUBLE PRECISION,
            province_confidence DOUBLE PRECISION,
            frame_ts TIMESTAMPTZ NOT NULL,
            image_path TEXT,
            plate_crop_path TEXT,
            model_version VARCHAR(64) NOT NULL,
            is_active_model BOOLEAN NOT NULL DEFAULT TRUE,
            is_valid BOOLEAN NOT NULL DEFAULT FALSE,
            is_verified BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (id, frame_ts)
        ) PARTITION BY RANGE (frame_ts)
        """
    )
    op.execute("CREATE INDEX ix_plate_reads_plate_text ON plate_reads (plate_text)")
    op.execute("CREATE INDEX ix_plate_reads_camera_frame_ts ON plate_reads (camera_id, frame_ts)")
    op.execute("CREATE INDEX ix_plate_reads_track_id ON plate_reads (track_id)")
    op.execute("CREATE INDEX ix_plate_reads_frame_ts ON plate_reads (frame_ts)")
    op.execute(
        "CREATE INDEX ix_plate_reads_plate_text_trgm "
        "ON plate_reads USING gin (plate_text gin_trgm_ops)"
    )
    # Catch-all so inserts never fail before a monthly partition is provisioned.
    op.execute("CREATE TABLE plate_reads_default PARTITION OF plate_reads DEFAULT")

    op.create_table(
        "gate_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("plate_read_id", postgresql.UUID(as_uuid=True)),
        sa.Column(
            "action",
            sa.Enum("open", "deny", "manual_override", name="gate_action"),
            nullable=False,
        ),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("reason", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("actor", sa.String(128), nullable=False),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("entity", sa.String(64), nullable=False),
        sa.Column("entity_id", sa.String(64)),
        sa.Column("before", postgresql.JSONB()),
        sa.Column("after", postgresql.JSONB()),
        sa.Column("reason", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_audit_log_actor", "audit_log", ["actor"])
    op.create_index("ix_audit_log_created_at", "audit_log", ["created_at"])

    op.create_table(
        "outbox",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("aggregate", sa.String(64), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_outbox_unprocessed", "outbox", ["processed_at", "id"])


def downgrade() -> None:
    op.drop_table("outbox")
    op.drop_table("audit_log")
    op.drop_table("gate_events")
    op.execute("DROP TABLE IF EXISTS plate_reads")
    op.drop_table("vehicles")
    op.drop_table("users")
    op.drop_table("cameras")
    op.drop_table("provinces")
    op.execute("DROP TYPE IF EXISTS list_type")
    op.execute("DROP TYPE IF EXISTS gate_action")
