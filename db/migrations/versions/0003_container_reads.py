"""container reads

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-12
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "container_reads",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("camera_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False),
        sa.Column("track_id", sa.String(64)),
        sa.Column("container_number", sa.String(11), nullable=False),
        sa.Column("owner_code", sa.String(4), nullable=False),
        sa.Column("checksum_ok", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("is_known", sa.Boolean()),
        sa.Column("ocr_text", sa.String(32)),
        sa.Column("was_snapped", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("frame_ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        # No foreign key: plate_reads is partitioned with the composite key
        # (id, frame_ts), so Postgres cannot reference id alone. Same choice as
        # gate_events in 0001; the ORM keeps the relationship for SQLite.
        sa.Column("plate_read_id", postgresql.UUID(as_uuid=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_container_reads_number", "container_reads", ["container_number"])
    op.create_index("ix_container_reads_camera_frame_ts", "container_reads", ["camera_id", "frame_ts"])
    op.create_index("ix_container_reads_track_id", "container_reads", ["track_id"])
    op.create_index("ix_container_reads_frame_ts", "container_reads", ["frame_ts"])


def downgrade() -> None:
    op.drop_table("container_reads")
