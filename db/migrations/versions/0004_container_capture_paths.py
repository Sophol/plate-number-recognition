"""container read evidence paths

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-12
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("container_reads", sa.Column("crop_path", sa.Text()))
    op.add_column("container_reads", sa.Column("frame_path", sa.Text()))


def downgrade() -> None:
    op.drop_column("container_reads", "frame_path")
    op.drop_column("container_reads", "crop_path")
