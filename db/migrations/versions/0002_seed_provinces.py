"""seed Cambodian provinces

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-11
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Plate prefix codes 1-25 follow the standard Cambodian province numbering.
PROVINCES = [
    ("1", "បន្ទាយមានជ័យ", "Banteay Meanchey"),
    ("2", "បាត់ដំបង", "Battambang"),
    ("3", "កំពង់ចាម", "Kampong Cham"),
    ("4", "កំពង់ឆ្នាំង", "Kampong Chhnang"),
    ("5", "កំពង់ស្ពឺ", "Kampong Speu"),
    ("6", "កំពង់ធំ", "Kampong Thom"),
    ("7", "កំពត", "Kampot"),
    ("8", "កណ្ដាល", "Kandal"),
    ("9", "កោះកុង", "Koh Kong"),
    ("10", "ក្រចេះ", "Kratie"),
    ("11", "មណ្ឌលគិរី", "Mondulkiri"),
    ("12", "ភ្នំពេញ", "Phnom Penh"),
    ("13", "ព្រះវិហារ", "Preah Vihear"),
    ("14", "ព្រៃវែង", "Prey Veng"),
    ("15", "ពោធិ៍សាត់", "Pursat"),
    ("16", "រតនគិរី", "Ratanakiri"),
    ("17", "សៀមរាប", "Siem Reap"),
    ("18", "ព្រះសីហនុ", "Preah Sihanouk"),
    ("19", "ស្ទឹងត្រែង", "Stung Treng"),
    ("20", "ស្វាយរៀង", "Svay Rieng"),
    ("21", "តាកែវ", "Takeo"),
    ("22", "ឧត្ដរមានជ័យ", "Oddar Meanchey"),
    ("23", "កែប", "Kep"),
    ("24", "ប៉ៃលិន", "Pailin"),
    ("25", "ត្បូងឃ្មុំ", "Tboung Khmum"),
]


def upgrade() -> None:
    provinces = sa.table(
        "provinces",
        sa.column("code", sa.String),
        sa.column("name_km", sa.String),
        sa.column("name_en", sa.String),
    )
    op.bulk_insert(
        provinces,
        [{"code": c, "name_km": km, "name_en": en} for c, km, en in PROVINCES],
    )


def downgrade() -> None:
    codes = tuple(c for c, _, _ in PROVINCES)
    op.execute(sa.text("DELETE FROM provinces WHERE code IN :codes").bindparams(codes=codes))
