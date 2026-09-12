"""End-to-end test against real PostgreSQL.

Skipped automatically when TEST_DATABASE_URL is unset, so the suite still runs
on machines without a database. Exercises partitioning, JSONB, the outbox
relay, and the gate decision path -- none of which SQLite can verify.
"""
import json
import os
import uuid
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from apps.event_worker import outbox_relay
from apps.event_worker.handlers import handle_plate_read
from apps.inference_worker.main import persist
from apps.inference_worker.pipeline import CommittedRead
from db.models import (
    Camera,
    GateAction,
    GateEvent,
    ListType,
    Outbox,
    PlateRead,
    Vehicle,
)

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set; skipping Postgres integration"
)

CAMERA_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine(TEST_DATABASE_URL)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    async with maker() as s:
        await s.execute(text("DELETE FROM gate_events"))
        await s.execute(text("DELETE FROM outbox"))
        await s.execute(text("DELETE FROM plate_reads"))
        await s.execute(text("DELETE FROM vehicles"))
        await s.execute(text("DELETE FROM cameras"))
        s.add(Camera(id=CAMERA_ID, name="gate-1", rtsp_url="rtsp://x", is_active=True))
        await s.commit()
        yield s
    await engine.dispose()


def committed_read(plate_text="2D-0888", confidence=0.95, is_valid=True) -> CommittedRead:
    return CommittedRead(
        id=uuid.uuid4(),
        camera_id=str(CAMERA_ID),
        track_id="t1",
        plate_text=plate_text,
        province_code="2",
        plate_type="private_car",
        confidence=confidence,
        detector_confidence=0.9,
        ocr_confidence=0.88,
        province_confidence=0.9,
        frame_ts=datetime(2026, 9, 11, 10, 0, tzinfo=UTC),
        model_version="test-v1",
        is_valid=is_valid,
    )


@pytest.mark.asyncio
async def test_persist_writes_read_and_outbox_atomically(session):
    read = committed_read()
    await persist(session, read)
    await session.commit()

    stored = (await session.execute(select(PlateRead))).scalars().all()
    assert len(stored) == 1
    assert stored[0].plate_text == "2D-0888"

    events = (await session.execute(select(Outbox))).scalars().all()
    assert len(events) == 1
    assert events[0].aggregate == "plate_read"
    # JSONB round-trip through real Postgres.
    assert events[0].payload["plate_text"] == "2D-0888"
    assert events[0].processed_at is None


@pytest.mark.asyncio
async def test_read_routes_to_monthly_partition(session):
    await persist(session, committed_read())
    await session.commit()

    partition = await session.execute(
        text("SELECT tableoid::regclass::text FROM plate_reads LIMIT 1")
    )
    # Provisioned in migration verification; falls back to DEFAULT otherwise.
    assert partition.scalar_one() in ("plate_reads_2026_09", "plate_reads_default")


@pytest.mark.asyncio
async def test_trigram_index_finds_misread_plate(session):
    await persist(session, committed_read())
    await session.commit()

    result = await session.execute(
        text("SELECT plate_text FROM plate_reads WHERE plate_text % :q"), {"q": "2D-0889"}
    )
    assert result.scalar_one() == "2D-0888"


@pytest.mark.asyncio
async def test_outbox_relay_publishes_and_marks_processed(session, monkeypatch):
    await persist(session, committed_read())
    await session.commit()

    published: list[tuple[str, str]] = []

    async def fake_publish(routing_key: str, body: str) -> None:
        published.append((routing_key, body))

    engine = create_async_engine(TEST_DATABASE_URL)
    monkeypatch.setattr(
        outbox_relay, "SessionLocal", async_sessionmaker(engine, expire_on_commit=False)
    )

    count = await outbox_relay.relay_once(fake_publish)
    assert count == 1
    assert published[0][0] == "plate_read"
    assert json.loads(published[0][1])["plate_text"] == "2D-0888"

    # A second pass must publish nothing; rows are marked processed.
    assert await outbox_relay.relay_once(fake_publish) == 0
    await engine.dispose()


@pytest.mark.asyncio
async def test_full_flow_read_to_gate_decision(session):
    session.add(Vehicle(plate_text="2D-0888", list_type=ListType.whitelist))
    await session.commit()

    read = committed_read()
    await persist(session, read)
    await session.commit()

    outbox_row = (await session.execute(select(Outbox))).scalars().one()
    decision = await handle_plate_read(session, outbox_row.payload)
    await session.commit()

    assert decision.action is GateAction.open
    events = (await session.execute(select(GateEvent))).scalars().all()
    assert len(events) == 1
    assert events[0].plate_read_id == read.id


@pytest.mark.asyncio
async def test_full_flow_denies_unregistered_plate(session):
    await persist(session, committed_read(plate_text="9Z-9999"))
    await session.commit()

    outbox_row = (await session.execute(select(Outbox))).scalars().one()
    decision = await handle_plate_read(session, outbox_row.payload)
    await session.commit()

    assert decision.action is GateAction.deny
    assert "not registered" in decision.reason
