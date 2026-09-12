import uuid

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from apps.event_worker.handlers import handle_plate_read
from db.models import Base, GateAction, GateEvent, ListType, Vehicle
from plate_types import VANITY


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        for table in ("vehicles", "gate_events"):
            await conn.run_sync(Base.metadata.tables[table].create)

    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        s.add(Vehicle(plate_text="2D-0888", list_type=ListType.whitelist))
        s.add(Vehicle(plate_text="3A-1111", list_type=ListType.blacklist))
        s.add(Vehicle(plate_text="4B-2222", list_type=ListType.neutral))
        await s.commit()
        yield s
    await engine.dispose()


def payload(**overrides) -> dict:
    base = {
        "plate_read_id": str(uuid.uuid4()),
        "plate_text": "2D-0888",
        "confidence": 0.95,
        "is_valid": True,
    }
    base.update(overrides)
    return base


@pytest.mark.asyncio
async def test_whitelisted_plate_opens(session):
    decision = await handle_plate_read(session, payload())
    assert decision.action is GateAction.open


@pytest.mark.asyncio
async def test_blacklisted_plate_denied(session):
    decision = await handle_plate_read(session, payload(plate_text="3A-1111"))
    assert decision.action is GateAction.deny
    assert "blacklist" in decision.reason


@pytest.mark.asyncio
async def test_registered_but_not_whitelisted_denied(session):
    decision = await handle_plate_read(session, payload(plate_text="4B-2222"))
    assert decision.action is GateAction.deny


@pytest.mark.asyncio
async def test_unknown_plate_denied(session):
    decision = await handle_plate_read(session, payload(plate_text="9Z-9999"))
    assert decision.action is GateAction.deny
    assert "not registered" in decision.reason


@pytest.mark.asyncio
async def test_low_confidence_denied_even_if_whitelisted(session):
    decision = await handle_plate_read(session, payload(confidence=0.40))
    assert decision.action is GateAction.deny
    assert "confidence" in decision.reason


@pytest.mark.asyncio
async def test_invalid_format_denied_even_if_whitelisted(session):
    decision = await handle_plate_read(session, payload(is_valid=False))
    assert decision.action is GateAction.deny
    assert "validation" in decision.reason


@pytest.mark.asyncio
async def test_empty_plate_denied(session):
    decision = await handle_plate_read(session, payload(plate_text=""))
    assert decision.action is GateAction.deny


@pytest.mark.asyncio
async def test_every_decision_writes_a_gate_event(session):
    from sqlalchemy import select

    await handle_plate_read(session, payload())
    await handle_plate_read(session, payload(plate_text="3A-1111"))
    await session.commit()

    events = (await session.execute(select(GateEvent))).scalars().all()
    assert len(events) == 2
    assert {e.action for e in events} == {GateAction.open, GateAction.deny}
    assert all(e.actor == "system:anpr" for e in events)
    assert all(e.reason for e in events)


@pytest.mark.asyncio
async def test_vanity_plate_denied_with_its_own_reason(session):
    """A VIP vanity plate must not open the gate, but must not look like a bad read.

    Fail-closed is deliberate: no format means no verified identity. The value of
    the distinct reason is operational -- it tells whoever is watching that a real
    vehicle is waiting for a manual decision.
    """
    decision = await handle_plate_read(
        session,
        payload(plate_text="កុល", is_valid=False, plate_type=VANITY),
    )

    assert decision.action is GateAction.deny
    assert decision.reason == "vanity plate - manual check required"


@pytest.mark.asyncio
async def test_ordinary_invalid_plate_keeps_the_format_reason(session):
    decision = await handle_plate_read(
        session, payload(plate_text="XX!!", is_valid=False, plate_type="private_car")
    )

    assert decision.action is GateAction.deny
    assert decision.reason == "plate failed format validation"


@pytest.mark.asyncio
async def test_vanity_plate_cannot_open_even_when_whitelisted(session):
    """Registering the text of a vanity plate must not create a way in.

    The plate text for these reads is whatever OCR made of Khmer script, so it is
    neither stable nor unique; matching on it would admit anything that produced
    the same garbage.
    """
    session.add(Vehicle(plate_text="កុល", list_type=ListType.whitelist))
    await session.commit()

    decision = await handle_plate_read(
        session,
        payload(plate_text="កុល", is_valid=False, plate_type=VANITY),
    )

    assert decision.action is GateAction.deny
