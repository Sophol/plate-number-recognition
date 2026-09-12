import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import GateAction, GateEvent, ListType, Vehicle

log = structlog.get_logger()

# A read below this confidence never opens a gate, whatever the list says.
MIN_GATE_CONFIDENCE = 0.75


@dataclass(slots=True)
class GateDecision:
    action: GateAction
    reason: str
    plate_text: str
    vehicle_id: uuid.UUID | None = None


async def decide(session: AsyncSession, payload: dict) -> GateDecision:
    """Decide whether a plate read should open the gate.

    Fails closed: anything unrecognised, low-confidence, or format-invalid is
    denied rather than admitted. Only an explicit whitelist match opens.
    """
    plate_text = payload.get("plate_text", "")
    confidence = float(payload.get("confidence", 0.0))
    is_valid = bool(payload.get("is_valid", False))

    if not plate_text:
        return GateDecision(GateAction.deny, "empty plate text", plate_text)

    if not is_valid:
        return GateDecision(GateAction.deny, "plate failed format validation", plate_text)

    if confidence < MIN_GATE_CONFIDENCE:
        return GateDecision(
            GateAction.deny,
            f"confidence {confidence:.2f} below threshold {MIN_GATE_CONFIDENCE}",
            plate_text,
        )

    result = await session.execute(select(Vehicle).where(Vehicle.plate_text == plate_text))
    vehicle = result.scalar_one_or_none()

    if vehicle is None:
        return GateDecision(GateAction.deny, "plate not registered", plate_text)

    if vehicle.list_type is ListType.blacklist:
        return GateDecision(GateAction.deny, "plate is blacklisted", plate_text, vehicle.id)

    if vehicle.list_type is ListType.whitelist:
        return GateDecision(GateAction.open, "whitelisted", plate_text, vehicle.id)

    return GateDecision(GateAction.deny, "plate registered but not whitelisted", plate_text, vehicle.id)


async def handle_plate_read(session: AsyncSession, payload: dict) -> GateDecision:
    """Decide and record a gate event. Caller owns the commit."""
    decision = await decide(session, payload)

    plate_read_id = payload.get("plate_read_id")
    session.add(
        GateEvent(
            plate_read_id=uuid.UUID(plate_read_id) if plate_read_id else None,
            action=decision.action,
            actor="system:anpr",
            reason=decision.reason,
        )
    )
    log.info(
        "gate_decision",
        action=decision.action.value,
        plate_text=decision.plate_text,
        reason=decision.reason,
    )
    return decision
