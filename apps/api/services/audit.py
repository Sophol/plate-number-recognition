from sqlalchemy.ext.asyncio import AsyncSession

from db.models import AuditLog


async def record(
    session: AsyncSession,
    *,
    actor: str,
    action: str,
    entity: str,
    entity_id: str | None = None,
    before: dict | None = None,
    after: dict | None = None,
    reason: str | None = None,
) -> None:
    """Append an audit entry. Caller owns the surrounding transaction."""
    session.add(
        AuditLog(
            actor=actor,
            action=action,
            entity=entity,
            entity_id=entity_id,
            before=before,
            after=after,
            reason=reason,
        )
    )
