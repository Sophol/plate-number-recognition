import asyncio
import json

import structlog
from sqlalchemy import func, select

from db.models import Outbox
from db.session import SessionLocal

log = structlog.get_logger()

POLL_INTERVAL_SECONDS = 1.0
BATCH_SIZE = 100


async def relay_once(publish) -> int:
    """Publish unprocessed outbox rows, marking them processed in the same transaction.

    SKIP LOCKED lets multiple relay instances run without double-publishing.
    """
    async with SessionLocal() as session:
        stmt = (
            select(Outbox)
            .where(Outbox.processed_at.is_(None))
            .order_by(Outbox.id)
            .limit(BATCH_SIZE)
            .with_for_update(skip_locked=True)
        )
        rows = (await session.execute(stmt)).scalars().all()
        if not rows:
            return 0

        for row in rows:
            await publish(row.aggregate, json.dumps(row.payload))
            row.processed_at = func.now()

        await session.commit()
        return len(rows)


async def run(publish) -> None:
    while True:
        try:
            count = await relay_once(publish)
        except Exception:
            log.exception("outbox_relay_failed")
            count = 0
        if count == 0:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
