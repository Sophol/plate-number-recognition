import asyncio
import json

import aio_pika
import structlog

from apps.event_worker import outbox_relay
from apps.event_worker.handlers import handle_plate_read
from apps.event_worker.rabbitmq import EXCHANGE_NAME, Broker
from config import get_settings
from db.session import SessionLocal

log = structlog.get_logger()

PLATE_READ_ROUTING_KEY = "plate_read"
QUEUE_NAME = "anpr.gate"


async def consume_gate_events(broker: Broker) -> None:
    connection = await aio_pika.connect_robust(get_settings().rabbitmq_url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=16)

    exchange = await channel.declare_exchange(
        EXCHANGE_NAME, aio_pika.ExchangeType.TOPIC, durable=True
    )
    queue = await channel.declare_queue(QUEUE_NAME, durable=True)
    await queue.bind(exchange, routing_key=PLATE_READ_ROUTING_KEY)

    async with queue.iterator() as messages:
        async for message in messages:
            # Requeue on failure rather than acking, so a transient DB error
            # does not silently drop a gate decision.
            async with message.process(requeue=True):
                payload = json.loads(message.body.decode())
                async with SessionLocal() as session:
                    await handle_plate_read(session, payload)
                    await session.commit()


async def main() -> None:
    settings = get_settings()
    broker = Broker(settings.rabbitmq_url)
    await broker.connect()

    log.info("event_worker_started")
    try:
        await asyncio.gather(
            outbox_relay.run(broker.publish),
            consume_gate_events(broker),
        )
    finally:
        await broker.close()


if __name__ == "__main__":
    asyncio.run(main())
