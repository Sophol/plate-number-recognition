import aio_pika
import structlog

log = structlog.get_logger()

EXCHANGE_NAME = "anpr.events"


class Broker:
    """Thin aio-pika wrapper with a durable topic exchange."""

    def __init__(self, url: str) -> None:
        self.url = url
        self._connection: aio_pika.abc.AbstractRobustConnection | None = None
        self._exchange: aio_pika.abc.AbstractExchange | None = None

    async def connect(self) -> None:
        # Robust connection reconnects on its own if the broker restarts.
        self._connection = await aio_pika.connect_robust(self.url)
        channel = await self._connection.channel()
        self._exchange = await channel.declare_exchange(
            EXCHANGE_NAME, aio_pika.ExchangeType.TOPIC, durable=True
        )
        log.info("broker_connected", exchange=EXCHANGE_NAME)

    async def publish(self, routing_key: str, body: str) -> None:
        if self._exchange is None:
            raise RuntimeError("connect() must be called before publish()")
        await self._exchange.publish(
            aio_pika.Message(
                body=body.encode(),
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                content_type="application/json",
            ),
            routing_key=routing_key,
        )

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            self._exchange = None
