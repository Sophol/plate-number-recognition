"""End-to-end demo: synthetic frames -> inference -> Postgres -> gate decision.

Usage: python -m scripts.demo_pipeline
Requires a database with migrations applied (see README).
"""
import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import cv2
import numpy as np
from sqlalchemy import delete, select

from apps.camera_worker.sampler import Frame
from apps.event_worker.handlers import handle_plate_read
from apps.event_worker.outbox_relay import relay_once
from apps.inference_worker.interfaces import Detection, OCRResult
from apps.inference_worker.main import persist
from apps.inference_worker.pipeline import InferencePipeline
from apps.inference_worker.province import EnglishZoneProvinceClassifier
from config import get_settings
from db.models import Camera, GateEvent, ListType, Outbox, PlateRead, Vehicle
from db.session import SessionLocal

CAMERA_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
PLATE = "2D-0888"


def scene() -> np.ndarray:
    img = np.full((480, 640, 3), 70, dtype=np.uint8)
    cv2.rectangle(img, (200, 200), (440, 290), (245, 245, 245), -1)
    cv2.putText(img, PLATE, (215, 262), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (15, 15, 15), 4)
    return img


class FixedDetector:
    """Stands in for YOLO so the demo is deterministic."""

    def detect(self, image):
        return [Detection(x1=200, y1=200, x2=440, y2=290, confidence=0.92)]


class FixedOCR:
    def read(self, plate_image):
        return OCRResult(text=PLATE, confidence=0.91)


async def reset() -> None:
    async with SessionLocal() as session:
        await session.execute(delete(GateEvent))
        await session.execute(delete(Outbox))
        await session.execute(delete(PlateRead))
        await session.execute(delete(Vehicle))
        await session.execute(delete(Camera).where(Camera.id == CAMERA_ID))
        session.add(Camera(id=CAMERA_ID, name="demo-gate", rtsp_url="rtsp://demo"))
        session.add(Vehicle(plate_text=PLATE, owner_name="Demo", list_type=ListType.whitelist))
        await session.commit()


async def main() -> None:
    settings = get_settings()
    await reset()

    ocr = FixedOCR()
    pipeline = InferencePipeline(
        detector=FixedDetector(),
        ocr=ocr,
        province_classifier=EnglishZoneProvinceClassifier(ocr),
        model_version=settings.model_version,
        votes_required=3,
    )

    print("1. Feeding 5 frames of one vehicle through the pipeline...")
    base = datetime.now(UTC)
    committed = []
    for i in range(5):
        frame = Frame(camera_id=str(CAMERA_ID), image=scene(), frame_ts=base + timedelta(seconds=i * 0.1))
        results = pipeline.process(frame)
        print(f"   frame {i + 1}: {'COMMITTED ' + results[0].plate_text if results else 'voting...'}")
        committed.extend(results)

    if not committed:
        print("   no read reached quorum")
        return

    print("\n2. Persisting read + outbox row in one transaction...")
    async with SessionLocal() as session:
        for read in committed:
            await persist(session, read)
        await session.commit()

    async with SessionLocal() as session:
        stored = (await session.execute(select(PlateRead))).scalars().all()
        print(f"   plate_reads rows: {len(stored)} -> {stored[0].plate_text} (valid={stored[0].is_valid})")

    print("\n3. Relaying outbox to broker...")
    published = []

    async def fake_publish(routing_key, body):
        published.append((routing_key, body))

    count = await relay_once(fake_publish)
    print(f"   published {count} message(s), routing_key={published[0][0] if published else None}")

    print("\n4. Gate decision...")
    async with SessionLocal() as session:
        row = (await session.execute(select(Outbox))).scalars().first()
        decision = await handle_plate_read(session, row.payload)
        await session.commit()
        print(f"   action={decision.action.value.upper()} reason={decision.reason!r}")

    async with SessionLocal() as session:
        events = (await session.execute(select(GateEvent))).scalars().all()
        print(f"\n   gate_events rows: {len(events)}")
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
