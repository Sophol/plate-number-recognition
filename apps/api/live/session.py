"""One live preview per camera, shared by every viewer watching it.

The camera and inference workers keep their frames in an in-process asyncio
queue, so the API cannot observe them. This opens its own short-lived RTSP
read purely for preview and runs the same pipeline stages to show what the
detector and OCR see. It never writes plate_reads -- the inference worker
remains the only writer.
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import cv2
import numpy as np
import structlog

from apps.api.live.annotate import _CYAN, _GREY, draw_box, draw_detection, draw_status, placeholder
from apps.api.live.devicesource import DeviceSource, parse_device
from apps.api.live.filesource import LoopingFileSource
from apps.api.live.probe import probe
from apps.camera_worker.rtsp import RTSPSource
from apps.inference_worker.main import (
    build_container_reader, build_detector, build_ocr, build_vehicle_detector,
)
from apps.inference_worker.perspective import split_zones, warp_plate
from apps.inference_worker.province import EnglishZoneProvinceClassifier
from apps.inference_worker.validator import correct, validate
from config import get_settings

log = structlog.get_logger()

JPEG_QUALITY = 72
PREVIEW_FPS = 6.0
# Stop reading the camera once the last viewer has been gone this long, so a
# closed browser tab does not hold an RTSP connection open indefinitely.
IDLE_SHUTDOWN_SECONDS = 10.0
PLACEHOLDER_SIZE = (960, 540)


@dataclass
class StageEvent:
    """Per-detection trace of how one plate was read, for the explainer panel."""

    camera_id: str
    at: str
    detector_confidence: float
    ocr_text: str
    ocr_confidence: float
    corrected_text: str
    province_code: str | None
    province_confidence: float | None
    plate_type: str
    is_valid: bool
    warped: bool
    vehicle_type: str | None = None
    vehicle_colour: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "plate",
            "vehicle_type": self.vehicle_type,
            "vehicle_colour": self.vehicle_colour,
            "camera_id": self.camera_id,
            "at": self.at,
            "detector_confidence": round(self.detector_confidence, 3),
            "ocr_text": self.ocr_text,
            "ocr_confidence": round(self.ocr_confidence, 3),
            "corrected_text": self.corrected_text,
            "province_code": self.province_code,
            "province_confidence": (
                round(self.province_confidence, 3) if self.province_confidence is not None else None
            ),
            "plate_type": self.plate_type,
            "is_valid": self.is_valid,
            "warped": self.warped,
        }


@dataclass
class ContainerEvent:
    """One container number read from a frame, with its PAS status."""

    camera_id: str
    at: str
    container_number: str
    confidence: float
    is_known: bool | None
    was_snapped: bool
    ocr_text: str
    vehicle_type: str | None = None
    vehicle_colour: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "container",
            "camera_id": self.camera_id,
            "at": self.at,
            "container_number": self.container_number,
            "container_confidence": round(self.confidence, 3),
            "container_known": self.is_known,
            "container_snapped": self.was_snapped,
            "container_ocr_text": self.ocr_text,
            "vehicle_type": self.vehicle_type,
            "vehicle_colour": self.vehicle_colour,
        }


class PreviewSession:
    """Reads one camera, annotates frames, and fans them out to viewers."""

    def __init__(self, camera_id: str, camera_name: str, rtsp_url: str) -> None:
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url

        self._frame: bytes | None = None
        self._frame_ready = asyncio.Event()
        self._events: list[dict[str, Any]] = []
        self._task: asyncio.Task | None = None
        self._viewers = 0
        self._last_seen = time.monotonic()
        self._fps = 0.0
        self._connected = False
        self._detail = "starting"
        self._attempts = 0

        # Build the same backends the inference worker uses, so the preview
        # shows what the deployed pipeline actually sees -- not the classical-CV
        # fallback. A preview drawing no box while the trained detector fires on
        # every frame is worse than no preview: it reads as "detection is
        # broken" when it is working.
        settings = get_settings()
        self._detector = build_detector(settings.detector_backend, settings.detector_model_path)
        self._ocr = build_ocr(settings.ocr_backend, settings.ocr_use_gpu)
        self._province = EnglishZoneProvinceClassifier(self._ocr)
        # Both are None when disabled in settings or their model is missing.
        self._vehicles = build_vehicle_detector(settings)
        self._containers = build_container_reader(settings)

    # --- viewer bookkeeping -------------------------------------------------

    def acquire(self) -> None:
        self._viewers += 1
        self._last_seen = time.monotonic()
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    def release(self) -> None:
        self._viewers = max(0, self._viewers - 1)
        self._last_seen = time.monotonic()

    @property
    def status(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.camera_name,
            "connected": self._connected,
            "fps": round(self._fps, 1),
            "viewers": self._viewers,
            "detail": self._detail,
            "attempts": self._attempts,
        }

    def recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._events[-limit:][::-1]

    # --- frame access -------------------------------------------------------

    async def next_frame(self, timeout: float = 5.0) -> bytes:
        """Latest annotated JPEG. Falls back to a placeholder if nothing arrives."""
        self._last_seen = time.monotonic()
        try:
            await asyncio.wait_for(self._frame_ready.wait(), timeout)
        except TimeoutError:
            pass
        self._frame_ready.clear()
        if self._frame is None:
            return _encode(placeholder(*PLACEHOLDER_SIZE, "connecting to camera..."))
        return self._frame

    # --- capture loop -------------------------------------------------------

    async def _run(self) -> None:
        log.info("preview_started", camera_id=self.camera_id, url=self.rtsp_url)

        # One DESCRIBE up front turns an endless silent retry into a clear
        # message: bad credentials, dead host, or wrong path.
        self._detail = "checking stream..."
        self._publish(placeholder(*PLACEHOLDER_SIZE, f"checking {self.camera_name}..."))
        ok, reason = await asyncio.to_thread(probe, self.rtsp_url)
        self._detail = reason
        if not ok:
            log.warning("preview_unavailable", camera_id=self.camera_id, reason=reason)
            self._publish(placeholder(*PLACEHOLDER_SIZE, reason))
            self._connected = False
            return

        # Three source kinds: an attached camera, a local file (looped, since it
        # has a known end), or a network stream (RTSPSource reconnects forever).
        source: DeviceSource | LoopingFileSource | RTSPSource
        device_index = parse_device(self.rtsp_url)
        if device_index is not None:
            source = DeviceSource(device_index, fps=PREVIEW_FPS)
        elif os.path.exists(self.rtsp_url):
            source = LoopingFileSource(self.rtsp_url)
        else:
            source = RTSPSource(self.rtsp_url, self.camera_id)
        interval = 1.0 / PREVIEW_FPS
        last = 0.0
        ticks: list[float] = []

        self._publish(placeholder(*PLACEHOLDER_SIZE, f"connecting to {self.camera_name}..."))

        try:
            async for image, frame_ts in source.frames():
                self._connected = True
                self._detail = "streaming"
                now = time.monotonic()
                if now - last < interval:
                    continue
                last = now

                ticks.append(now)
                ticks[:] = [t for t in ticks if now - t <= 2.0]
                self._fps = len(ticks) / 2.0

                annotated = await asyncio.to_thread(self._analyse, image.copy(), frame_ts)
                self._publish(annotated)

                if self._viewers == 0 and now - self._last_seen > IDLE_SHUTDOWN_SECONDS:
                    log.info("preview_idle_stop", camera_id=self.camera_id)
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a dead camera must not take the API down
            log.warning("preview_failed", camera_id=self.camera_id, error=str(exc))
            self._publish(placeholder(*PLACEHOLDER_SIZE, f"stream error: {exc}"))
        finally:
            self._connected = False
            log.info("preview_stopped", camera_id=self.camera_id)

    def _analyse(self, image: np.ndarray, frame_ts: datetime) -> np.ndarray:
        """Runs detect -> warp -> OCR -> province on one frame (worker thread)."""
        # The vehicle + container chain is the expensive part; the preview runs
        # it on alternate frames, which is still more than once a second.
        self._chain_tick = getattr(self, "_chain_tick", 0) + 1
        run_chain = self._chain_tick % 2 == 0
        vehicles = self._vehicles.detect(image) if (self._vehicles is not None and run_chain) else []
        primary = vehicles[0] if vehicles else None            # largest first
        for v in vehicles:
            draw_box(image, v.x1, v.y1, v.x2, v.y2,
                     f"{v.colour} {v.vehicle_type}", _GREY, v.type_confidence)

        detections = self._detector.detect(image)

        for detection in detections:
            crop = image[detection.y1 : detection.y2, detection.x1 : detection.x2]
            if crop.size == 0:
                continue

            plate = warp_plate(crop, detection.corners, origin=(detection.x1, detection.y1))
            zones = split_zones(plate)
            reading = self._ocr.read(zones["middle"])

            if not reading.text:
                draw_detection(image, detection, "no text", detection.confidence)
                continue

            plate_type = detection.plate_type or "private_car"
            corrected = correct(reading.text, plate_type)
            is_valid = validate(corrected, plate_type)
            province = self._province.classify(zones["bottom"])

            draw_detection(image, detection, corrected, reading.confidence)
            self._record(
                StageEvent(
                    camera_id=self.camera_id,
                    at=datetime.now(UTC).isoformat(),
                    detector_confidence=detection.confidence,
                    ocr_text=reading.text,
                    ocr_confidence=reading.confidence,
                    corrected_text=corrected,
                    province_code=province.code,
                    province_confidence=province.confidence,
                    plate_type=plate_type,
                    is_valid=is_valid,
                    warped=detection.corners is not None,
                    vehicle_type=primary.vehicle_type if primary else None,
                    vehicle_colour=primary.colour if primary else None,
                )
            )

        container_line = "no container number"
        if self._containers is not None and run_chain:
            found = self._containers.read_frame(
                image, [(v.x1, v.y1, v.x2, v.y2) for v in vehicles]
            )
            if found is not None:
                r, reg = found.result, found.region
                status = ("known" if r.is_known else
                          "NOT in PAS" if r.is_known is False else "PAS n/a")
                draw_box(image, reg.x1, reg.y1, reg.x2, reg.y2,
                         f"{r.number.canonical} {status}", _CYAN, r.confidence)
                container_line = f"container {r.number.canonical} ({status})"
                self._record(
                    ContainerEvent(
                        camera_id=self.camera_id,
                        at=datetime.now(UTC).isoformat(),
                        container_number=r.number.canonical,
                        confidence=r.confidence,
                        is_known=r.is_known,
                        was_snapped=r.was_snapped,
                        ocr_text=r.text,
                        vehicle_type=primary.vehicle_type if primary else None,
                        vehicle_colour=primary.colour if primary else None,
                    )
                )

        draw_status(
            image,
            [
                f"{self.camera_name}  |  {self._fps:.1f} fps preview",
                f"{len(detections)} plate(s), {len(vehicles)} vehicle(s) in frame",
                container_line,
                frame_ts.strftime("%H:%M:%S"),
            ],
        )
        return image

    def _record(self, event: "StageEvent | ContainerEvent") -> None:
        self._events.append(event.as_dict())
        del self._events[:-100]

    def _publish(self, image: np.ndarray) -> None:
        self._frame = _encode(image)
        self._frame_ready.set()


def _encode(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return buf.tobytes() if ok else b""


class PreviewRegistry:
    """One session per camera, created on demand."""

    def __init__(self) -> None:
        self._sessions: dict[str, PreviewSession] = {}

    def get(self, camera_id: str, name: str, rtsp_url: str) -> PreviewSession:
        session = self._sessions.get(camera_id)
        if session is None or session.rtsp_url != rtsp_url:
            session = PreviewSession(camera_id, name, rtsp_url)
            self._sessions[camera_id] = session
        return session

    def existing(self, camera_id: str) -> PreviewSession | None:
        return self._sessions.get(camera_id)

    def all(self) -> list[PreviewSession]:
        return list(self._sessions.values())


registry = PreviewRegistry()
