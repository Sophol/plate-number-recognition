import asyncio

import structlog
from prometheus_client import Counter
from sqlalchemy.ext.asyncio import AsyncSession

from apps.camera_worker.sampler import BoundedFrameQueue, Frame
from apps.inference_worker.detector import ContourPlateDetector
from apps.inference_worker.ocr import TemplateOCR
from apps.inference_worker.pipeline import CommittedRead, InferencePipeline
from apps.inference_worker.province import EnglishZoneProvinceClassifier
from config import get_settings
from db.models import ContainerRead, Outbox, PlateRead
from db.session import SessionLocal

log = structlog.get_logger()

reads_committed = Counter("anpr_reads_committed_total", "Plate reads committed after voting")
frames_processed = Counter("anpr_frames_processed_total", "Frames run through inference")
containers_committed = Counter("anpr_containers_committed_total", "Container numbers committed after voting")


def build_ocr(backend: str = "template", use_gpu: bool = False):
    """Pick an OCR backend by name.

    "template" is the classical-CV fallback that needs no extra dependencies;
    "paddle" reads real plate typefaces but requires paddlepaddle + paddleocr.
    An unavailable paddle install falls back rather than taking the worker down,
    because a degraded read is better than no pipeline at all.
    """
    if backend == "paddle":
        from apps.inference_worker.paddle_ocr import PaddlePlateOCR

        try:
            return PaddlePlateOCR(use_gpu=use_gpu)
        except RuntimeError as exc:
            log.warning("paddle_unavailable_using_template", error=str(exc))
            return TemplateOCR()
    return TemplateOCR()


def build_detector(backend: str = "contour", model_path: str | None = None):
    """Pick a plate locator by name.

    "onnx" is the trained YOLO detector and the only one meant for production;
    "contour" needs no extra dependencies but finds bright rectangles rather
    than plates; "paddle" runs PaddleOCR's text detector and keeps only
    plate-shaped results. Falls back rather than taking the worker down.

    The onnx fallback is the one to watch in production: a missing model file
    silently downgrades gate accuracy to the classical-CV locator. The warning
    is the only signal, so alert on it.
    """
    if backend == "onnx":
        from apps.inference_worker.onnx_detector import OnnxPlateDetector

        try:
            settings = get_settings()
            return OnnxPlateDetector(
                model_path or settings.detector_model_path,
                conf=settings.detector_min_confidence,
                threads=settings.ort_intra_op_threads,
            )
        except RuntimeError as exc:
            log.warning("onnx_detector_unavailable_using_contour", error=str(exc))
            return ContourPlateDetector()
    if backend == "paddle":
        from apps.inference_worker.paddle_detector import PaddlePlateDetector

        try:
            return PaddlePlateDetector()
        except RuntimeError as exc:
            log.warning("paddle_detector_unavailable_using_contour", error=str(exc))
            return ContourPlateDetector()
    return ContourPlateDetector()


def build_vehicle_detector(settings=None):
    """The COCO vehicle detector, or None when disabled or its model is absent."""
    settings = settings or get_settings()
    if settings.vehicle_backend != "onnx":
        return None
    from apps.inference_worker.vehicle import OnnxColourClassifier, OnnxVehicleDetector

    # The learned colour classifier is optional: enable it with
    # vehicle_colour_backend="onnx", but if its model is missing, log and carry
    # on with the heuristic rather than dropping the whole vehicle detector.
    colour_classifier = None
    if settings.vehicle_colour_backend == "onnx":
        try:
            colour_classifier = OnnxColourClassifier(
                settings.vehicle_colour_model_path, threads=settings.ort_intra_op_threads)
            log.info("vehicle_colour_classifier_loaded", model=settings.vehicle_colour_model_path)
        except RuntimeError as exc:
            log.warning("vehicle_colour_classifier_unavailable_using_heuristic", error=str(exc))

    try:
        return OnnxVehicleDetector(settings.vehicle_model_path, threads=settings.ort_intra_op_threads,
                                   colour_classifier=colour_classifier)
    except RuntimeError as exc:
        log.warning("vehicle_detector_unavailable", error=str(exc))
        return None


def build_container_capture(settings=None):
    """Crop + frame evidence for each committed container read, or None when unset."""
    settings = settings or get_settings()
    if not settings.container_capture_dir:
        return None
    from apps.inference_worker.capture import ContainerCapture
    return ContainerCapture(settings.container_capture_dir)


def build_container_reader(settings=None):
    """Locator + CRNN + PAS lookup, or None when disabled or the model is absent.

    Like the detector fallback, a missing model degrades silently to "no
    container reads" -- container_reader_unavailable is the line to alert on.
    """
    settings = settings or get_settings()
    if settings.container_backend != "onnx":
        return None
    from apps.inference_worker.container import load_known
    from apps.inference_worker.container_ocr import OnnxContainerOCR
    from apps.inference_worker.container_pipeline import ContainerReader
    from apps.inference_worker.pas import PasClient

    try:
        known = load_known(settings.container_known_list_path)
        ocr = OnnxContainerOCR(settings.container_ocr_model_path, known=known, threads=1)
    except RuntimeError as exc:
        log.warning("container_reader_unavailable", error=str(exc))
        return None
    pas = PasClient(settings.pas_sql_url, fallback=known) if settings.pas_sql_url else None
    return ContainerReader(ocr, pas=pas, min_confidence=settings.container_min_confidence)


def build_pipeline(
    model_version: str,
    votes_required: int,
    ocr_backend: str = "template",
    ocr_use_gpu: bool = False,
    detector_backend: str = "contour",
) -> InferencePipeline:
    """Wire the inference backends. Swap these for ONNX adapters once trained."""
    ocr = build_ocr(ocr_backend, ocr_use_gpu)
    return InferencePipeline(
        detector=build_detector(detector_backend),
        ocr=ocr,
        # Province falls back to reading the Latin bottom zone, which the
        # template matcher handles; keep it on that backend regardless.
        province_classifier=EnglishZoneProvinceClassifier(TemplateOCR()),
        model_version=model_version,
        votes_required=votes_required,
        vehicle_detector=build_vehicle_detector(),
        container_reader=build_container_reader(),
        container_every_n_frames=get_settings().container_every_n_frames,
        container_capture=build_container_capture(),
        container_unknown_min_confidence=get_settings().container_unknown_min_confidence,
    )


async def persist(session: AsyncSession, read: CommittedRead) -> None:
    """Write the read and its outbox event in one transaction.

    The outbox row is what makes publishing reliable: if the process dies after
    commit, the relay still delivers it; if the commit fails, neither exists.
    """
    session.add(
        PlateRead(
            id=read.id,
            camera_id=read.camera_id,
            track_id=read.track_id,
            plate_text=read.plate_text,
            province_code=read.province_code,
            plate_type=read.plate_type,
            confidence=read.confidence,
            detector_confidence=read.detector_confidence,
            ocr_confidence=read.ocr_confidence,
            province_confidence=read.province_confidence,
            frame_ts=read.frame_ts,
            model_version=read.model_version,
            is_valid=read.is_valid,
        )
    )
    session.add(
        Outbox(
            aggregate="plate_read",
            payload={
                "plate_read_id": str(read.id),
                "camera_id": str(read.camera_id),
                "plate_text": read.plate_text,
                "province_code": read.province_code,
                "plate_type": read.plate_type,
                "confidence": read.confidence,
                "is_valid": read.is_valid,
                "frame_ts": read.frame_ts.isoformat(),
            },
        )
    )


async def consume(queue: BoundedFrameQueue, pipeline: InferencePipeline) -> None:
    while True:
        frame: Frame = await queue.get()
        frames_processed.inc()

        try:
            # pipeline.process is synchronous and CPU-bound -- a 640px ONNX
            # detection costs ~100ms on a CPU box. Running it inline would block
            # the event loop for that long, which in the combined runner means
            # RTSP sockets stop being drained and frames pile up in the kernel
            # buffer. onnxruntime and OpenCV release the GIL, so a worker thread
            # genuinely overlaps inference with capture.
            committed = await asyncio.to_thread(pipeline.process, frame)
        except Exception:
            log.exception("inference_failed", camera_id=frame.camera_id)
            continue

        try:
            containers = await asyncio.to_thread(pipeline.process_containers, frame)
        except Exception:
            log.exception("container_inference_failed", camera_id=frame.camera_id)
            containers = []
        if containers:
            async with SessionLocal() as session:
                for c in containers:
                    session.add(ContainerRead(
                        id=c.id, camera_id=c.camera_id, container_number=c.container_number,
                        owner_code=c.owner_code, checksum_ok=c.checksum_ok, is_known=c.is_known,
                        ocr_text=c.ocr_text, was_snapped=c.was_snapped, confidence=c.confidence,
                        frame_ts=c.frame_ts, model_version=c.model_version,
                        crop_path=c.crop_path, frame_path=c.frame_path,
                    ))
                    log.info("container_committed", camera_id=c.camera_id,
                             container_number=c.container_number, is_known=c.is_known,
                             confidence=round(c.confidence, 3), crop=c.crop_path)
                await session.commit()
            containers_committed.inc(len(containers))

        if not committed:
            continue

        async with SessionLocal() as session:
            for read in committed:
                await persist(session, read)
                log.info(
                    "plate_committed",
                    camera_id=read.camera_id,
                    plate_text=read.plate_text,
                    province_code=read.province_code,
                    is_valid=read.is_valid,
                )
            await session.commit()
        reads_committed.inc(len(committed))


async def main() -> None:
    settings = get_settings()
    queue = BoundedFrameQueue(settings.frame_queue_maxsize)
    pipeline = build_pipeline(
        settings.model_version,
        settings.votes_required,
        ocr_backend=settings.ocr_backend,
        ocr_use_gpu=settings.ocr_use_gpu,
        detector_backend=settings.detector_backend,
    )

    log.info("inference_worker_started", model_version=settings.model_version)
    await consume(queue, pipeline)


if __name__ == "__main__":
    asyncio.run(main())
