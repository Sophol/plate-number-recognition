from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://anpr:anpr@localhost:5432/anpr"
    database_url_sync: str = "postgresql+psycopg2://anpr:anpr@localhost:5432/anpr"

    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    redis_url: str = "redis://localhost:6379/0"

    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "anpr-crops"

    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # Sampled rate per camera; cameras usually deliver 25-30 FPS natively.
    target_fps: float = 8.0
    # Fraction of ROI pixels that must change before a frame is considered moving.
    motion_min_changed_fraction: float = 0.002

    # Frame voting requires this many agreeing reads within one track.
    votes_required: int = 3
    # Bounded queue depth between camera and inference workers; oldest frame
    # is dropped when full so ingest never blocks on GPU backpressure.
    frame_queue_maxsize: int = 32

    # "template" is the classical-CV fallback that ships by default; "paddle"
    # reads real plate typefaces but needs paddlepaddle + paddleocr installed.
    # "onnx" is the trained YOLO detector and the production choice; "contour"
    # is the classical-CV locator that ships by default; "paddle"
    # uses PaddleOCR's text detector filtered by plate format, which is what
    # actually finds plates in real photographs.
    detector_backend: str = "contour"
    ocr_backend: str = "template"
    ocr_use_gpu: bool = False

    detector_model_path: str = "models/plate_detector.onnx"
    # Minimum YOLO confidence for a box to count as a plate. Site-dependent:
    # a cluttered barrier scene wants this higher than a clean driveway.
    detector_min_confidence: float = 0.5
    province_model_path: str = "models/province_classifier.onnx"
    model_version: str = "v0-dev"

    # PAS (port system) SQL gateway. Read-only source of every container number
    # that has passed through the terminal -- a known-container lookup that is a
    # far stronger check than the ISO 6346 checksum alone, and real owner-code
    # distributions to ground the synthetic OCR data in.
    pas_sql_url: str = ""

    # Container-number recognition. "off" skips it entirely; "onnx" runs the
    # trained CRNN. The known list is the fetched PAS export; without it reads
    # are still checksum-validated but cannot be snapped or marked known.
    container_backend: str = "off"
    container_ocr_model_path: str = "models/container_ocr.onnx"
    container_known_list_path: str = "dataset/pas/containers.json"
    container_min_confidence: float = 0.5

    # Vehicle type (car/truck/bus/motorcycle) and colour from a COCO-pretrained
    # YOLO via ONNX. "off" skips it. Colour is a heuristic and unreliable at a
    # steep overhead angle; type is dependable.
    vehicle_backend: str = "off"
    vehicle_model_path: str = "models/vehicle_detector.onnx"


@lru_cache
def get_settings() -> Settings:
    return Settings()
