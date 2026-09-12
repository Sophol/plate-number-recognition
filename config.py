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


@lru_cache
def get_settings() -> Settings:
    return Settings()
