import os
from pathlib import Path

from dotenv import load_dotenv

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_BACKEND_ROOT / ".env")

_cors_raw = os.getenv("CORS_ORIGINS", "http://localhost:5173")


class Config:
    MAX_IMAGE_SIZE_MB: int = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
    DEEPFACE_MODEL: str = os.getenv("DEEPFACE_MODEL", "Facenet")
    CORS_ORIGINS: list[str] = [p.strip() for p in _cors_raw.split(",") if p.strip()]


config = Config()


def _parse_bool(v: str | None) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "on"}


class _AppSettings:
    """Same field names as the former Pydantic settings (used by ``main`` and routes)."""

    def __init__(self) -> None:
        self.max_image_size_mb = config.MAX_IMAGE_SIZE_MB
        self.deepface_model = config.DEEPFACE_MODEL
        self.cors_origins = _cors_raw
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.use_deepface = _parse_bool(os.getenv("USE_DEEPFACE"))
        self.allowed_extensions = os.getenv("ALLOWED_EXTENSIONS", "jpg,jpeg,png,webp")

    def parsed_allowed_extensions(self) -> frozenset[str]:
        return frozenset(
            x.strip().lower().lstrip(".")
            for x in self.allowed_extensions.split(",")
            if x.strip()
        )


settings = _AppSettings()
