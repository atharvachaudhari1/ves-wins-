from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from core.config import config, settings
from core.logger import get_logger
from services.deepface_probe import probe_after_each_layer, probe_all_layers
from services.noise_composer import ShieldParams, compose_shield, compose_with_stages
from utils.image_utils import (
    base64_to_numpy,
    bgr_to_base64_png,
    bytes_to_bgr,
    numpy_to_base64,
    resize_if_needed,
    validate_image,
)
from utils.metrics import psnr_db

router = APIRouter(prefix="/shield", tags=["shield"])
route_log = get_logger("faceshield.routes.shield")


class ShieldSettings(BaseModel):
    layer1_intensity: int = Field(default=20, ge=0, le=200)
    layer1_pattern: str = Field(default="gaussian")
    layer2_intensity: float = Field(default=0.08, ge=0.0, le=1.0)
    layer2_band: str = Field(default="mid")
    layer3_intensity: int = Field(default=25, ge=0, le=200)


class ShieldJsonRequest(BaseModel):
    image: str = Field(..., min_length=1)
    settings: ShieldSettings = Field(default_factory=ShieldSettings)


def _params_from_form(
    enable_l1: bool,
    enable_l2: bool,
    enable_l3: bool,
    l1_intensity: int,
    l1_pattern: str,
    l2_intensity: float,
    l2_band: str,
    l3_intensity: int,
) -> ShieldParams:
    return ShieldParams(
        enable_l1=enable_l1,
        enable_l2=enable_l2,
        enable_l3=enable_l3,
        l1_intensity=l1_intensity,
        l1_pattern=l1_pattern,
        l2_intensity=l2_intensity,
        l2_band=l2_band,
        l3_intensity=l3_intensity,
    )


def _shield_json_response(body: ShieldJsonRequest) -> Any:
    start = time.perf_counter()
    try:
        img = base64_to_numpy(body.image)
        img = resize_if_needed(img)

        if not validate_image(img, settings.max_image_size_mb):
            raise ValueError(
                f"Image too large for configured limit ({settings.max_image_size_mb} MB raw buffer)"
            )

        s = body.settings
        composed = compose_shield(
            img,
            layer1_intensity=s.layer1_intensity,
            layer1_pattern=s.layer1_pattern,
            layer2_intensity=s.layer2_intensity,
            layer2_band=s.layer2_band,
            layer3_intensity=s.layer3_intensity,
        )

        scores = probe_all_layers(
            img,
            composed["layer1_image"],
            composed["layer2_image"],
            composed["layer3_image"],
            model_name=config.DEEPFACE_MODEL,
        )

        elapsed = round(time.perf_counter() - start, 2)
        route_log.info("POST /api/shield completed in %.2fs", elapsed)

        return {
            "shielded_image": numpy_to_base64(composed["shielded_image"]),
            "layer_images": {
                "layer1": numpy_to_base64(composed["layer1_image"]),
                "layer2": numpy_to_base64(composed["layer2_image"]),
                "layer3": numpy_to_base64(composed["layer3_image"]),
            },
            "confidence_scores": scores,
            "diff_scores": {
                "l1": composed["diff_l1"],
                "l2": composed["diff_l2"],
                "l3": composed["diff_l3"],
            },
            "processing_time": elapsed,
        }
    except Exception as e:
        route_log.warning("shield JSON failed: %s", e)
        return JSONResponse(status_code=422, content={"error": str(e)})


@router.post("", response_model=None)
@router.post("/", response_model=None)
def shield_compose_json(body: ShieldJsonRequest) -> Any:
    """POST /api/shield and POST /api/shield/ — JSON base64 image + settings."""
    return _shield_json_response(body)


@router.post("/process", response_model=None)
async def process_shield(
    file: UploadFile = File(...),
    enable_l1: bool = Form(True),
    enable_l2: bool = Form(True),
    enable_l3: bool = Form(True),
    l1_intensity: int = Form(20),
    l1_pattern: str = Form("gaussian"),
    l2_intensity: float = Form(0.08),
    l2_band: str = Form("mid"),
    l3_intensity: int = Form(25),
) -> Any:
    """Multipart pipeline: returns PNG data URL, metrics, and wrapped confidence rows."""
    t0 = time.perf_counter()
    raw = await file.read()
    if not raw:
        return JSONResponse(status_code=422, content={"error": "Empty file"})

    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(raw) > max_bytes:
        return JSONResponse(
            status_code=422,
            content={"error": f"File exceeds maximum size of {settings.max_image_size_mb} MB"},
        )

    suffix = Path(file.filename or "").suffix.lower().lstrip(".")
    allowed = settings.parsed_allowed_extensions()
    if not suffix or suffix not in allowed:
        return JSONResponse(
            status_code=422,
            content={
                "error": (
                    f"Invalid or missing file extension; allowed: {', '.join(sorted(allowed))}"
                )
            },
        )
    try:
        bgr = bytes_to_bgr(raw)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})

    params = _params_from_form(
        enable_l1,
        enable_l2,
        enable_l3,
        l1_intensity,
        l1_pattern,
        l2_intensity,
        l2_band,
        l3_intensity,
    )
    stages, layer_metrics = compose_with_stages(bgr, params)
    final = stages.after_layer3
    try:
        b64 = bgr_to_base64_png(final)
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})

    confidence = probe_after_each_layer(
        stages.original,
        stages.after_layer1,
        stages.after_layer2,
        stages.after_layer3,
    )
    elapsed = time.perf_counter() - t0
    route_log.info("POST /api/shield/process completed in %.3fs", elapsed)
    return {
        "image_base64": b64,
        "mime": "image/png",
        "psnr_db": round(psnr_db(stages.original, final), 2),
        "layer_metrics": layer_metrics,
        "confidence": confidence,
        "params": params.__dict__,
    }
