from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from services.layer1_pixel import apply_pixel_noise, get_layer1_diff as _get_layer1_diff
from services.layer2_frequency import apply_frequency_noise, get_layer2_diff
from services.layer3_semantic import apply_semantic_noise, get_layer3_diff
from utils.metrics import psnr_db


def _mean_abs_diff_vs_original(original: np.ndarray, current: np.ndarray) -> float:
    """Mean absolute per-channel difference between ``current`` and ``original`` (same shape)."""
    if original.shape != current.shape:
        raise ValueError("Shape mismatch")
    a = original.astype(np.float32)
    b = current.astype(np.float32)
    return float(np.mean(np.abs(b - a)))


@dataclass
class ShieldParams:
    """User-tunable parameters for each noise layer (all layers remain independent)."""

    enable_l1: bool = True
    enable_l2: bool = True
    enable_l3: bool = True
    l1_intensity: int = 20
    l1_pattern: str = "gaussian"
    l2_intensity: float = 0.08
    l2_band: str = "mid"
    l3_intensity: int = 25


@dataclass
class StageImages:
    """BGR ``uint8`` arrays after each pipeline stage (including the untouched original)."""

    original: np.ndarray
    after_layer1: np.ndarray
    after_layer2: np.ndarray
    after_layer3: np.ndarray


def compose_shield(
    image: np.ndarray,
    layer1_intensity: int = 20,
    layer1_pattern: str = "gaussian",
    layer2_intensity: float = 0.08,
    layer2_band: str = "mid",
    layer3_intensity: int = 25,
) -> dict:
    """
    Orchestrate all 3 noise layers sequentially.
    Each layer receives the output of the previous.
    Returns intermediate images and diff scores per layer (L1 vs input, L2 vs L1, L3 vs L2).
    """
    l1_image = apply_pixel_noise(image, layer1_intensity, layer1_pattern)
    diff_l1 = _get_layer1_diff(image, l1_image)

    l2_image = apply_frequency_noise(l1_image, layer2_intensity, layer2_band)
    diff_l2 = get_layer2_diff(l1_image, l2_image)

    l3_image = apply_semantic_noise(l2_image, layer3_intensity)
    diff_l3 = get_layer3_diff(l2_image, l3_image)

    return {
        "shielded_image": l3_image,
        "layer1_image": l1_image,
        "layer2_image": l2_image,
        "layer3_image": l3_image,
        "diff_l1": round(diff_l1, 4),
        "diff_l2": round(diff_l2, 4),
        "diff_l3": round(diff_l3, 4),
    }


def compose_with_stages(bgr: np.ndarray, p: ShieldParams) -> Tuple[StageImages, Dict[str, Any]]:
    """Run the pipeline with optional layer skips; returns stages and PSNR / diff metrics."""
    original = bgr.copy()

    if p.enable_l1 and p.enable_l2 and p.enable_l3:
        r = compose_shield(
            original,
            layer1_intensity=p.l1_intensity,
            layer1_pattern=p.l1_pattern,
            layer2_intensity=p.l2_intensity,
            layer2_band=p.l2_band,
            layer3_intensity=p.l3_intensity,
        )
        x1 = r["layer1_image"]
        x2 = r["layer2_image"]
        x3 = r["layer3_image"]
        diff_l1 = float(r["diff_l1"])
        diff_l2 = float(r["diff_l2"])
        diff_l3 = float(r["diff_l3"])
    else:
        x1 = (
            apply_pixel_noise(original, p.l1_intensity, p.l1_pattern)
            if p.enable_l1
            else original.copy()
        )
        x2 = (
            apply_frequency_noise(x1, p.l2_intensity, p.l2_band)
            if p.enable_l2
            else x1.copy()
        )
        x3 = apply_semantic_noise(x2, p.l3_intensity) if p.enable_l3 else x2.copy()
        diff_l1 = _mean_abs_diff_vs_original(original, x1)
        diff_l2 = _mean_abs_diff_vs_original(original, x2)
        diff_l3 = _mean_abs_diff_vs_original(original, x3)

    stages = StageImages(original=original, after_layer1=x1, after_layer2=x2, after_layer3=x3)
    metrics: Dict[str, Any] = {
        "layer1": {
            "applied": p.enable_l1,
            "psnr_vs_original_db": round(psnr_db(original, x1), 3),
            "mean_abs_diff": round(_get_layer1_diff(original, x1), 4),
            "diff_from_original": round(diff_l1, 4),
        },
        "layer2": {
            "applied": p.enable_l2,
            "psnr_vs_original_db": round(psnr_db(original, x2), 3),
            "diff_from_original": round(diff_l2, 4),
        },
        "layer3": {
            "applied": p.enable_l3,
            "psnr_vs_original_db": round(psnr_db(original, x3), 3),
            "diff_from_original": round(diff_l3, 4),
        },
        "diff_l1": round(diff_l1, 4),
        "diff_l2": round(diff_l2, 4),
        "diff_l3": round(diff_l3, 4),
    }
    return stages, metrics


def compose(bgr: np.ndarray, p: ShieldParams) -> np.ndarray:
    """Return only the final BGR image after :func:`compose_with_stages` (convenience wrapper)."""
    stages, _ = compose_with_stages(bgr, p)
    return stages.after_layer3
