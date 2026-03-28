from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import cv2
import numpy as np

from core.config import settings


def _img_to_temp_file(img: np.ndarray) -> str:
    """Save numpy image to temp file for DeepFace."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, img)
    return tmp.name


def _get_confidence(img1: np.ndarray, img2: np.ndarray, model_name: str) -> float:
    """
    Compare two images with DeepFace.
    Returns 0-100 confidence score (100 = same face, 0 = different).
    """
    from deepface import DeepFace  # type: ignore

    p1 = p2 = None
    try:
        p1 = _img_to_temp_file(img1)
        p2 = _img_to_temp_file(img2)
        result = DeepFace.verify(
            img1_path=p1,
            img2_path=p2,
            model_name=model_name,
            enforce_detection=False,
            silent=True,
        )
        distance = float(result.get("distance", 0.0))
        th = result.get("threshold")
        if th is None:
            th = result.get("max_threshold_to_verify")
        threshold = float(th if th is not None else 0.4)
        if threshold < 1e-9:
            threshold = 0.4
        confidence = max(0.0, min(100.0, 100.0 - (distance / threshold * 100.0)))
        return round(confidence, 1)
    except Exception as e:
        print(f"DeepFace probe error: {e}")
        return 0.0
    finally:
        for p in (p1, p2):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass


def probe_all_layers(
    original: np.ndarray,
    layer1_img: np.ndarray,
    layer2_img: np.ndarray,
    layer3_img: np.ndarray,
    model_name: str = "Facenet",
) -> dict:
    """
    Run DeepFace confidence check after each layer.
    Returns confidence scores at each stage.
    """
    return {
        "original": 100.0,
        "after_layer1": _get_confidence(original, layer1_img, model_name),
        "after_layer2": _get_confidence(original, layer2_img, model_name),
        "after_layer3": _get_confidence(original, layer3_img, model_name),
    }


def _pixel_similarity_confidence_0_100(
    img_ref: np.ndarray,
    img_cmp: np.ndarray,
    dampen: float,
) -> float:
    """Deterministic 0–100 stand-in from mean absolute RGB difference (demo when DeepFace is off)."""
    a = cv2.resize(img_ref, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float64)
    b = cv2.resize(img_cmp, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float64)
    diff = float(np.mean(np.abs(a - b)))
    base = max(8.0, min(98.0, 100.0 * (1.0 - diff / 95.0)))
    return round(max(0.0, min(100.0, base - dampen)), 1)


def _demo_probe_scores(
    original: np.ndarray,
    after_layer1: np.ndarray,
    after_layer2: np.ndarray,
    after_layer3: np.ndarray,
) -> Dict[str, float]:
    """Heuristic 0–100 scores for UI when DeepFace is disabled."""
    return {
        "original": 100.0,
        "after_layer1": _pixel_similarity_confidence_0_100(original, after_layer1, 8.0),
        "after_layer2": _pixel_similarity_confidence_0_100(original, after_layer2, 16.0),
        "after_layer3": _pixel_similarity_confidence_0_100(original, after_layer3, 24.0),
    }


def _wrap_scores_for_api(scores: Dict[str, float], mode: str) -> Dict[str, Any]:
    """Convert flat 0–100 scores to the structure expected by multipart clients."""

    def row(v: float) -> Dict[str, Any]:
        x = float(v)
        return {"confidence_pct": x, "raw": x / 100.0}

    return {
        "mode": mode,
        "original": row(scores["original"]),
        "after_layer1": row(scores["after_layer1"]),
        "after_layer2": row(scores["after_layer2"]),
        "after_layer3": row(scores["after_layer3"]),
    }


def probe_after_each_layer(
    original: np.ndarray,
    after_layer1: np.ndarray,
    after_layer2: np.ndarray,
    after_layer3: np.ndarray,
) -> Dict[str, Any]:
    """API-facing probe: uses :func:`probe_all_layers` when DeepFace is enabled, else demo scores."""
    use_df = settings.use_deepface or os.environ.get("USE_DEEPFACE", "0") == "1"
    model = settings.deepface_model

    if use_df:
        try:
            scores = probe_all_layers(
                original,
                after_layer1,
                after_layer2,
                after_layer3,
                model_name=model,
            )
            return _wrap_scores_for_api(scores, "deepface")
        except Exception:
            pass

    demo = _demo_probe_scores(original, after_layer1, after_layer2, after_layer3)
    return _wrap_scores_for_api(demo, "demo")
