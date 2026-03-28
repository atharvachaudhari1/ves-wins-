"""Landmark-targeted noise: legacy ``FaceMesh`` when available, else Tasks ``FaceLandmarker``."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any, List, Optional, Sequence

import cv2
import mediapipe as mp
import numpy as np

# Key landmark indices (Face Mesh topology; compatible with Tasks Face Landmarker)
EYE_LANDMARKS = [
    33,
    133,
    362,
    263,
    159,
    145,
    386,
    374,
    160,
    144,
    387,
    373,
    161,
    163,
    388,
    390,
]
NOSE_LANDMARKS = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
JAW_LANDMARKS = [
    0,
    17,
    18,
    200,
    199,
    175,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
]

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

_face_landmarker_tasks = None


def _model_path() -> Path:
    root = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "faceshield_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root / "face_landmarker.task"


def _ensure_task_model() -> Path:
    p = _model_path()
    if not p.is_file():
        urllib.request.urlretrieve(_MODEL_URL, p)  # noqa: S310
    return p


def _get_face_landmarker_tasks():
    global _face_landmarker_tasks
    if _face_landmarker_tasks is None:
        path = str(_ensure_task_model())
        base = mp.tasks.BaseOptions(model_asset_path=path)
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            output_face_blendshapes=False,
        )
        _face_landmarker_tasks = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    return _face_landmarker_tasks


class _NormLM:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


def _has_solutions_face_mesh() -> bool:
    return hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")


def _landmarks_via_solutions(rgb: np.ndarray) -> Optional[Sequence[Any]]:
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark


def _landmarks_via_tasks(rgb: np.ndarray) -> Optional[List[_NormLM]]:
    h, w = rgb.shape[:2]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    det = _get_face_landmarker_tasks()
    res = det.detect(mp_image)
    if not res.face_landmarks:
        return None
    out: List[_NormLM] = []
    for lm in res.face_landmarks[0]:
        out.append(_NormLM(lm.x, lm.y, getattr(lm, "z", 0.0)))
    return out


def _detect_landmarks(rgb: np.ndarray) -> Optional[Sequence[Any]]:
    if _has_solutions_face_mesh():
        return _landmarks_via_solutions(rgb)
    return _landmarks_via_tasks(rgb)


def get_landmark_mask(
    image: np.ndarray,
    landmarks: Sequence[Any],
    indices: list,
    radius: int = 20,
) -> np.ndarray:
    """Build a weighted mask around specific landmark points."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    for idx in indices:
        if idx >= len(landmarks):
            continue
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(mask, (x, y), radius, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask


def apply_semantic_noise(
    image: np.ndarray,
    intensity: int = 25,
) -> np.ndarray:
    """
    Apply landmark-targeted adversarial noise.
    Uses ``mediapipe.solutions.face_mesh`` when present; otherwise Tasks ``FaceLandmarker``.
    Falls back to full-image Gaussian noise if no face is detected.
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_img = image.astype(np.float32)

    try:
        landmarks = _detect_landmarks(img_rgb)
    except Exception:
        landmarks = None

    _need = max(EYE_LANDMARKS + NOSE_LANDMARKS + JAW_LANDMARKS) + 1
    if landmarks is None or len(landmarks) < _need:
        print("Warning: No face detected in Layer 3 — using full-image fallback")
        noise = np.random.normal(0, intensity, image.shape).astype(np.float32)
        return np.clip(result_img + noise, 0, 255).astype(np.uint8)

    eye_mask = get_landmark_mask(image, landmarks, EYE_LANDMARKS, radius=25)
    nose_mask = get_landmark_mask(image, landmarks, NOSE_LANDMARKS, radius=20)
    jaw_mask = get_landmark_mask(image, landmarks, JAW_LANDMARKS, radius=18)

    composite_mask = np.clip(
        eye_mask * 1.0 + nose_mask * 0.8 + jaw_mask * 0.6,
        0,
        1.0,
    )

    mask_3ch = np.stack([composite_mask] * 3, axis=-1)

    noise = np.random.normal(0, intensity, result_img.shape).astype(np.float32)
    targeted_noise = noise * mask_3ch

    result_img = np.clip(result_img + targeted_noise, 0, 255)
    return result_img.astype(np.uint8)


def get_layer3_diff(original: np.ndarray, shielded: np.ndarray) -> float:
    """Return mean absolute difference after layer 3."""
    return float(
        np.mean(np.abs(original.astype(np.float32) - shielded.astype(np.float32))),
    )


def apply_layer3(bgr: np.ndarray, intensity: int = 25) -> np.ndarray:
    """Composer hook; same as :func:`apply_semantic_noise`."""
    return apply_semantic_noise(bgr, intensity=intensity)
