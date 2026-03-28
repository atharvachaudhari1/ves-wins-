from __future__ import annotations

import base64
import io
import re
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def base64_to_numpy(b64_string: str) -> np.ndarray:
    """Convert base64 encoded image string to numpy BGR array."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    b64_string = re.sub(r"\s+", "", b64_string)
    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from base64")
    return img


def numpy_to_base64(img: np.ndarray) -> str:
    """Convert numpy BGR array to a data-URL base64 PNG string."""
    _, buffer = cv2.imencode(".png", img)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def validate_image(img: np.ndarray, max_mb: int = 10) -> bool:
    """Check image is within size limits (raw array size in memory)."""
    size_mb = img.nbytes / (1024 * 1024)
    return size_mb <= max_mb


def resize_if_needed(img: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """Resize image if either dimension exceeds max_dim."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def bytes_to_bgr(data: bytes) -> np.ndarray:
    """Decode raw image bytes (e.g. multipart body) into a BGR ``uint8`` array via ``cv2.imdecode``."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


def bgr_to_png_bytes(bgr: np.ndarray) -> bytes:
    """Encode a BGR ``uint8`` image to PNG bytes using OpenCV (no base64)."""
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("Could not encode image to PNG")
    return buf.tobytes()


def bgr_to_base64_png(bgr: np.ndarray) -> str:
    """Encode a BGR image to a PNG data URL (same as :func:`numpy_to_base64`)."""
    return numpy_to_base64(bgr)


def bgr_to_jpeg_bytes(bgr: np.ndarray, quality: int = 92) -> bytes:
    """Encode a BGR ``uint8`` image to JPEG bytes at the given quality."""
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ValueError("Could not encode image to JPEG")
    return buf.tobytes()


def resize_max_side(bgr: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Resize so the longest side is at most ``max_side``; return the image and linear scale factor."""
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr, 1.0
    scale = max_side / m
    nw, nh = int(w * scale), int(h * scale)
    out = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, scale


def pil_bytes_to_bgr(data: bytes) -> np.ndarray:
    """Decode image bytes with Pillow when format support is needed, then convert RGB→BGR with OpenCV."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
