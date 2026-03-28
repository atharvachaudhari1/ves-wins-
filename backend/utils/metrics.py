from __future__ import annotations

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two same-shaped ``uint8``/numeric images."""
    if a.shape != b.shape:
        raise ValueError("Shape mismatch")
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff))


def psnr_db(a: np.ndarray, b: np.ndarray, max_val: float = 255.0) -> float:
    """Peak signal-to-noise ratio in decibels from MSE (capped when MSE is near zero)."""
    m = mse(a, b)
    if m <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((max_val * max_val) / m))
