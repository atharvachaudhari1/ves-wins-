"""JPEG-resilience check for :mod:`services.layer2_frequency` (uses OpenCV encode/decode)."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from services.layer2_frequency import apply_frequency_noise


def _jpeg_cycle_bgr(bgr: np.ndarray, quality: int = 85) -> np.ndarray:
    """Encode BGR ``uint8`` to JPEG at ``quality`` and decode back to BGR."""
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    dec = cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert dec is not None
    return dec


class TestLayer2JpegPersistence(unittest.TestCase):
    """Mid-band FFT noise should retain measurable correlation after JPEG Q=85."""

    def test_perturbation_correlates_after_jpeg(self) -> None:
        rng = np.random.default_rng(42)
        img = (rng.random((160, 160, 3)) * 220 + 10).astype(np.uint8)
        # Intensity tuned for tutorial-style FFT scaling (complex noise × mean(|spectrum|)).
        noisy = apply_frequency_noise(img, intensity=0.45, band="mid")
        jpeg_noisy = _jpeg_cycle_bgr(noisy, quality=85)

        r_pre = (noisy.astype(np.float32) - img.astype(np.float32)).ravel()
        r_post = (jpeg_noisy.astype(np.float32) - img.astype(np.float32)).ravel()
        if np.std(r_pre) < 1e-6 or np.std(r_post) < 1e-6:
            self.fail("insufficient perturbation for correlation test")

        corr = float(np.corrcoef(r_pre, r_post)[0, 1])
        self.assertGreater(
            corr,
            0.12,
            "FFT perturbation should remain partially aligned with pre-JPEG residual",
        )

        energy_pre = float(np.mean(np.abs(r_pre)))
        energy_post = float(np.mean(np.abs(r_post)))
        self.assertGreater(
            energy_post,
            0.18 * energy_pre,
            "mean absolute deviation from original should not collapse after JPEG",
        )


if __name__ == "__main__":
    unittest.main()
