import numpy as np
import cv2


def get_frequency_band_mask(shape: tuple, band: str) -> np.ndarray:
    """
    Create a mask targeting specific frequency bands in FFT space.
    low=center, mid=ring, high=edges
    """
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = float(np.sqrt(cx**2 + cy**2))
    if max_dist < 1e-6:
        max_dist = 1.0

    if band == "low":
        mask[dist < max_dist * 0.1] = 1.0
    elif band == "mid":
        mask[(dist >= max_dist * 0.1) & (dist < max_dist * 0.5)] = 1.0
    elif band == "high":
        mask[dist >= max_dist * 0.5] = 1.0

    return mask


def apply_frequency_noise(
    image: np.ndarray,
    intensity: float = 0.08,
    band: str = "mid",
) -> np.ndarray:
    """
    Apply frequency-domain adversarial noise via FFT.
    JPEG-robust — perturbations survive compression.
    """
    img = image.astype(np.float32)
    result = img.copy()
    h, w = img.shape[:2]
    mask = get_frequency_band_mask((h, w), band)

    for ch in range(3):
        channel = img[:, :, ch]
        f = np.fft.fft2(channel)
        f_shifted = np.fft.fftshift(f)

        noise_real = np.random.normal(0, 1, (h, w)) * mask
        noise_imag = np.random.normal(0, 1, (h, w)) * mask
        mag_mean = float(np.mean(np.abs(f_shifted)) + 1e-6)
        perturbation = (noise_real + 1j * noise_imag) * intensity * mag_mean

        f_shifted_perturbed = f_shifted + perturbation
        f_ishift = np.fft.ifftshift(f_shifted_perturbed)
        channel_back = np.fft.ifft2(f_ishift).real

        result[:, :, ch] = np.clip(channel_back, 0, 255)

    return result.astype(np.uint8)


def get_layer2_diff(original: np.ndarray, shielded: np.ndarray) -> float:
    """Return mean absolute difference after layer 2."""
    return float(
        np.mean(np.abs(original.astype(np.float32) - shielded.astype(np.float32))),
    )


def apply_layer2(
    bgr: np.ndarray,
    intensity: float = 0.08,
    band: str = "mid",
) -> np.ndarray:
    """Composer entry point; same as :func:`apply_frequency_noise`."""
    return apply_frequency_noise(bgr, intensity=intensity, band=band)
