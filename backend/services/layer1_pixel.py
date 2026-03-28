import numpy as np


def _gaussian_random(shape: tuple, mean: float, std: float) -> np.ndarray:
    """Box-Muller gaussian noise."""
    u1 = np.random.uniform(0.001, 1, shape)
    u2 = np.random.uniform(0.001, 1, shape)
    z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return mean + std * z


def apply_pixel_noise(
    image: np.ndarray,
    intensity: int = 20,
    pattern: str = "gaussian",
) -> np.ndarray:
    """
    Apply pixel-space adversarial noise to image.
    Patterns: gaussian | structured | checkerboard
    """
    img = image.astype(np.float32)
    h, w = img.shape[:2]

    if pattern == "gaussian":
        noise_r = np.random.normal(0, intensity, (h, w))
        noise_g = np.random.normal(0, intensity * 0.95, (h, w))
        noise_b = np.random.normal(0, intensity * 1.05, (h, w))

    elif pattern == "structured":
        base = np.fromfunction(
            lambda y, x: np.where((x + y) % 2 == 0, 1, -1) * intensity * 0.7,
            (h, w),
        )
        noise_r = base + np.random.normal(0, intensity * 0.3, (h, w))
        noise_g = base + np.random.normal(0, intensity * 0.3, (h, w))
        noise_b = base + np.random.normal(0, intensity * 0.3, (h, w))

    elif pattern == "checkerboard":
        block = max(4, intensity // 4)
        base = np.fromfunction(
            lambda y, x: np.where(
                (np.floor(x / block) + np.floor(y / block)) % 2 == 0, 1, -1
            )
            * intensity
            * 0.8,
            (h, w),
        )
        noise_r = base + np.random.normal(0, intensity * 0.2, (h, w))
        noise_g = base + np.random.normal(0, intensity * 0.2, (h, w))
        noise_b = base + np.random.normal(0, intensity * 0.2, (h, w))

    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    img[:, :, 0] = np.clip(img[:, :, 0] + noise_b, 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] + noise_g, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] + noise_r, 0, 255)

    return img.astype(np.uint8)


def get_layer1_diff(original: np.ndarray, shielded: np.ndarray) -> float:
    """Return mean absolute pixel difference between original and shielded."""
    return float(
        np.mean(
            np.abs(original.astype(np.float32) - shielded.astype(np.float32)),
        ),
    )


def apply_layer1(bgr: np.ndarray, intensity: int = 20, pattern: str = "gaussian") -> np.ndarray:
    """Composer hook: same as :func:`apply_pixel_noise`."""
    return apply_pixel_noise(bgr, intensity=intensity, pattern=pattern)
