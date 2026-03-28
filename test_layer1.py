"""Smoke test for layer-1 pixel noise. Run from ``faceshield``: ``python test_layer1.py``."""

import cv2

from backend.services.layer1_pixel import apply_pixel_noise, get_layer1_diff

img = cv2.imread("test_face1.jpg")
if img is None:
    raise SystemExit("Missing test_face1.jpg in the current directory.")

noisy = apply_pixel_noise(img, intensity=20, pattern="gaussian")
diff = get_layer1_diff(img, noisy)
cv2.imwrite("test_layer1_output.jpg", noisy)
print(f"Layer 1 working OK | Mean diff: {diff:.2f}")
