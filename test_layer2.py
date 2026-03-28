"""Layer-2 FFT smoke + JPEG round-trip. Run from ``faceshield``: ``python test_layer2.py``."""

import cv2

from backend.services.layer2_frequency import apply_frequency_noise

img = cv2.imread("test_face1.jpg")
if img is None:
    raise SystemExit("Missing test_face1.jpg in the current directory.")

noisy = apply_frequency_noise(img, intensity=0.08, band="mid")
cv2.imwrite("test_layer2_output.jpg", noisy)

cv2.imwrite("test_layer2_compressed.jpg", noisy, [cv2.IMWRITE_JPEG_QUALITY, 85])
_ = cv2.imread("test_layer2_compressed.jpg")
print("Layer 2 working OK")
print("Check test_layer2_output.jpg and test_layer2_compressed.jpg visually")
