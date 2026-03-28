"""Layer-3 semantic noise smoke test. Run from ``faceshield``: ``python test_layer3.py``."""

import cv2

from backend.services.layer3_semantic import apply_semantic_noise

img = cv2.imread("test_face1.jpg")
if img is None:
    raise SystemExit("Missing test_face1.jpg in the current directory.")

noisy = apply_semantic_noise(img, intensity=25)
cv2.imwrite("test_layer3_output.jpg", noisy)
print("Layer 3 working OK — check test_layer3_output.jpg")
