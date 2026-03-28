"""End-to-end compose smoke test. Run from ``faceshield``: ``python test_composer.py``."""

import cv2

from backend.services.noise_composer import compose_shield

img = cv2.imread("test_face1.jpg")
if img is None:
    raise SystemExit("Missing test_face1.jpg in the current directory.")

result = compose_shield(img)

cv2.imwrite("output_layer1.jpg", result["layer1_image"])
cv2.imwrite("output_layer2.jpg", result["layer2_image"])
cv2.imwrite("output_layer3.jpg", result["layer3_image"])

print("Composer working OK")
print(f"Diff L1: {result['diff_l1']}")
print(f"Diff L2: {result['diff_l2']}")
print(f"Diff L3: {result['diff_l3']}")
