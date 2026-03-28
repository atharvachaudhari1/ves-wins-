"""DeepFace probe demo (CPU, slow on first run). Run from ``faceshield``: ``python test_probe.py``."""

import cv2

from backend.services.deepface_probe import probe_all_layers
from backend.services.noise_composer import compose_shield

img = cv2.imread("test_face1.jpg")
if img is None:
    raise SystemExit("Missing test_face1.jpg in the current directory.")

composed = compose_shield(img)
scores = probe_all_layers(
    img,
    composed["layer1_image"],
    composed["layer2_image"],
    composed["layer3_image"],
)

print("\n=== CONFIDENCE SCORES ===")
print(f"Original:       {scores['original']}%")
print(f"After Layer 1:  {scores['after_layer1']}%")
print(f"After Layer 2:  {scores['after_layer2']}%")
print(f"After Layer 3:  {scores['after_layer3']}%")
print("========================\n")
