"""Isolate DeepFace on CPU. Run from faceshield: .\\venv\\Scripts\\python test_deepface.py"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from deepface import DeepFace
import cv2
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
img1_path = os.path.join(_SCRIPT_DIR, "test_face1.jpg")
img2_path = os.path.join(_SCRIPT_DIR, "test_face2.jpg")


def _write_dummy_face(path: str, seed: int) -> None:
    """Simple face-like pattern so verify runs without bundled real photos."""
    rng = np.random.default_rng(seed)
    img = np.full((224, 224, 3), 235, dtype=np.uint8)
    cv2.ellipse(img, (112, 118), (72, 88), 0, 0, 360, (200, 180, 160), -1)
    cv2.circle(img, (88, 102), 9, (45, 45, 45), -1)
    cv2.circle(img, (136, 102), 9, (45, 45, 45), -1)
    cv2.ellipse(img, (112, 158), (36, 18), 0, 0, 360, (100, 70, 90), -1)
    noise = rng.integers(-20, 21, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img)


def ensure_images() -> None:
    if os.path.isfile(img1_path) and os.path.isfile(img2_path):
        return
    print("Creating synthetic test_face1.jpg / test_face2.jpg (replace with real photos if you like).")
    _write_dummy_face(img1_path, 1)
    _write_dummy_face(img2_path, 2)


ensure_images()

try:
    result = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name="Facenet",
        enforce_detection=False,
    )
    print("DeepFace working ✓")
    print("Distance:", result["distance"])
    print("Verified:", result["verified"])
except Exception as e:
    print("DeepFace error:", e)
