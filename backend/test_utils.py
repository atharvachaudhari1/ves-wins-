"""Quick check for :mod:`utils.image_utils`. Run from ``backend``: ``python test_utils.py``."""

from __future__ import annotations

import os

import cv2

from utils.image_utils import base64_to_numpy, numpy_to_base64

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
img1_path = os.path.join(_root, "test_face1.jpg")

if __name__ == "__main__":
    img = cv2.imread(img1_path)
    if img is None:
        raise SystemExit(f"Missing or unreadable {img1_path} (create it or use a real photo path).")
    b64 = numpy_to_base64(img)
    back = base64_to_numpy(b64)
    print("Utils working OK", back.shape)
