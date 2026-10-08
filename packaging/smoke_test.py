"""Smoke test for a built executable: stitch a synthetic dual-fisheye photo.

Usage: python smoke_test.py PATH_TO_EXECUTABLE
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

exe = sys.argv[1]
with tempfile.TemporaryDirectory() as tmp:
    src, dst = Path(tmp) / "in.jpg", Path(tmp) / "out.jpg"
    # Same side-by-side layout and lens size as a real SM-C200 photo.
    rng = np.random.default_rng(0)
    image = rng.integers(60, 200, (3872, 2 * 3872, 3), dtype=np.uint8)
    cv2.imwrite(str(src), image)
    subprocess.run([exe, str(src), "-o", str(dst), "--width", "1024"], check=True)
    out = cv2.imread(str(dst))
    assert out is not None and out.shape[1] == 1024, "no/odd output"
print("smoke test passed:", out.shape)
