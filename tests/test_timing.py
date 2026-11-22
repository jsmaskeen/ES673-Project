"""Frame-timing test for step 02 (CPU, ~10 s):  python tests/test_timing.py
Builds a 29.97 fps clip whose luma encodes the native frame number, re-encodes it
with the exact filter string of 02_reencode.py, and asserts that output frame f
shows the latest native frame at or before f/4 s (never a later one)."""
import importlib
import os
import subprocess
import sys
import tempfile

import av
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
re_ = importlib.import_module("02_reencode")
T = tempfile.mkdtemp()
subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "lavfi", "-t", "30", "-i",
                "nullsrc=s=320x180:r=30000/1001,geq=lum='mod(N\\,200)+20':cb=128:cr=128",
                "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", f"{T}/clock.mkv"], check=True)
for mode in ["letterbox", "crop"]:
    cfg = {"video": {"fps": 4, "size": 32, "mode": mode}}
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", f"{T}/clock.mkv", "-vf", re_.vf_string(cfg),
                    "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv420p", f"{T}/out.mp4"], check=True)
    with av.open(f"{T}/out.mp4") as c:
        out = np.array([f.to_ndarray(format="yuv420p")[:f.height][12:20, 8:24].mean() for f in c.decode(video=0)])
    lag = (np.arange(len(out)) / 4 * 30000 / 1001 - (out - 20) + 100) % 200 - 100
    assert lag.min() >= -0.01 and lag.max() < 1.01, (mode, lag.min(), lag.max())
    print(f"{mode}: content age in native frames {lag.min():.2f}..{lag.max():.2f} -> OK")
print("TIMING TEST PASSED")
