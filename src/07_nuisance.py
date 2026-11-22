"""Step 07 - low-level nuisance regressors per TR (CPU). <work>/features/nuisance/<ep>.npz

All regressors at TR k use only the window [t_k - TR, t_k) (past only):
  cuts_count   number of shot cuts (ffmpeg scene score > threshold on the native video)
  lum          mean luminance of the 4 fps frames
  fdiff        mean |frame_f - frame_{f-1}| (grayscale, 4 fps)
  rms          audio RMS (16 kHz mono)
  n_words      words whose offset falls in the previous transcript chunk (row k-1)
  speech       n_words > 0
  cut_in_surp  1 if a cut falls inside the 4 s clip used for the surprise value at
               row k (clip of anchor row k-shift: frames [e-8, e+8)); 0 otherwise
`valid_nuis` is False for k = 0 (empty window).

    python src/07_nuisance.py --jobs 4
"""
import ast
import json
import math
import os
import re
import subprocess
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed

import av
import numpy as np
import pandas as pd

from common import (audio_path, config_arg, ep_season, frame_end, frames_path, load_config, load_index, log,
                    save_npz_atomic, surprise_shift)

NAMES = ["cuts_count", "lum", "fdiff", "rms", "n_words", "speech", "cut_in_surp"]


def scene_cuts(path, thr):
    p = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-i", path, "-map", "0:v:0", "-an", "-vf",
                        f"scale=160:-2,select='gt(scene,{thr})',showinfo", "-f", "null", "-"],
                       capture_output=True, text=True, check=True)
    return np.array(sorted(float(m) for m in re.findall(r"pts_time:([0-9.]+)", p.stderr)))


def gray_frames(path):
    with av.open(path) as c:
        g = [f.to_ndarray(format="rgb24")[::4, ::4].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
             for f in c.decode(video=0)]
    return np.stack(g)


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    return x, sr


def count_words(row):
    v = row.get("words_per_tr")
    if isinstance(v, str) and v.strip():
        try:
            lst = ast.literal_eval(v)
            if isinstance(lst, (list, tuple)):
                return len(lst)
        except (ValueError, SyntaxError):
            return len(v.split())
    t = row.get("text_per_tr")
    return len(t.split()) if isinstance(t, str) else 0


def one(cfg, ep, info):
    out_path = os.path.join(cfg["dirs"]["nuisance"], f"{ep}.npz")
    if os.path.exists(out_path):
        return ep, "skip"
    TR, fps = cfg["tr"], cfg["video"]["fps"]
    N = info["n_tr_max"]
    CTX, FUT = cfg["extract"]["ctx_frames"], cfg["extract"]["fut_frames"]
    shift = surprise_shift(TR, fps, FUT)
    cuts = scene_cuts(info["video"], cfg["nuisance"]["scene_threshold"])
    g = gray_frames(frames_path(cfg, ep))
    x, sr = read_wav(audio_path(cfg, ep))
    df = pd.read_csv(info["transcript"], sep="\t")
    words = np.array([count_words(r) for _, r in df.iterrows()], np.float32)
    res = {n: np.zeros(N, np.float32) for n in NAMES}
    for k in range(1, N):
        t0, t1 = (k - 1) * TR, k * TR
        res["cuts_count"][k] = np.sum((cuts >= t0) & (cuts < t1))
        f0, f1 = math.ceil(fps * t0), frame_end(k, TR, fps)          # frames with t0 <= f/fps < t1
        f1 = min(f1, len(g))
        if f1 > f0:
            res["lum"][k] = g[f0:f1].mean()
            a = max(f0, 1)
            if f1 > a:
                res["fdiff"][k] = np.abs(g[a:f1] - g[a - 1:f1 - 1]).mean()
        s0, s1 = int(round(t0 * sr)), min(int(round(t1 * sr)), len(x))
        if s1 > s0:
            res["rms"][k] = np.sqrt(np.mean(x[s0:s1] ** 2))
        if k - 1 < len(words):
            res["n_words"][k] = words[k - 1]
        if k >= shift:
            e = frame_end(k - shift, TR, fps)
            c0, c1 = (e - CTX) / fps, (e + FUT) / fps
            res["cut_in_surp"][k] = float(np.any((cuts >= c0) & (cuts < c1)))
    res["speech"] = (res["n_words"] > 0).astype(np.float32)
    valid = np.ones(N, bool)
    valid[0] = False
    save_npz_atomic(out_path, valid_nuis=valid, cut_times=cuts, **res)
    return ep, f"ok, {len(cuts)} cuts"


def main():
    p = config_arg()
    p.add_argument("--seasons", type=int, nargs="*", default=None)
    p.add_argument("--jobs", type=int, default=2)
    args = p.parse_args()
    cfg = load_config(args.config)
    index = load_index(cfg)
    eps = [ep for ep in sorted(index["episodes"]) if (args.seasons is None or ep_season(ep) in args.seasons)
           and os.path.exists(frames_path(cfg, ep).replace(".mp4", ".json"))]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(one, cfg, ep, index["episodes"][ep]) for ep in eps]
        for fu in as_completed(futs):
            ep, msg = fu.result()
            log(cfg, "nuisance", f"{ep}: {msg}")


if __name__ == "__main__":
    main()
