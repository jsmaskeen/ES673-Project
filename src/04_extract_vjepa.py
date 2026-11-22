"""Step 04 - V-JEPA 2 features per TR (GPU). One <work>/features/vjepa/<ep>.npz per episode.

For TR k (t_k = k*TR, e_k = floor(fps*t_k)), frame windows (frame f = time f/fps):
  past  = frames [e_k-8,  e_k)      2 s before t_k          (causal)
  long  = frames [e_k-24, e_k)      6 s before t_k          (causal)
  full  = frames [e_k-8,  e_k+8)    past + the next 2 s     (oracle only)

Saved arrays ([N] rows, N = max fMRI length over subjects; NaN where invalid):
  c_mid    mean-pooled encoder(past) after block `layer_mid`      causal
  c_last   mean-pooled encoder(past) final layer                   causal
  clong    mean-pooled encoder(long) final layer                   causal
  hhat     mean-pooled predictor forecast of next 2 s from past    causal
  hhat_rand  same with randomly re-initialised predictor           causal
  oracle   mean-pooled encoder(full) future-half tokens            uses frames up to t_k+2 s (control)
  fore_l1/fore_cos   forecast vs layer_norm(target), token-wise    anchored at k (window [t_k, t_k+2))
  rand_l1/rand_cos   random predictor vs target
  copy_l1/copy_cos   copy-forward (last past slot repeated) vs target
  surp_l1, surp_cos (=1-cos), surp_rand_l1, surp_rand_cos, chg_l1, chg_cos:
      the anchored errors moved to row k+shift (shift=2 for TR 1.49 s, 4 fps,
      8 future frames), so row k only uses frames before t_k.
  valid, surp_valid (bool), shift (int), frame_end (e_k)

    python src/04_extract_vjepa.py --seasons 6            # test season first
    python src/04_extract_vjepa.py --seasons 4 5 --max-minutes 110
"""
import json
import os
import time

import av
import numpy as np
import torch

from common import (config_arg, ep_season, frame_end, frames_path, load_config, load_index, log,
                    save_npz_atomic, surprise_shift)
from vjepa import Normalizer, batch_features, load_models, torch_dtype

POOLED = ["c_mid", "c_last", "clong", "hhat", "hhat_rand", "oracle"]
ANCHORED = ["fore_l1", "fore_cos", "rand_l1", "rand_cos", "copy_l1", "copy_cos"]


def load_frames(cfg, ep):
    path = frames_path(cfg, ep)
    side = json.load(open(path.replace(".mp4", ".json")))
    with av.open(path) as c:
        fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    x = np.stack(fr)
    S = cfg["video"]["size"]
    assert x.shape[1:] == (S, S, 3), x.shape
    assert len(x) == side["n_frames"], f"{path}: decoded {len(x)} frames, sidecar says {side['n_frames']}"
    return x


def run_episode(cfg, ep, n_tr, model, rand_pred, tps, norm):
    ex = cfg["extract"]
    fps, TR = cfg["video"]["fps"], cfg["tr"]
    FUT, LONG, BS = ex["fut_frames"], ex["long_frames"], ex["batch_trs"]
    Fr = load_frames(cfg, ep)
    nF = len(Fr)
    D = model.config.hidden_size
    shift = surprise_shift(TR, fps, FUT)
    e_all = np.array([frame_end(k, TR, fps) for k in range(n_tr)])
    assert all(e_all[k] + FUT <= e_all[k + shift] for k in range(n_tr - shift)), "shift would leak"
    valid = (e_all - LONG >= 0) & (e_all + FUT <= nF)
    out = {n: np.full((n_tr, D), np.nan, np.float32) for n in POOLED}
    out.update({n: np.full(n_tr, np.nan, np.float32) for n in ANCHORED})
    ks = np.where(valid)[0]
    for i in range(0, len(ks), BS):
        kb = ks[i:i + BS]
        r = batch_features(cfg, model, rand_pred, tps, norm, Fr, list(e_all[kb]))
        for n in POOLED + ANCHORED:
            out[n][kb] = r[n]
    for n, v in out.items():
        if not np.isfinite(v[valid]).all():
            raise FloatingPointError(f"{ep}: non-finite values in {n}; set model.dtype: bfloat16 and rerun")

    def shifted(a):
        b = np.full_like(a, np.nan)
        b[shift:] = a[:-shift]
        return b
    surp_valid = np.zeros(n_tr, bool)
    surp_valid[shift:] = valid[:-shift]
    res = {n: out[n].astype(np.float16) for n in POOLED}
    res.update({n: out[n] for n in ANCHORED})
    res.update({
        "surp_l1": shifted(out["fore_l1"]), "surp_cos": shifted(1 - out["fore_cos"]),
        "surp_rand_l1": shifted(out["rand_l1"]), "surp_rand_cos": shifted(1 - out["rand_cos"]),
        "chg_l1": shifted(out["copy_l1"]), "chg_cos": shifted(1 - out["copy_cos"]),
        "valid": valid, "surp_valid": surp_valid, "shift": np.int64(shift), "frame_end": e_all.astype(np.int64),
    })
    return res, int(valid.sum())


def main():
    p = config_arg()
    p.add_argument("--seasons", type=int, nargs="*", default=None)
    p.add_argument("--episodes", nargs="*", default=None)
    p.add_argument("--max-minutes", type=float, default=None, help="stop starting new episodes after this long")
    args = p.parse_args()
    cfg = load_config(args.config)
    index = load_index(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not cfg["model"].get("tiny_random"):
        log(cfg, "extract", "WARNING: no CUDA device; this will be extremely slow")
    model, rand_pred, tps = load_models(cfg, device)
    norm = Normalizer(device, torch_dtype(cfg["model"]["dtype"]))
    seasons = args.seasons or [cfg["test_season"]] + cfg["train_seasons"]
    eps = args.episodes or [ep for s in seasons for ep in sorted(index["episodes"]) if ep_season(ep) == s]
    t_start = time.time()
    for ep in eps:
        out_path = os.path.join(cfg["dirs"]["vjepa"], f"{ep}.npz")
        if os.path.exists(out_path):
            continue
        if not os.path.exists(frames_path(cfg, ep).replace(".mp4", ".json")):
            log(cfg, "extract", f"{ep}: frames not re-encoded yet, skipping")
            continue
        if args.max_minutes is not None and (time.time() - t_start) / 60 > args.max_minutes:
            log(cfg, "extract", "time budget reached; stopping (rerun to resume)")
            break
        t0 = time.time()
        res, nv = run_episode(cfg, ep, index["episodes"][ep]["n_tr_max"], model, rand_pred, tps, norm)
        save_npz_atomic(out_path, **res)
        dt = time.time() - t0
        log(cfg, "extract", f"{ep}: {nv} valid TRs in {dt:.0f}s ({nv / max(dt, 1e-9):.1f} TR/s)")


if __name__ == "__main__":
    main()
