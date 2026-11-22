"""Step 04b - extra V-JEPA 2 features for the horizon and spatial experiments (GPU).
One <work>/features/extra/<ep>.npz per episode; resumable; same frames and model as step 04.

For TR k (e_k = floor(fps * k * TR)):
  context = frames [e_k-8, e_k)            2 s before t_k (causal)
  future  = frames [e_k, e_k+16)           next 4 s = 8 slots of 0.5 s (oracle / target only)
Saved ([N] rows, NaN where invalid):
  hz_fore, hz_rand, hz_orac     [N, 8, D] float16   per-slot forecast / random predictor / oracle
  hz_surp_l1, hz_surp_cos, hz_surpr_l1, hz_surpr_cos, hz_chg_l1, hz_chg_cos   [N, 8]
      per-slot errors (cos stored as 1 - cos) moved to row k + hz_shift[j], the first TR whose onset
      is after slot j ends, so row k only uses frames before t_k
  hz_surp_valid [N, 8] bool
  q_cmid, q_clast, q_fore, q_rand, q_orac   [N, 4, D] float16   quadrant (2x2) means; future = first 2 s
  q_surp_l1, q_surp_cos, q_surpr_l1, q_surpr_cos, q_chg_l1, q_chg_cos   [N, 4]   shifted by q_shift (2 s window)
  q_surp_valid [N] bool, valid [N] bool, frame_end [N], hz_shift [8], q_shift

    python3 src/04b_extract_extra.py                 # test season first, then training seasons
    python3 src/04b_extract_extra.py --seasons 6
"""
import importlib
import os
import time

import numpy as np
import torch

from common import config_arg, ep_season, frame_end, frames_path, load_config, load_index, log, save_npz_atomic, surprise_shift
from vjepa import Normalizer, batch_features_extra, load_models, torch_dtype

load_frames = importlib.import_module("04_extract_vjepa").load_frames
VEC = ["hz_fore", "hz_rand", "hz_orac", "q_cmid", "q_clast", "q_fore", "q_rand", "q_orac"]
ERR = [f"{p}_{n}_{m}" for p in ("hz", "q") for n in ("fore", "rand", "copy") for m in ("l1", "cos")]


def run_episode(cfg, ep, n_tr, model, rand_pred, tps, norm):
    ex = cfg["extract"]
    fps, TR, CTX, BS = cfg["video"]["fps"], cfg["tr"], ex["ctx_frames"], ex["batch_trs"]
    FUT2 = cfg.get("extra", {}).get("fut_frames", 16)
    n_hz = FUT2 // 2
    Fr = load_frames(cfg, ep)
    nF, D = len(Fr), model.config.hidden_size
    e_all = np.array([frame_end(k, TR, fps) for k in range(n_tr)])
    valid = (e_all - CTX >= 0) & (e_all + FUT2 <= nF)
    shapes = {"hz": (n_hz,), "q": (4,)}
    out = {n: np.full((n_tr,) + shapes[n.split("_")[0]] + (D,), np.nan, np.float32) for n in VEC}
    out.update({n: np.full((n_tr,) + shapes[n.split("_")[0]], np.nan, np.float32) for n in ERR})
    ks = np.where(valid)[0]
    for i in range(0, len(ks), BS):
        kb = ks[i:i + BS]
        r = batch_features_extra(cfg, model, rand_pred, tps, norm, Fr, list(e_all[kb]))
        for n in VEC + ERR:
            out[n][kb] = r[n]
    for n, v in out.items():
        if not np.isfinite(v[valid]).all():
            raise FloatingPointError(f"{ep}: non-finite values in {n}; set model.dtype: bfloat16 and rerun")

    def shifted(a, s):
        b = np.full_like(a, np.nan)
        b[s:] = a[:-s]
        return b
    hz_shift = np.array([surprise_shift(TR, fps, 2 * (j + 1)) for j in range(n_hz)])
    q_shift = surprise_shift(TR, fps, ex["fut_frames"])
    for j in range(n_hz):   # slot j = frames [e_k+2j, e_k+2j+2) must end before t_{k+shift}
        assert all(e_all[k] + 2 * (j + 1) <= e_all[k + hz_shift[j]] for k in range(n_tr - hz_shift[j]))
    res = {n: out[n].astype(np.float16) for n in VEC}
    hz_sv = np.zeros((n_tr, n_hz), bool)
    for src, dst, inv in [("fore", "surp", False), ("rand", "surpr", False), ("copy", "chg", False)]:
        for m in ("l1", "cos"):
            a = out[f"hz_{src}_{m}"]
            a = 1 - a if m == "cos" else a
            res[f"hz_{dst}_{m}"] = np.stack([shifted(a[:, j], hz_shift[j]) for j in range(n_hz)], 1)
            b = out[f"q_{src}_{m}"]
            b = 1 - b if m == "cos" else b
            res[f"q_{dst}_{m}"] = shifted(b, q_shift)
    for j in range(n_hz):
        hz_sv[hz_shift[j]:, j] = valid[:-hz_shift[j]]
    q_sv = np.zeros(n_tr, bool)
    q_sv[q_shift:] = valid[:-q_shift]
    res.update({"hz_surp_valid": hz_sv, "q_surp_valid": q_sv, "valid": valid, "frame_end": e_all.astype(np.int64),
                "hz_shift": hz_shift.astype(np.int64), "q_shift": np.int64(q_shift)})
    return res, int(valid.sum())


def main():
    p = config_arg()
    p.add_argument("--seasons", type=int, nargs="*", default=None)
    p.add_argument("--episodes", nargs="*", default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    index = load_index(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not cfg["model"].get("tiny_random"):
        log(cfg, "extract_extra", "WARNING: no CUDA device; this will be extremely slow")
    model, rand_pred, tps = load_models(cfg, device)
    norm = Normalizer(device, torch_dtype(cfg["model"]["dtype"]))
    seasons = args.seasons or [cfg["test_season"]] + cfg["train_seasons"]
    eps = args.episodes or [ep for s in seasons for ep in sorted(index["episodes"]) if ep_season(ep) == s]
    for ep in eps:
        out_path = os.path.join(cfg["dirs"]["extra"], f"{ep}.npz")
        if os.path.exists(out_path):
            continue
        if not os.path.exists(frames_path(cfg, ep).replace(".mp4", ".json")):
            log(cfg, "extract_extra", f"{ep}: frames not re-encoded yet, skipping")
            continue
        t0 = time.time()
        res, nv = run_episode(cfg, ep, index["episodes"][ep]["n_tr_max"], model, rand_pred, tps, norm)
        save_npz_atomic(out_path, **res)
        dt = time.time() - t0
        log(cfg, "extract_extra", f"{ep}: {nv} valid TRs in {dt:.0f}s ({nv / max(dt, 1e-9):.1f} TR/s)")


if __name__ == "__main__":
    main()
