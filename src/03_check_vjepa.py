"""Step 03 - one-off checks of the real V-JEPA 2 model on one re-encoded episode (GPU, ~10 min).

Checks, each with a hard pass/fail:
 A  model config is ViT-L/16, 256 px, tubelet 2, 24 layers, hidden 1024
 B  our encoder-only call == VJEPA2Model.forward(..., skip_predictor=True)
 C  leakage: the built-in VJEPA2Model.forward forecast CHANGES when only the
    future frames change (it leaks), ours does NOT (it cannot see them)
 D  positions: each predicted token matches the target token at the same
    (h,w) location better than a spatially shuffled target, in >= 90% of clips
 E  benchmark: TRs/s at the configured batch size -> estimated GPU hours for
    all indexed episodes; peak GPU memory
Also prints a preview of forecast skill (forecast vs copy-forward vs random).

    python src/03_check_vjepa.py --episode s06e01a
"""
import json
import time

import numpy as np
import torch

from common import config_arg, ep_season, frame_end, load_config, load_index, log
from vjepa import Normalizer, batch_features, encode, load_models, masks, predict, torch_dtype

import importlib
extract = importlib.import_module("04_extract_vjepa")


def main():
    p = config_arg()
    p.add_argument("--episode", default=None)
    p.add_argument("--n-clips", type=int, default=32)
    args = p.parse_args()
    cfg = load_config(args.config)
    index = load_index(cfg)
    ep = args.episode or sorted(e for e in index["episodes"] if ep_season(e) == cfg["test_season"])[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, rand_pred, tps = load_models(cfg, dev)
    dt = torch_dtype(cfg["model"]["dtype"])
    norm = Normalizer(dev, dt)
    ex = cfg["extract"]
    CTX, FUT, LONG = ex["ctx_frames"], ex["fut_frames"], ex["long_frames"]
    cs, fs = CTX // 2, FUT // 2
    c = model.config
    report = {}

    # A
    if not cfg["model"].get("tiny_random"):
        assert (c.hidden_size, c.num_hidden_layers, c.patch_size, c.crop_size, c.tubelet_size) == (1024, 24, 16, 256, 2), c
        assert tps == 256
    report["A_config"] = "pass"

    Fr = extract.load_frames(cfg, ep)
    n_tr = index["episodes"][ep]["n_tr_max"]
    E_ok = [frame_end(k, cfg["tr"], cfg["video"]["fps"]) for k in range(n_tr)]
    E_ok = [e for e in E_ok if e - LONG >= 0 and e + 2 * FUT <= len(Fr)]
    rng = np.random.default_rng(0)
    E = sorted(rng.choice(E_ok, size=min(args.n_clips, len(E_ok)), replace=False).tolist())

    with torch.inference_mode():
        # B
        x = norm(np.stack([Fr[e - CTX:e] for e in E[:4]]))
        ours, _ = encode(model, x)
        ref = model(pixel_values_videos=x, skip_predictor=True).last_hidden_state
        assert torch.allclose(ours.float(), ref.float(), atol=1e-3), "encoder path mismatch"
        report["B_encoder_equivalence"] = "pass"

        # C
        full = norm(np.stack([Fr[e - CTX:e + FUT] for e in E[:4]]))
        alt = full.clone()
        alt[:, CTX:] = alt[:, CTX:].flip(0) if alt.shape[0] > 1 else torch.zeros_like(alt[:, CTX:])
        cm, tm = masks(4 if len(E) >= 4 else len(E), cs, range(cs, cs + fs), tps, dev)
        leak_a = model(pixel_values_videos=full, context_mask=cm, target_mask=tm).predictor_output.last_hidden_state
        leak_b = model(pixel_values_videos=alt, context_mask=cm, target_mask=tm).predictor_output.last_hidden_state
        hp, _ = encode(model, full[:, :CTX].contiguous())
        hq, _ = encode(model, alt[:, :CTX].contiguous())
        ours_a, ours_b = predict(model.predictor, hp, cm, tm), predict(model.predictor, hq, cm, tm)
        d_leak = (leak_a - leak_b).abs().max().item()
        d_ours = (ours_a - ours_b).abs().max().item()
        assert d_ours == 0.0, "our forecast changed when only future frames changed"
        assert d_leak > 0.0, "expected the built-in forward to leak"
        report["C_leakage"] = {"builtin_forward_maxdiff": d_leak, "ours_maxdiff": d_ours, "result": "pass"}

        # D
        r = batch_features(cfg, model, rand_pred, tps, norm, Fr, E, return_tokens=True)
        P, Tn = r["_tokens"]["P"].float(), r["_tokens"]["Tn"].float()
        g = int(round(tps ** 0.5))
        perm = torch.randperm(tps, generator=torch.Generator().manual_seed(0))
        while (perm == torch.arange(tps)).any():
            perm = torch.randperm(tps)
        Tp = Tn.view(len(E), fs, tps, -1)[:, :, perm].reshape_as(Tn)
        matched = torch.nn.functional.cosine_similarity(P, Tn, -1).mean(1)
        shuffled = torch.nn.functional.cosine_similarity(P, Tp, -1).mean(1)
        frac = (matched > shuffled).float().mean().item()
        report["D_positions"] = {"matched_cos": matched.mean().item(), "shuffled_cos": shuffled.mean().item(),
                                 "fraction_clips_matched_better": frac,
                                 "result": "pass" if frac >= 0.9 else "FAIL"}
        report["skill_preview"] = {k: float(np.mean(r[k])) for k in
                                   ["fore_cos", "copy_cos", "rand_cos", "fore_l1", "copy_l1", "rand_l1"]}

        # E
        if dev == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        BS = ex["batch_trs"]
        Eb = (E_ok * 2)[:BS * 6]
        t0 = time.time()
        for i in range(0, len(Eb), BS):
            batch_features(cfg, model, rand_pred, tps, norm, Fr, Eb[i:i + BS])
        if dev == "cuda":
            torch.cuda.synchronize()
        rate = len(Eb) / (time.time() - t0)
        total = sum(e["n_tr_max"] for e in index["episodes"].values())
        report["E_benchmark"] = {"TR_per_s": rate, "indexed_TRs": total, "est_gpu_hours_all_indexed": total / rate / 3600,
                                 "peak_mem_GB": torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else None,
                                 "rule": "if est_gpu_hours exceeds your GPU time, set extract.long_frames: 16 and rerun this check"}

    print(json.dumps(report, indent=1))
    with open(f"{cfg['dirs']['logs']}/check_vjepa.json", "w") as f:
        json.dump(report, f, indent=1)
    if report["D_positions"]["result"] != "pass" and not cfg["model"].get("tiny_random"):
        log(cfg, "check_vjepa", "FAIL D: position check failed -> do not extract; send check_vjepa.json")
        raise SystemExit(1)
    if cfg["model"].get("tiny_random"):
        log(cfg, "check_vjepa", "tiny test model: check D is not enforced (random weights)")
    else:
        log(cfg, "check_vjepa", "all checks passed")


if __name__ == "__main__":
    main()
