"""Unit tests for the extra experiments (CPU, tiny random V-JEPA 2, no data). Run:  python3 tests/test_extra.py
  1 horizon/quadrant forecasts and context features do not depend on future frames; oracle does
  2 quadrant means average to the whole-frame mean; per-slot means average to the 2 s mean
  3 per-slot surprise shifts never leak (every TR / slot)
  4 banded_fast candidates: equal draw = all ones; gamma = 0 candidates of 'full' == candidates of 'full_noF'
  5 ridge_cv_scales == brute-force scikit-learn over candidates x alphas
  6 experiment overlays load, and horizon models / contrasts are generated
"""
import importlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import HORIZON_SLOTS, frame_end, horizon_models_contrasts, surprise_shift  # noqa: E402
from vjepa import Normalizer, batch_features_extra, load_models, quadrant_pool  # noqa: E402

fit = importlib.import_module("09_fit_encoding")
cfg = {"model": {"tiny_random": True, "dtype": "float32", "layer_mid": 8}, "video": {"size": 64},
       "extract": {"seed_rand_predictor": 1234, "ctx_frames": 8, "fut_frames": 8}, "extra": {"fut_frames": 16}}
model, rand, tps = load_models(cfg, "cpu")
norm = Normalizer("cpu", torch.float32)
rng = np.random.default_rng(0)
Fr = rng.integers(0, 255, (80, 64, 64, 3), dtype=np.uint8)
Fr2 = Fr.copy()
E = [8, 34, 60]   # future windows never overlap another TR's context
for e in E:
    Fr2[e:e + 16] = rng.integers(0, 255, (16, 64, 64, 3), dtype=np.uint8)   # change only the future frames
a = batch_features_extra(cfg, model, rand, tps, norm, Fr, E)
b = batch_features_extra(cfg, model, rand, tps, norm, Fr2, E)
for k in ["hz_fore", "hz_rand", "q_cmid", "q_clast", "q_fore", "q_rand"]:
    assert np.array_equal(a[k], b[k]), f"{k} depends on future frames"
for k in ["hz_orac", "q_orac", "hz_fore_l1", "q_fore_l1"]:
    assert not np.allclose(a[k], b[k]), f"{k} should depend on the future"
assert a["hz_fore"].shape == (3, 8, model.config.hidden_size) and a["q_fore"].shape == (3, 4, model.config.hidden_size)
print("1 no leakage in horizon / quadrant forecasts; oracle and errors use the future")

x = torch.randn(2, 3 * tps, 5)
assert torch.allclose(quadrant_pool(x, 3, int(tps ** .5)).mean(1), x.mean(1), atol=1e-5)
assert np.allclose(a["q_fore"].mean(1), a["hz_fore"][:, :4].mean(1), atol=1e-4)
assert np.allclose(a["q_orac"].mean(1), a["hz_orac"][:, :4].mean(1), atol=1e-4)
print("2 quadrant and slot pooling consistent")

for tr in [1.49, 1.0, 2.0]:
    for j in range(8):
        s = surprise_shift(tr, 4, 2 * (j + 1))
        assert all(frame_end(k, tr, 4) + 2 * (j + 1) <= frame_end(k + s, tr, 4) for k in range(600))
print("3 per-slot surprise shifts never leak; shifts for TR 1.49:", [surprise_shift(1.49, 4, 2 * (j + 1)) for j in range(8)])

bcfg = {"encoding": {"banded": {"n_draws": 5, "gammas": [0, 0.5, 1, 2], "seed": 0}}}
w = [3, 4, 2, 5]
Sf, gf = fit.banded_scales(bcfg, ["NUIS", "CUR", "LONG", "FORE"], w)
Sn, gn = fit.banded_scales(bcfg, ["NUIS", "CUR", "LONG"], w[:3])
assert np.allclose(Sf[0], np.r_[np.ones(9), np.zeros(5)]) and np.allclose(Sn[0], 1)
assert np.allclose(Sf[gf == 0][:, :9], Sn), "gamma=0 candidates of full must equal the noF candidates"
assert Sf.shape == (20, 14) and np.isnan(gn).all()
print("4 banded candidates nest the reduced model")

from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
X = rng.standard_normal((600, 14)).astype("float32")
Y = (X @ rng.standard_normal((14, 6)) * 0.3 + rng.standard_normal((600, 6))).astype("float32")
g = np.repeat(np.arange(12), 50)
al = np.logspace(-2, 4, 7)
pred, alc, ci = fit.ridge_cv_scales(X[:500], Y[:500], g[:500], X[500:], al, 4, "cpu", Sf)
sse = np.zeros((len(Sf) * len(al), 6))
for tr, va in GroupKFold(4).split(X[:500], Y[:500], g[:500]):
    for j, m in enumerate(Sf):
        Xs = X * m.astype("float32")
        for i, a_ in enumerate(al):
            sse[j * len(al) + i] += ((Ridge(alpha=a_, fit_intercept=False).fit(Xs[tr], Y[tr]).predict(Xs[va]) - Y[va]) ** 2).sum(0)
bst = sse.argmin(0)
assert np.array_equal(ci, bst // len(al)) and np.allclose(alc, al[bst % len(al)])
ref = np.stack([Ridge(alpha=alc[t], fit_intercept=False).fit(X[:500] * Sf[ci[t]], Y[:500, t]).predict(X[500:] * Sf[ci[t]])
                for t in range(6)], 1)
assert np.allclose(pred, ref, atol=1e-3)
print("5 ridge_cv_scales == scikit-learn brute force")

M, C, H = horizon_models_contrasts(list(HORIZON_SLOTS))
assert len(M) == 1 + 6 * len(HORIZON_SLOTS) and len(C) == 6 * len(HORIZON_SLOTS) and len(H) == 2 * len(HORIZON_SLOTS)
assert all(set(m) <= set(M) for m in C.values())
assert fit.tested_band({"encoding": {}}, M["full_h20"]) == "FORE_h20"
assert fit.tested_band({"encoding": {}}, ["NUIS", "CUR_q", "SURPR_q"]) == "SURPR_q"
assert fit.tested_band({"encoding": {}}, M["full_noF_h20"]) is None
print("6 horizon models / contrasts and tested-band detection ok")
print("ALL EXTRA TESTS PASSED")
