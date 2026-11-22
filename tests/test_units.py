"""Unit tests (CPU, no data, no download). Run:  python tests/test_units.py
  1 leakage: VJEPA2Model.forward leaks future frames into the forecast; our path does not
  2 encoder-only call equals VJEPA2Model.forward(skip_predictor=True)
  3 predictor accepts target indices beyond the encoder sequence; output shape
  4 random predictor differs from trained one, encoder untouched
  5 surprise_shift never leaks for many TR/fps/horizon combinations
  6 delay_ep never crosses episode start
  7 bootstrap R2 from sufficient statistics == direct R2; BH == statsmodels
  8 ridge_cv == scikit-learn Ridge + GroupKFold (alphas and predictions)
  9 two-group ridge (gamma on the tested band) == brute force; gamma = 0 nests the reduced model
"""
import importlib
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import frame_end, surprise_shift  # noqa: E402
from vjepa import encode, load_models, masks, predict  # noqa: E402

fit = importlib.import_module("09_fit_encoding")
stats = importlib.import_module("11_stats")

cfg = {"model": {"tiny_random": True, "dtype": "float32", "layer_mid": 8},
       "video": {"size": 32}, "extract": {"seed_rand_predictor": 1234}}
model, rand, tps = load_models(cfg, "cpu")
g = torch.Generator().manual_seed(0)
full = torch.randn(2, 16, 3, 32, 32, generator=g)
alt = full.clone()
alt[:, 8:] = torch.randn(2, 8, 3, 32, 32, generator=g)
cm, tm = masks(2, 4, range(4, 8), tps, "cpu")
with torch.inference_mode():
    a = model(pixel_values_videos=full, context_mask=cm, target_mask=tm).predictor_output.last_hidden_state
    b = model(pixel_values_videos=alt, context_mask=cm, target_mask=tm).predictor_output.last_hidden_state
    assert (a - b).abs().max() > 1e-4, "expected built-in forward to leak"
    ha, _ = encode(model, full[:, :8].contiguous())
    hb, _ = encode(model, alt[:, :8].contiguous())
    pa, pb = predict(model.predictor, ha, cm, tm), predict(model.predictor, hb, cm, tm)
    assert torch.equal(pa, pb), "our forecast must not depend on future frames"
    print("1 leakage test ok: built-in diff", float((a - b).abs().max()), "ours diff", float((pa - pb).abs().max()))
    ref = model(pixel_values_videos=full[:, :8], skip_predictor=True).last_hidden_state
    assert torch.allclose(ref, ha, atol=1e-6)
    print("2 encoder equivalence ok")
    assert ha.shape[1] == 4 * tps and int(tm[0].max()) >= ha.shape[1]
    assert pa.shape == (2, 4 * tps, model.config.hidden_size)
    print("3 predictor shape ok", tuple(pa.shape))
    assert not torch.equal(rand.proj.weight, model.predictor.proj.weight)
    print("4 random predictor ok")

for tr in [1.49, 1.0, 2.0, 0.8]:
    for fps in [4, 8]:
        for fut in [2, 4, 8, 16]:
            s = surprise_shift(tr, fps, fut)
            for k in range(500):
                assert frame_end(k, tr, fps) + fut <= frame_end(k + s, tr, fps)
assert surprise_shift(1.49, 4, 8) == 2
print("5 surprise shift ok (shift=2 for TR 1.49 s, 4 fps, 8 frames)")

X = np.arange(1, 11, dtype=np.float32)[:, None]
D = fit.delay_ep(X, [1, 2, 3])
assert (D[0] == 0).all() and D[1, 0] == 1 and D[3, 2] == 1 and D[2, 2] == 0
print("6 delays ok")

rng = np.random.default_rng(0)
eps = np.repeat(np.array(["a", "b", "c", "d"]), 50)
Y = rng.standard_normal((200, 5))
P = Y * 0.5 + rng.standard_normal((200, 5))
ue, n, s_, ss, sse = fit.suff_stats(Y, P, eps)
r2_direct, _ = fit.r2_and_r(Y, P)
r2_ss = stats.r2_boot(np.ones((1, 4)), {"n": n, "sum_y": s_, "sumsq_y": ss, "sse": sse})[0]
assert np.allclose(r2_direct, r2_ss)
from statsmodels.stats.multitest import multipletests  # noqa: E402
p = rng.uniform(size=1000) ** 2
assert np.allclose(stats.bh(p), multipletests(p, method="fdr_bh")[1])
print("7 sufficient-statistics R2 and BH-FDR ok")
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
X = rng.standard_normal((1000, 40)).astype("float32")
Y = (X @ rng.standard_normal((40, 20)) * 0.2 + rng.standard_normal((1000, 20))).astype("float32")
g = np.repeat(np.arange(10), 100)
alphas = np.logspace(-2, 6, 9)
pred, al, _ = fit.ridge_cv(X[:800], Y[:800], g[:800], X[800:], alphas, 5, "cpu")
sse = np.zeros((9, 20))
for tr, va in GroupKFold(5).split(X[:800], Y[:800], g[:800]):
    for i, a in enumerate(alphas):
        sse[i] += ((Ridge(alpha=a, fit_intercept=False).fit(X[tr], Y[tr]).predict(X[va]) - Y[va]) ** 2).sum(0)
assert np.allclose(al, alphas[sse.argmin(0)])
ref = np.stack([Ridge(alpha=a, fit_intercept=False).fit(X[:800], Y[:800, j]).predict(X[800:]) for j, a in enumerate(al)], 1)
assert np.allclose(pred, ref, atol=1e-3)
print("8 ridge solver == sklearn (per-target CV alphas and predictions)")
# 9 two-group ridge: last 10 columns = tested band, scaled by gamma; brute force with sklearn
Xb = rng.standard_normal((1000, 10)).astype("float32")
Yb = Y + (Xb @ rng.standard_normal((10, 20)) * np.r_[np.zeros(10), np.full(10, 0.3)]).astype("float32")
XX = np.concatenate([X, Xb], 1)
mask = np.r_[np.zeros(40, bool), np.ones(10, bool)]
gams = [0, 0.5, 1, 2]
pred, al, gm = fit.ridge_cv(XX[:800], Yb[:800], g[:800], XX[800:], alphas, 5, "cpu", scale_cols=mask, gammas=gams)
sse = np.zeros((len(gams) * 9, 20))
for tr, va in GroupKFold(5).split(XX[:800], Yb[:800], g[:800]):
    for j, gg in enumerate(gams):
        Xs = XX * np.where(mask, gg, 1).astype("float32")
        for i, a in enumerate(alphas):
            sse[j * 9 + i] += ((Ridge(alpha=a, fit_intercept=False).fit(Xs[tr], Yb[tr]).predict(Xs[va]) - Yb[va]) ** 2).sum(0)
b = sse.argmin(0)
assert np.allclose(gm, np.array(gams)[b // 9]) and np.allclose(al, alphas[b % 9])
ref = np.stack([Ridge(alpha=al[t], fit_intercept=False).fit(XX[:800] * np.where(mask, gm[t], 1), Yb[:800, t])
                .predict(XX[800:] * np.where(mask, gm[t], 1)) for t in range(20)], 1)
assert np.allclose(pred, ref, atol=1e-3)
assert (gm[:10] == 0).mean() >= 0.5 and (gm[10:] > 0).all(), (gm[:10], gm[10:])  # band helps only targets 10-19
p0, a0_, _ = fit.ridge_cv(XX[:800], Yb[:800], g[:800], XX[800:], alphas, 5, "cpu", scale_cols=mask, gammas=[0])
p1, a1_, _ = fit.ridge_cv(X[:800], Yb[:800], g[:800], X[800:], alphas, 5, "cpu")
assert np.allclose(p0, p1, atol=1e-4) and np.allclose(a0_, a1_)  # gamma = 0 == model without the band
print("9 two-group ridge == sklearn brute force; gamma=0 nests the reduced model; band picked where it helps")
print("ALL UNIT TESTS PASSED")
