"""Step 06 - ONLY if step 05 said NO-GO: a linear past-only forecaster (CPU).

Forecast of the next 2 s:  oracle(k) ~ ridge( [c_last(k), c_last(k-1), c_last(k-2)] )
(all inputs use frames before t_k). Training seasons only; training-episode
forecasts are out-of-fold (GroupKFold by episode) so the encoding model never
sees in-sample fits; test-season forecasts come from the model fit on all
training episodes. Control: identical pipeline trained on targets circularly
shifted by >= 20 TRs within each episode (destroys the past->future mapping).

Writes <work>/features/ridgefc/<ep>.npz with hhat_ridge, hhat_ridge_shuf,
surp_ridge_l1, surp_ridge_cos, surp_ridge_shuf_l1, surp_ridge_shuf_cos,
valid_ridge, surp_valid_ridge.
"""
import glob
import os

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from common import config_arg, ep_season, load_config, log, save_npz_atomic

ALPHAS = np.logspace(-1, 5, 13)


def lagged(d):
    c, v = d["c_last"].astype(np.float32), d["valid"]
    N = len(v)
    X = np.zeros((N, 3 * c.shape[1]), np.float32)
    ok = np.zeros(N, bool)
    for k in range(2, N):
        if v[k] and v[k - 1] and v[k - 2]:
            X[k] = np.concatenate([c[k], c[k - 1], c[k - 2]])
            ok[k] = True
    return X, ok


def ln(x):
    x = x - x.mean(-1, keepdims=True)
    return x / np.maximum(x.std(-1, keepdims=True), 1e-6)


def main():
    args = config_arg().parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(0)
    files = sorted(glob.glob(os.path.join(cfg["dirs"]["vjepa"], "*.npz")))
    eps = [os.path.basename(f)[:-4] for f in files]
    data = {ep: np.load(f) for ep, f in zip(eps, files)}
    train = [ep for ep in eps if ep_season(ep) in cfg["train_seasons"]]
    Xs, Ys, Yshuf, G, idx = [], [], [], [], []
    for gi, ep in enumerate(train):
        X, ok = lagged(data[ep])
        Y = data[ep]["oracle"].astype(np.float32)
        rows = np.where(ok)[0]
        off = int(rng.integers(20, max(21, len(rows) - 20)))
        Xs.append(X[rows]); Ys.append(Y[rows]); Yshuf.append(np.roll(Y[rows], off, axis=0))
        G.append(np.full(len(rows), gi)); idx.append((ep, rows))
    X, Y, Ysh, G = np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Yshuf), np.concatenate(G)
    log(cfg, "fallback", f"training rows {len(X)} from {len(train)} episodes")
    sc = StandardScaler().fit(X)
    X = sc.transform(X).astype(np.float32)
    cv = list(GroupKFold(5).split(X, Y, G))

    def ridge_path(Xa, Ta):
        """Closed-form ridge (with intercept) for all ALPHAS via one eigendecomposition."""
        xm, tm = Xa.mean(0), Ta.mean(0)
        Xc = (Xa - xm).astype(np.float64)
        s, V = np.linalg.eigh(Xc.T @ Xc)
        VtXtT = V.T @ (Xc.T @ (Ta - tm).astype(np.float64))
        return lambda a: (V @ (VtXtT / (s + a)[:, None]), xm, tm)

    def apply(model, Xb):
        W, xm, tm = model
        return ((Xb - xm) @ W + tm).astype(np.float32)

    out_tr = {}
    for name, T in [("hhat_ridge", Y), ("hhat_ridge_shuf", Ysh)]:
        paths = [(ridge_path(X[tr], T[tr]), va) for tr, va in cv]
        errs = [sum(((apply(path(a), X[va]) - T[va]) ** 2).mean() for path, va in paths) for a in ALPHAS]
        a = ALPHAS[int(np.argmin(errs))]
        oof = np.zeros_like(T)
        for path, va in paths:
            oof[va] = apply(path(a), X[va])
        full_model = ridge_path(X, T)(a)
        full = type("RidgeFit", (), {"predict": staticmethod(lambda Xb, m=full_model: apply(m, Xb))})
        out_tr[name] = (oof, full)
        log(cfg, "fallback", f"{name}: alpha={a:g}  OOF R2={1 - ((oof - T) ** 2).sum() / ((T - T.mean(0)) ** 2).sum():.3f}")

    starts = np.cumsum([0] + [len(r) for _, r in idx])
    pos = {ep: (starts[i], starts[i + 1], rows) for i, (ep, rows) in enumerate(idx)}
    shift = None
    for ep in eps:
        d = data[ep]
        shift = int(d["shift"])
        Xe, ok = lagged(d)
        N = len(ok)
        res = {}
        for name in ["hhat_ridge", "hhat_ridge_shuf"]:
            H = np.full((N, Y.shape[1]), np.nan, np.float32)
            if ep in pos:
                a, b, rows = pos[ep]
                H[rows] = out_tr[name][0][a:b]
            elif ok.any():
                H[ok] = out_tr[name][1].predict(sc.transform(Xe[ok]))
            res[name] = H
        orc = d["oracle"].astype(np.float32)
        for name, key in [("hhat_ridge", "surp_ridge"), ("hhat_ridge_shuf", "surp_ridge_shuf")]:
            l1 = np.abs(ln(res[name]) - ln(orc)).mean(1)
            cs = (ln(res[name]) * ln(orc)).mean(1)
            for suf, v in [("_l1", l1), ("_cos", 1 - cs)]:
                s = np.full(N, np.nan, np.float32)
                s[shift:] = v[:-shift]
                res[key + suf] = s
        sv = np.zeros(N, bool)
        sv[shift:] = ok[:-shift]
        res = {k: (v.astype(np.float16) if v.ndim == 2 else v) for k, v in res.items()}
        save_npz_atomic(os.path.join(cfg["dirs"]["ridgefc"], f"{ep}.npz"), valid_ridge=ok, surp_valid_ridge=sv, **res)
    log(cfg, "fallback", f"wrote ridge forecasts for {len(eps)} episodes")


if __name__ == "__main__":
    main()
