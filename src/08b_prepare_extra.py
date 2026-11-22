"""Step 08b - regression bands for the horizon / spatial experiments (CPU).
Writes <work>/features/reduced_<exp>/<ep>.npz. Needs step 08 (main reduced features) and step 04b.

NUIS, CUR, LONG are copied from the main reduced features (already z-scored / PCA'd on the
training seasons). New bands (every transform FIT ON TRAINING SEASONS ONLY; scaler and PCA are fit
on a random subset of at most `max_fit_rows` training rows, which is plenty for 128-256 components):
  horizon, for every h in common.HORIZON_SLOTS (slot j = the 0.5 s ending at h):
    ORAC_h, FORE_h, RAND_h   per-slot oracle / forecast / random predictor -> z-score, PCA(pca_components)
    CHG_h, SURP_h, SURPR_h   [l1, 1-cos] of copy-forward / forecast / random predictor for slot j -> z-score
  spatial (2x2 quadrants, first 2 s of the future):
    CUR_q   [q_cmid, q_clast] (8 x D)          -> z-score, PCA(pca_components_quadrant)
    ORAC_q, FORE_q, RAND_q (4 x D)             -> z-score, PCA(pca_components_quadrant)
    CHG_q, SURP_q, SURPR_q  4 quadrants x [l1, 1-cos] -> z-score
Row validity = main validity AND extra validity AND (horizon) every slot's surprise valid /
(spatial) the quadrant surprise valid, identical for all bands of the experiment.

    python3 src/08b_prepare_extra.py --exp horizon
    python3 src/08b_prepare_extra.py --exp spatial
"""
import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import HORIZON_SLOTS, config_arg, ep_season, load_config, log, save_npz_atomic

COPY = ["NUIS", "CUR", "LONG"]


def sources(exp):
    """band -> (list of npz keys, function(arrays dict) -> [N, d])."""
    S = {}
    if exp == "horizon":
        for h, j in HORIZON_SLOTS.items():
            for band, key in [("ORAC", "hz_orac"), ("FORE", "hz_fore"), ("RAND", "hz_rand")]:
                S[f"{band}_{h}"] = ([key], lambda a, key=key, j=j: a[key][:, j])
            for band, pre in [("CHG", "hz_chg"), ("SURP", "hz_surp"), ("SURPR", "hz_surpr")]:
                S[f"{band}_{h}"] = ([f"{pre}_l1", f"{pre}_cos"],
                                    lambda a, pre=pre, j=j: np.stack([a[f"{pre}_l1"][:, j], a[f"{pre}_cos"][:, j]], 1))
    elif exp == "spatial":
        S["CUR_q"] = (["q_cmid", "q_clast"], lambda a: np.concatenate([a["q_cmid"].reshape(len(a["q_cmid"]), -1),
                                                                        a["q_clast"].reshape(len(a["q_clast"]), -1)], 1))
        for band, key in [("ORAC_q", "q_orac"), ("FORE_q", "q_fore"), ("RAND_q", "q_rand")]:
            S[band] = ([key], lambda a, key=key: a[key].reshape(len(a[key]), -1))
        for band, pre in [("CHG_q", "q_chg"), ("SURP_q", "q_surp"), ("SURPR_q", "q_surpr")]:
            S[band] = ([f"{pre}_l1", f"{pre}_cos"], lambda a, pre=pre: np.concatenate([a[f"{pre}_l1"], a[f"{pre}_cos"]], 1))
    else:
        raise SystemExit("--exp must be horizon or spatial")
    return S


def main():
    args = config_arg().parse_args()
    exp = args.exp
    cfg = load_config(args.config, exp)
    seasons = cfg["train_seasons"] + [cfg["test_season"]]
    lsd = lambda d: {f[:-4] for f in os.listdir(d) if f.endswith(".npz") and ".tmp" not in f and not f.startswith("_")}
    eps = sorted(ep for ep in lsd(cfg["dirs"]["reduced_main"]) & lsd(cfg["dirs"]["extra"]) if ep_season(ep) in seasons)
    train = [ep for ep in eps if ep_season(ep) in cfg["train_seasons"]]
    assert train, "no training episodes with both main reduced features (step 08) and extra features (step 04b)"
    S = sources(exp)
    path_x = lambda ep: os.path.join(cfg["dirs"]["extra"], f"{ep}.npz")

    # rows and validity
    out, valid, N = {}, {}, {}
    for ep in eps:
        m = np.load(os.path.join(cfg["dirs"]["reduced_main"], f"{ep}.npz"))
        x = np.load(path_x(ep))
        n = min(len(m["valid"]), len(x["valid"]))
        v = m["valid"][:n] & x["valid"][:n]
        v &= x["hz_surp_valid"][:n][:, list(HORIZON_SLOTS.values())].all(1) if exp == "horizon" else x["q_surp_valid"][:n]
        N[ep], valid[ep] = n, v
        out[ep] = {b: m[b][:n].copy() for b in COPY}
        out[ep]["cut_in_surp_raw"] = m["cut_in_surp_raw"][:n]
    n_valid_train = sum(int(valid[ep].sum()) for ep in train)
    max_rows = int(cfg.get("max_fit_rows", 60000))
    rng = np.random.default_rng(0)
    pick = {ep: (rng.random(N[ep]) < max_rows / n_valid_train) & valid[ep] for ep in train}

    # group bands by the npz keys they need, so each key is read twice per episode (fit, transform)
    groups = {}
    for band, (keys, fn) in S.items():
        groups.setdefault(tuple(keys), []).append(band)
    info = {}
    for keys, bands in groups.items():
        fit_rows = {b: [] for b in bands}
        for ep in train:
            x = np.load(path_x(ep))
            a = {k: x[k][:N[ep]].astype(np.float32) for k in keys}
            for b in bands:
                fit_rows[b].append(S[b][1](a)[pick[ep]])
        fitted = {}
        for b in bands:
            X = np.concatenate(fit_rows[b])
            assert np.isfinite(X).all(), f"non-finite training values in {b}"
            sc = StandardScaler().fit(X)
            if X.shape[1] > 16:
                k = cfg["pca_components_quadrant"] if b.endswith("_q") and "pca_components_quadrant" in cfg \
                    else cfg["encoding"]["pca_components"]
                k = min(k, X.shape[1], X.shape[0] - 1)
                pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(sc.transform(X))
                info[b] = {"n_components": int(k), "explained_variance": float(pca.explained_variance_ratio_.sum()),
                           "raw_dim": int(X.shape[1]), "fit_rows": int(len(X))}
            else:
                pca = None
                info[b] = {"n_columns": int(X.shape[1])}
            fitted[b] = (sc, pca)
        del fit_rows
        for ep in eps:
            x = np.load(path_x(ep))
            a = {k: x[k][:N[ep]].astype(np.float32) for k in keys}
            for b in bands:
                Z = fitted[b][0].transform(np.nan_to_num(S[b][1](a), nan=0.0))
                if fitted[b][1] is not None:
                    Z = fitted[b][1].transform(Z)
                Z = Z.astype(np.float32)
                Z[~valid[ep]] = 0.0
                out[ep][b] = Z
        log(cfg, "prepare_extra", f"{exp}: fitted {bands}")
    for ep in eps:
        for b in COPY:
            out[ep][b][~valid[ep]] = 0.0
        save_npz_atomic(os.path.join(cfg["dirs"]["reduced"], f"{ep}.npz"), valid=valid[ep], **out[ep])
    info.update({"n_train_episodes": len(train), "n_episodes": len(eps), "experiment": exp,
                 "valid_rows_train": n_valid_train})
    with open(os.path.join(cfg["dirs"]["reduced"], "_bands.json"), "w") as f:
        json.dump(info, f, indent=1)
    log(cfg, "prepare_extra", f"{exp}: {len(eps)} episodes ({len(train)} train); {json.dumps(info)}")


if __name__ == "__main__":
    main()
