"""Step 08 - build the regression bands per episode (CPU). <work>/features/reduced/<ep>.npz

Bands (raw source -> transform; every transform is FIT ON TRAINING SEASONS ONLY):
  NUIS  7 nuisance regressors                       -> z-score
  CUR   [c_mid, c_last] (2048-d)                    -> z-score, PCA(k)
  LONG  clong                                       -> z-score, PCA(k)
  ORAC  oracle                                      -> z-score, PCA(k)
  FORE  hhat           (or hhat_ridge if fallback)  -> z-score, PCA(k)
  RAND  hhat_rand      (or hhat_ridge_shuf)         -> z-score, PCA(k)
  CHG   [chg_l1, chg_cos]   latent change (copy-forward error)  -> z-score
  SURP  [surp_l1, surp_cos] (or ridge versions)                 -> z-score
  SURPR [surp_rand_l1, surp_rand_cos] (or ridge shuffled)       -> z-score
Row validity `valid` = all sources valid at that row (identical for every band,
so every encoding model is fit and scored on exactly the same rows).
Invalid rows are written as 0 (they are later excluded, and also cannot leak
through delays because rows whose delayed sources are invalid are dropped).
"""
import json
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import config_arg, ep_season, load_config, log, save_npz_atomic

NUIS = ["cuts_count", "lum", "fdiff", "rms", "n_words", "speech", "cut_in_surp"]
PCA_BANDS = ["CUR", "LONG", "ORAC", "FORE", "RAND"]
SCALAR_BANDS = ["NUIS", "CHG", "SURP", "SURPR"]


def raw_bands(cfg, ep):
    d = np.load(os.path.join(cfg["dirs"]["vjepa"], f"{ep}.npz"))
    n = np.load(os.path.join(cfg["dirs"]["nuisance"], f"{ep}.npz"))
    N = min(len(d["valid"]), len(n["valid_nuis"]))
    f32 = lambda a: a[:N].astype(np.float32)
    b = {
        "NUIS": np.stack([f32(n[k]) for k in NUIS], 1),
        "CUR": np.concatenate([f32(d["c_mid"]), f32(d["c_last"])], 1),
        "LONG": f32(d["clong"]), "ORAC": f32(d["oracle"]),
        "CHG": np.stack([f32(d["chg_l1"]), f32(d["chg_cos"])], 1),
    }
    valid = d["valid"][:N] & d["surp_valid"][:N] & n["valid_nuis"][:N]
    cut_raw = n["cut_in_surp"][:N] > 0
    if cfg["encoding"]["forecast_source"] == "ridge":
        r = np.load(os.path.join(cfg["dirs"]["ridgefc"], f"{ep}.npz"))
        b["FORE"], b["RAND"] = f32(r["hhat_ridge"]), f32(r["hhat_ridge_shuf"])
        b["SURP"] = np.stack([f32(r["surp_ridge_l1"]), f32(r["surp_ridge_cos"])], 1)
        b["SURPR"] = np.stack([f32(r["surp_ridge_shuf_l1"]), f32(r["surp_ridge_shuf_cos"])], 1)
        valid &= r["valid_ridge"][:N] & r["surp_valid_ridge"][:N]
    else:
        b["FORE"], b["RAND"] = f32(d["hhat"]), f32(d["hhat_rand"])
        b["SURP"] = np.stack([f32(d["surp_l1"]), f32(d["surp_cos"])], 1)
        b["SURPR"] = np.stack([f32(d["surp_rand_l1"]), f32(d["surp_rand_cos"])], 1)
    for k, v in b.items():
        assert np.isfinite(v[valid]).all(), f"{ep}: non-finite {k} on valid rows"
    return b, valid, cut_raw


def main():
    args = config_arg().parse_args()
    cfg = load_config(args.config)
    seasons = cfg["train_seasons"] + [cfg["test_season"]]
    need = [cfg["dirs"]["vjepa"], cfg["dirs"]["nuisance"]]
    if cfg["encoding"]["forecast_source"] == "ridge":
        need.append(cfg["dirs"]["ridgefc"])
    eps = sorted(set.intersection(*[{f[:-4] for f in os.listdir(dd) if f.endswith(".npz") and ".tmp" not in f}
                                    for dd in need]))
    eps = [ep for ep in eps if ep_season(ep) in seasons]
    train = [ep for ep in eps if ep_season(ep) in cfg["train_seasons"]]
    assert train, "no training episodes have features yet"
    raw = {ep: raw_bands(cfg, ep) for ep in eps}
    cuts = {ep: raw[ep][2] for ep in eps}
    k = cfg["encoding"]["pca_components"]
    fitted, info = {}, {}
    for band in PCA_BANDS + SCALAR_BANDS:
        Xtr = np.concatenate([raw[ep][0][band][raw[ep][1]] for ep in train])
        sc = StandardScaler().fit(Xtr)
        if band in PCA_BANDS:
            nk = min(k, Xtr.shape[1], Xtr.shape[0] - 1)
            pca = PCA(n_components=nk, svd_solver="randomized", random_state=0).fit(sc.transform(Xtr))
            fitted[band] = (sc, pca)
            info[band] = {"n_components": int(nk), "explained_variance": float(pca.explained_variance_ratio_.sum())}
        else:
            fitted[band] = (sc, None)
            info[band] = {"n_columns": int(Xtr.shape[1])}
    for ep in eps:
        b, valid, _ = raw[ep]
        out = {}
        for band, (sc, pca) in fitted.items():
            X = np.nan_to_num(b[band], nan=0.0)
            Z = sc.transform(X)
            if pca is not None:
                Z = pca.transform(Z)
            Z = Z.astype(np.float32)
            Z[~valid] = 0.0
            out[band] = Z
        save_npz_atomic(os.path.join(cfg["dirs"]["reduced"], f"{ep}.npz"), valid=valid, cut_in_surp_raw=cuts[ep], **out)
    info["n_train_episodes"], info["n_episodes"] = len(train), len(eps)
    info["forecast_source"] = cfg["encoding"]["forecast_source"]
    with open(os.path.join(cfg["dirs"]["reduced"], "_bands.json"), "w") as f:
        json.dump(info, f, indent=1)
    log(cfg, "prepare", f"{len(eps)} episodes ({len(train)} train); {json.dumps(info)}")


if __name__ == "__main__":
    main()
