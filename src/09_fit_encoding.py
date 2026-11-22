"""Step 09 - banded-ridge encoding models per subject (GPU via himalaya torch_cuda, or CPU).

For each subject and each model in config `models`:
  * Y: fMRI runs z-scored per run; X: reduced bands, delayed by `delays` TRs
    INSIDE each episode (no delay ever crosses an episode boundary).
  * Rows kept: valid feature rows whose every delayed source row is valid.
  * Train = training seasons, test = test season (whole episodes).
  * solver "ridge" (default): ridge with one penalty per parcel chosen by
    GroupKFold(5) by episode, closed form on GPU (see ridge_cv). The model's
    tested band (encoding.tested_bands: FORE/RAND/SURP/SURPR) gets its own
    weight gamma in {0, 0.25, ..., 4} per parcel, chosen by the same CV, so a
    model with the band nests the model without it (gamma = 0). Solver
    "banded": himalaya BandedRidgeCV(groups="input"), one weight per band; it is
    ~n_iter times slower and did not fit the time budget (Gate-2 pilot: 42 min
    on CPU for the 2-band base model at n_iter=10).
Outputs <work>/results/<sub>/<model>.npz with per-test-episode sufficient
statistics (n, sum y, sum y^2, SSE per parcel) for the bootstrap in step 10,
plus overall R2 / Pearson r per parcel and the fitted band weights.

    python src/09_fit_encoding.py --benchmark            # time one small fit first
    python src/09_fit_encoding.py --subjects sub-01 --models base
    python src/09_fit_encoding.py                        # everything
"""
import os
import time

import numpy as np
from himalaya.backend import set_backend
from himalaya.ridge import BandedRidgeCV, ColumnTransformerNoStack
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from common import config_arg, ep_season, load_config, load_fmri_friends, load_index, log, zscore_run, save_npz_atomic


def delay_ep(X, delays):
    out = np.zeros((X.shape[0], X.shape[1] * len(delays)), np.float32)
    for j, d in enumerate(delays):
        out[d:, j * X.shape[1]:(j + 1) * X.shape[1]] = X[:-d] if d > 0 else X
    return out


class LazyBands(dict):
    """band -> delayed design matrix, loaded from the reduced files on first access (so only the
    bands of the models being fit are ever in memory; main() drops bands the next model does not use)."""

    def __init__(self, loader):
        super().__init__()
        self.loader = loader

    def __missing__(self, band):
        v = self.loader(band)
        self[band] = v
        return v


def build_subject(cfg, sub, within_shot=False):
    index = load_index(cfg)
    delays = cfg["encoding"]["delays"]
    seasons = cfg["train_seasons"] + [cfg["test_season"]]
    fm, _ = load_fmri_friends(cfg, sub, seasons)
    eps = [ep for ep in index["subjects"][sub]["episodes"]
           if ep_season(ep) in seasons and os.path.exists(os.path.join(cfg["dirs"]["reduced"], f"{ep}.npz"))]
    data = {p: {"Y": [], "g": [], "ep": [], "k": [], "_rows": []} for p in ("train", "test")}
    for gi, ep in enumerate(sorted(eps)):
        path = os.path.join(cfg["dirs"]["reduced"], f"{ep}.npz")
        R = np.load(path)
        Y = zscore_run(fm[ep])
        N = min(len(Y), len(R["valid"]))
        v = R["valid"][:N]
        ok = np.zeros(N, bool)
        for k in range(max(delays), N):
            ok[k] = all(v[k - d] for d in delays)
        if within_shot:   # robustness: drop rows whose surprise clip contains a cut (applies to train and test)
            ok &= ~R["cut_in_surp_raw"][:N]
        part = "test" if ep_season(ep) == cfg["test_season"] else "train"
        D = data[part]
        D["_rows"].append((path, N, ok))
        D["Y"].append(Y[:N][ok])
        D["g"].append(np.full(ok.sum(), gi))
        D["ep"].append(np.array([ep] * ok.sum()))
        D["k"].append(np.where(ok)[0])
    for part in data.values():
        rows = part.pop("_rows")
        part["X"] = LazyBands(lambda band, rows=rows: np.concatenate(
            [delay_ep(np.load(path)[band][:N], delays)[ok] for path, N, ok in rows]))
        part["Y"] = np.concatenate(part["Y"]).astype(np.float32)
        part["g"] = np.concatenate(part["g"])
        part["ep"] = np.concatenate(part["ep"])
        part["k"] = np.concatenate(part["k"])
    return data


def ridge_cv_scales(Xtr, Ytr, groups, Xte, alphas, n_folds, device, scales):
    """Ridge without intercept; per target (parcel) a penalty alpha AND one candidate column scaling,
    both chosen by GroupKFold CV over the grid candidates x alphas.

    scales [C, p]: candidate per-column multipliers. Scaling a band's columns by m gives that band
    the penalty alpha/m^2, so a list of candidates is a (random or grid) search over band penalties
    (banded ridge); m = 0 removes a band exactly. Ties go to the earlier candidate.
    Closed form per candidate via eigendecomposition of D X'X D (float64; D = diag(m)), predictions
    in float32: W(a) = D V diag(1/(s+a)) V' D X'Y. X'X is formed once; fold Grams = total - validation.
    Returns test predictions, chosen alpha per target, chosen candidate index per target."""
    import torch
    dev = torch.device(device)
    X = torch.from_numpy(Xtr).to(dev)
    Y = torch.from_numpy(Ytr).to(dev)
    A = torch.tensor(alphas, dtype=torch.float64, device=dev)
    Sc = torch.as_tensor(np.asarray(scales, np.float64), device=dev)
    assert Sc.ndim == 2 and Sc.shape[1] == X.shape[1], (Sc.shape, X.shape)
    nA, nC = len(alphas), Sc.shape[0]
    sse = torch.zeros(nC * nA, Y.shape[1], dtype=torch.float64, device=dev)

    def gram(Xa, Ya):
        return (Xa.T @ Xa).double(), (Xa.T @ Ya).double()

    def decompose(G, C, f):
        # scaling columns by f: X'X -> D G D, X'Y -> D C (no new pass over the rows)
        s, V = torch.linalg.eigh(f[:, None] * G * f[None, :])
        return s.clamp_min(0), V, V.T @ (f[:, None] * C)

    G_all, C_all = gram(X, Y)          # one pass over the rows; fold Grams are G_all - G_val
    for tr, va in GroupKFold(n_folds).split(Xtr, Ytr, groups):
        va_t = torch.from_numpy(va).to(dev)
        Xv, Yv = X[va_t], Y[va_t]
        Gv, Cv = gram(Xv, Yv)
        G_tr, C_tr = G_all - Gv, C_all - Cv
        del Gv, Cv
        for j in range(nC):
            f = Sc[j]
            s, V, At = decompose(G_tr, C_tr, f)
            B = Xv @ (f[:, None] * V).float()
            for i in range(nA):
                P = B @ (At / (s + A[i])[:, None]).float()
                sse[j * nA + i] += ((Yv - P) ** 2).sum(0).double()
            del B, s, V, At
        del Xv, Yv, G_tr, C_tr
    best = sse.argmin(0)                       # flat index j*nA + i, first minimum wins
    bc, ba = best // nA, best % nA
    Xte_t = torch.from_numpy(Xte).to(dev)
    pred = torch.zeros(Xte.shape[0], Y.shape[1], dtype=torch.float32, device=dev)
    for j in torch.unique(bc).tolist():
        cols = torch.where(bc == j)[0]
        f = Sc[j]
        s, V, At = decompose(G_all, C_all[:, cols], f)
        W = (f[:, None] * (V @ (At / (s[:, None] + A[ba[cols]][None, :])))).float()
        pred[:, cols] = Xte_t @ W
        del s, V, At, W
    return pred.cpu().numpy().astype(np.float32), np.asarray(alphas)[ba.cpu().numpy()], bc.cpu().numpy()


def ridge_cv(Xtr, Ytr, groups, Xte, alphas, n_folds, device, scale_cols=None, gammas=(1.0,)):
    """Two-group ridge: the columns in `scale_cols` (the tested band) are multiplied by gamma from
    `gammas` (penalty alpha/gamma^2), the rest by 1; (gamma, alpha) chosen per target by CV.
    gamma = 0 removes the band, so the model nests the model without it.
    Returns test predictions, chosen alpha per target, chosen gamma per target."""
    p = Xtr.shape[1]
    if scale_cols is None:
        gammas = [1.0]
        scales = np.ones((1, p))
    else:
        gammas = [float(g) for g in gammas]
        scales = np.ones((len(gammas), p))
        for j, g in enumerate(gammas):
            scales[j, np.asarray(scale_cols, bool)] = g
    pred, al, ci = ridge_cv_scales(Xtr, Ytr, groups, Xte, alphas, n_folds, device, scales)
    return pred, al, np.asarray(gammas)[ci]


def tested_band(cfg, bands):
    """The model's tested band (prefix FORE/RAND/SURP/SURPR, e.g. FORE_h20, SURP_q), or None."""
    pre = cfg["encoding"].get("tested_bands", ["FORE", "RAND", "SURP", "SURPR"])
    t = [b for b in bands if b.split("_")[0] in pre]
    assert len(t) <= 1, f"at most one tested band per model, got {t}"
    return t[0] if t else None


def banded_scales(cfg, bands, widths):
    """Candidates for the fast banded ridge (encoding.solver: banded_fast).
    Non-tested bands: n_draws band weightings w ~ Dirichlet(c) (c alternating over `concentrations`),
    the first draw equal; band b's columns are multiplied by sqrt(w_b * k) (k = number of non-tested
    bands, so equal weights = 1). The same seed gives the same draws for full / full_noF / full_rand.
    Tested band: multiplier gamma from `gammas`, crossed with every draw, gamma = 0 first, so the
    candidate set of a model with the tested band contains that of the model without it (nesting).
    Returns scales [C, p], gamma per candidate (NaN if no tested band)."""
    bc = cfg["encoding"].get("banded", {})
    n_draws, conc = int(bc.get("n_draws", 16)), bc.get("concentrations", [0.1, 1.0])
    gammas = [float(g) for g in bc.get("gammas", [0, 0.5, 1, 2])]
    t = tested_band(cfg, bands)
    nt = [b for b in bands if b != t]
    k = len(nt)
    rng = np.random.default_rng(int(bc.get("seed", 0)))
    draws = [np.full(k, 1.0 / k)] + [rng.dirichlet(np.full(k, conc[i % len(conc)])) for i in range(1, n_draws)]
    edges = np.cumsum([0] + widths)
    cands, gam = [], []
    for g in (gammas if t else [np.nan]):
        for w in draws:
            m = np.ones(edges[-1])
            for bi, b in enumerate(bands):
                sl = slice(edges[bi], edges[bi + 1])
                m[sl] = g if b == t else np.sqrt(w[nt.index(b)] * k)
            cands.append(m)
            gam.append(g)
    return np.stack(cands), np.asarray(gam)


def fit_model(cfg, data, bands, backend, n_iter):
    """Returns (test predictions, chosen gamma of the tested band per parcel (empty if none), alphas)."""
    enc = cfg["encoding"]
    solver = enc.get("solver", "ridge")
    if solver in ("ridge", "banded_fast"):
        import torch
        Xtr = np.concatenate([data["train"]["X"][b] for b in bands], 1)
        Xte = np.concatenate([data["test"]["X"][b] for b in bands], 1)
        widths = [data["train"]["X"][b].shape[1] for b in bands]
        alphas = np.logspace(*enc["alphas_log10"])
        dev = "cuda" if torch.cuda.is_available() and enc["backend"] == "torch_cuda" else "cpu"
        t = tested_band(cfg, bands)
        if solver == "ridge":
            scale_cols = np.concatenate([np.full(w, b == t) for b, w in zip(bands, widths)]) if t else None
            pred, al, gams = ridge_cv(Xtr, data["train"]["Y"], data["train"]["g"], Xte, alphas, enc["n_folds"], dev,
                                      scale_cols=scale_cols, gammas=enc.get("gammas", [0, 0.25, 0.5, 1, 2, 4]))
        else:
            scales, gam = banded_scales(cfg, bands, widths)
            pred, al, ci = ridge_cv_scales(Xtr, data["train"]["Y"], data["train"]["g"], Xte, alphas, enc["n_folds"],
                                           dev, scales)
            gams = gam[ci]
        return pred, (gams if t else np.zeros((0,))), al
    return fit_model_banded(cfg, data, bands, backend, n_iter)


def fit_model_banded(cfg, data, bands, backend, n_iter):
    enc = cfg["encoding"]
    Xtr = np.concatenate([data["train"]["X"][b] for b in bands], 1)
    Xte = np.concatenate([data["test"]["X"][b] for b in bands], 1)
    widths = [data["train"]["X"][b].shape[1] for b in bands]
    edges = np.cumsum([0] + widths)
    ct = ColumnTransformerNoStack([(b, "passthrough", slice(int(edges[i]), int(edges[i + 1])))
                                   for i, b in enumerate(bands)])
    cv = list(GroupKFold(enc["n_folds"]).split(Xtr, data["train"]["Y"], data["train"]["g"]))
    a0, a1, an = enc["alphas_log10"]
    solver_params = dict(n_iter=n_iter, alphas=np.logspace(a0, a1, an), n_targets_batch=enc["n_targets_batch"],
                         n_alphas_batch=5, n_targets_batch_refit=enc["n_targets_batch"], progress_bar=False)
    model = make_pipeline(ct, BandedRidgeCV(groups="input", solver_params=solver_params, cv=cv, random_state=0))
    model.fit(Xtr, data["train"]["Y"])
    pred = backend.to_numpy(model.predict(Xte)).astype(np.float32)
    br = model[-1]
    return pred, backend.to_numpy(br.deltas_), backend.to_numpy(br.best_alphas_)


def suff_stats(Y, P, eps):
    ue = sorted(set(eps.tolist()))
    n = np.array([np.sum(eps == e) for e in ue])
    s = np.stack([Y[eps == e].sum(0) for e in ue]).astype(np.float64)
    ss = np.stack([(Y[eps == e] ** 2).sum(0) for e in ue]).astype(np.float64)
    sse = np.stack([((Y[eps == e] - P[eps == e]) ** 2).sum(0) for e in ue]).astype(np.float64)
    return np.array(ue), n, s, ss, sse


def r2_and_r(Y, P):
    r2 = 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)
    Yc, Pc = Y - Y.mean(0), P - P.mean(0)
    r = (Yc * Pc).sum(0) / np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0) + 1e-12)
    return r2, r


def main():
    p = config_arg()
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--models", nargs="*", default=None)
    p.add_argument("--benchmark", action="store_true", help="time model 'full' with n_iter=2 on the first subject")
    p.add_argument("--tag", default="", help="robustness variant; 'within_shot' drops rows whose surprise clip contains a cut")
    args = p.parse_args()
    cfg = load_config(args.config, args.exp)
    backend = set_backend(cfg["encoding"]["backend"], on_error="warn")
    log(cfg, "fit", f"solver: {cfg['encoding'].get('solver', 'ridge')}; himalaya backend (banded only): {backend.name}")
    subjects = args.subjects or cfg["subjects"]
    models = args.models or list(cfg["models"])
    for sub in subjects:
        t0 = time.time()
        data = build_subject(cfg, sub, within_shot=args.tag == "within_shot")
        log(cfg, "fit", f"{sub}: train rows {len(data['train']['Y'])}, test rows {len(data['test']['Y'])}, "
                        f"train eps {len(set(data['train']['ep']))}, test eps {len(set(data['test']['ep']))} "
                        f"(built in {time.time() - t0:.0f}s)")
        outdir = os.path.join(cfg["dirs"]["results"], args.tag, sub) if args.tag else os.path.join(cfg["dirs"]["results"], sub)
        os.makedirs(outdir, exist_ok=True)
        if args.benchmark:
            bench = max(cfg["models"], key=lambda m: len(cfg["models"][m]))   # largest model
            n_fits = len(cfg["models"]) * len(cfg["subjects"])
            t1 = time.time()
            if cfg["encoding"].get("solver", "ridge") in ("ridge", "banded_fast"):
                fit_model(cfg, data, cfg["models"][bench], backend, 1)
                per_fit = time.time() - t1
                log(cfg, "fit", f"BENCHMARK ({cfg['encoding'].get('solver', 'ridge')}): {per_fit:.0f}s for '{bench}'; all {n_fits} fits <= "
                                f"{per_fit * n_fits / 3600:.1f} h (smaller models are faster)")
            else:
                fit_model(cfg, data, cfg["models"]["full"], backend, n_iter=2)
                per_iter = (time.time() - t1) / 2
                log(cfg, "fit", f"BENCHMARK (banded): ~{per_iter:.1f}s per random-search iteration; "
                                f"{n_fits} fits at n_iter={cfg['encoding']['n_iter']}: "
                                f"{per_iter * cfg['encoding']['n_iter'] * n_fits / 3600:.1f} h")
            return
        Yte = data["test"]["Y"]
        if cfg["encoding"]["save_predictions"]:
            save_npz_atomic(os.path.join(outdir, "_test.npz"), Y=Yte.astype(np.float16), ep=data["test"]["ep"], k=data["test"]["k"])
        for mi, m in enumerate(models):
            nxt = set(cfg["models"][models[mi + 1]]) if mi + 1 < len(models) else set()
            out = os.path.join(outdir, f"{m}.npz")
            if os.path.exists(out):
                continue
            t1 = time.time()
            pred, deltas, alphas = fit_model(cfg, data, cfg["models"][m], backend, cfg["encoding"]["n_iter"])
            ue, n, s, ss, sse = suff_stats(Yte, pred, data["test"]["ep"])
            r2, r = r2_and_r(Yte, pred)
            extra = {"pred": pred.astype(np.float16)} if cfg["encoding"]["save_predictions"] else {}
            save_npz_atomic(out, test_eps=ue, n=n, sum_y=s, sumsq_y=ss, sse=sse, r2=r2, r=r, deltas=deltas,
                            best_alphas=alphas, bands=np.array(cfg["models"][m]), **extra)
            gtxt = ""
            if cfg["encoding"].get("solver", "ridge") in ("ridge", "banded_fast") and deltas.size:
                gtxt = f", tested-band gamma>0 in {100 * np.mean(deltas > 0):.0f}% of parcels (median {np.median(deltas):g})"
            log(cfg, "fit", f"{sub} {m}: median r={np.median(r):.3f}, top-decile r={np.quantile(r, 0.9):.3f}, "
                            f"mean R2={r2.mean():.4f}{gtxt} ({time.time() - t1:.0f}s)")
            for part in ("train", "test"):   # keep only the bands the next model needs
                for b in [b for b in data[part]["X"] if b not in nxt]:
                    del data[part]["X"][b]


if __name__ == "__main__":
    main()
