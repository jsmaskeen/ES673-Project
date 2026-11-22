"""Step 11 - statistics (CPU, minutes).

Contrasts (difference in held-out R^2 on the test season, model a - model b):
  fore_vs_none     full      - full_noF     unique variance of the forecast
  rand_vs_none     full_rand - full_noF     same for the random-weight predictor
  H1_fore_vs_rand  full      - full_rand    forecast beats the random predictor
  surp_vs_none     surp      - surp_noS     unique variance of surprise
  surpR_vs_none    surp_rand - surp_noS     same for random-predictor surprise
  H2_surp_vs_rand  surp      - surp_rand    surprise beats random-predictor surprise

Both models are scored on identical rows, so for parcel p
    dR2_p = sum_e c_ep,   c_ep = (SSE_b,e,p - SSE_a,e,p) / SST_p
where e runs over test EPISODES. Inference per subject treats episodes as the
independent units:
  p-values : one-sided sign-flip permutation test on the episode terms c_ep
             (B random +-1 flips; p = (1 + #{null >= observed}) / (B + 1)).
             Exact under a symmetric null; conservative when an extra band only
             adds noise (then c_ep tends to be negative).
             Network level: c_e = mean over the network's parcels of c_ep.
  CIs      : percentile bootstrap over episodes (resampled with replacement).
  FDR      : Benjamini-Hochberg over 1000 parcels, and separately over 7 networks.
Group rule: an effect is reported for a network if q < alpha and dR2 > 0 in
>= min_subjects subjects. H1 (H2) is supported in a network if BOTH its contrasts
(fore_vs_none & H1_fore_vs_rand; surp_vs_none & H2_surp_vs_rand) pass.
"""
import csv
import json
import os

import numpy as np

from common import config_arg, load_config, log

CONTRASTS = {
    "fore_vs_none": ("full", "full_noF"), "rand_vs_none": ("full_rand", "full_noF"),
    "H1_fore_vs_rand": ("full", "full_rand"), "surp_vs_none": ("surp", "surp_noS"),
    "surpR_vs_none": ("surp_rand", "surp_noS"), "H2_surp_vs_rand": ("surp", "surp_rand"),
}
NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


def contrasts_of(cfg):
    """Contrasts and hypotheses: the defaults above, or an experiment's own (common.load_config)."""
    C = {k: tuple(v) for k, v in cfg.get("contrasts", CONTRASTS).items()}
    H = cfg.get("hypotheses", {"H1": ["fore_vs_none", "H1_fore_vs_rand"], "H2": ["surp_vs_none", "H2_surp_vs_rand"]})
    return C, H


def load_sub(cfg, sub, tag=""):
    d = {}
    CON, _ = contrasts_of(cfg)
    for m in {x for pair in CON.values() for x in pair} | {"base"}:
        f = os.path.join(cfg["dirs"]["results"], tag, sub, f"{m}.npz")
        if os.path.exists(f):
            d[m] = dict(np.load(f))
    eps = None
    for m, v in d.items():
        if eps is None:
            eps = v["test_eps"]
        assert (v["test_eps"] == eps).all(), f"{sub}: models scored on different episodes"
    return d


def r2_boot(W, mod):
    """R^2 per parcel for each row of episode-weight matrix W [B, E]."""
    n, s, ss, sse = W @ mod["n"], W @ mod["sum_y"], W @ mod["sumsq_y"], W @ mod["sse"]
    sst = ss - s ** 2 / n[:, None]
    return 1 - sse / sst


def episode_terms(a, b):
    """c[e, p] with sum_e c[e, p] = R2_a - R2_b (all test episodes)."""
    n, s, ss = a["n"].sum(), a["sum_y"].sum(0), a["sumsq_y"].sum(0)
    sst = ss - s ** 2 / n
    return (b["sse"] - a["sse"]) / sst


def signflip_p(c, S):
    """c [E, P], S [B, E] of +-1 -> one-sided p per column."""
    obs = c.sum(0)
    null = S @ c
    return (1 + (null >= obs - 1e-12).sum(0)) / (S.shape[0] + 1)


def main():
    p = config_arg()
    p.add_argument("--tag", default="", help="read results/<tag>/ and write stats/<tag>/ (e.g. within_shot)")
    args = p.parse_args()
    cfg = load_config(args.config, args.exp)
    if args.tag:
        cfg["dirs"]["stats"] = os.path.join(cfg["dirs"]["stats"], args.tag)
        os.makedirs(cfg["dirs"]["stats"], exist_ok=True)
    st = cfg["stats"]
    rng = np.random.default_rng(st["seed"])
    nets = np.array(json.load(open(os.path.join(cfg["dirs"]["atlas"], "parcel_networks.json")))["networks"])
    net_rows, group, fits = [], {}, []
    for sub in cfg["subjects"]:
        d = load_sub(cfg, sub, args.tag)
        if not d:
            log(cfg, "stats", f"{sub}: no results, skipped")
            continue
        E = len(next(iter(d.values()))["test_eps"])
        Wb = rng.multinomial(E, np.full(E, 1.0 / E), size=st["n_boot"]).astype(np.float64)
        S = rng.choice([-1.0, 1.0], size=(st["n_boot"], E))
        W1 = np.ones((1, E))
        for m, v in d.items():
            r2 = r2_boot(W1, v)[0]
            fits.append({"subject": sub, "model": m, "median_r": float(np.median(v["r"])),
                         **{f"meanr_{n}": float(v["r"][nets == n].mean()) for n in NETWORKS},
                         **{f"meanR2_{n}": float(r2[nets == n].mean()) for n in NETWORKS}})
        parcel_out = {}
        for cname, (a, b) in contrasts_of(cfg)[0].items():
            if a not in d or b not in d:
                continue
            c = episode_terms(d[a], d[b])                                  # [E, 1000]
            obs = c.sum(0)
            assert np.allclose(obs, r2_boot(W1, d[a])[0] - r2_boot(W1, d[b])[0], atol=1e-8)
            p = signflip_p(c, S)
            q = bh(p)
            parcel_out[cname] = np.stack([obs, p, q])
            boot = r2_boot(Wb, d[a]) - r2_boot(Wb, d[b])                   # [B, 1000] for CIs
            rows, net_p = [], []
            for n in NETWORKS:
                sel = nets == n
                cn = c[:, sel].mean(1, keepdims=True)
                pn = float(signflip_p(cn, S)[0])
                bn = boot[:, sel].mean(1)
                rows.append({"subject": sub, "contrast": cname, "network": n, "mean": float(obs[sel].mean()),
                             "ci_lo": float(np.percentile(bn, 2.5)), "ci_hi": float(np.percentile(bn, 97.5)),
                             "p": pn, "n_parcels_q05": int((q[sel] < st["alpha"]).sum()), "n_test_episodes": E})
                net_p.append(pn)
            for r, qn in zip(rows, bh(net_p)):
                r["q"] = float(qn)
                net_rows.append(r)
                group.setdefault((cname, r["network"]), []).append((r["mean"], qn < st["alpha"] and r["mean"] > 0))
        np.savez(os.path.join(cfg["dirs"]["stats"], f"{sub}_parcels.npz"), **parcel_out)
        log(cfg, "stats", f"{sub}: {len(parcel_out)} contrasts, {E} test episodes, B={st['n_boot']}")

    def write(path, rows):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    write(os.path.join(cfg["dirs"]["stats"], "network_summary.csv"), net_rows)
    write(os.path.join(cfg["dirs"]["stats"], "model_fit.csv"), fits)
    grows = []
    for (c, n), v in group.items():
        grows.append({"contrast": c, "network": n, "n_subjects": len(v), "n_sig": sum(s for _, s in v),
                      "mean_across_subjects": float(np.mean([m for m, _ in v])),
                      "group_effect": sum(s for _, s in v) >= st["min_subjects"]})
    write(os.path.join(cfg["dirs"]["stats"], "group_summary.csv"), grows)
    ge = {(r["contrast"], r["network"]): r["group_effect"] for r in grows}
    for h, (c1, c2) in contrasts_of(cfg)[1].items():
        sup = [n for n in NETWORKS if ge.get((c1, n)) and ge.get((c2, n))]
        log(cfg, "stats", f"{h} supported in networks: {sup if sup else 'none'}")


if __name__ == "__main__":
    main()
