"""Step 12 - figures for the talk and report (CPU). Writes PNGs to <work>/figs/.

  go_nogo.png              per-episode forecast skill: forecast vs copy-forward vs random predictor
  model_fit.png            median test r per model and subject (sanity)
  network_H1.png / _H2.png network-level unique variance per contrast (bars = subject mean, dots = subjects)
  map_<contrast>.png       cortical surface map of the across-subject mean contrast (fsaverage5)
  timecourse_<ep>.png      surprise / latent change / cuts, and BOLD vs prediction in the best H2 parcel
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from common import atlas_path, config_arg, load_config, log

NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
H1 = ["fore_vs_none", "rand_vs_none", "H1_fore_vs_rand"]
H2 = ["surp_vs_none", "surpR_vs_none", "H2_surp_vs_rand"]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fig_go_nogo(cfg):
    path = os.path.join(cfg["dirs"]["results"], "go_nogo.json")
    if not os.path.exists(path):
        return
    g = json.load(open(path))
    rows = g["per_episode"]
    fig, ax = plt.subplots(figsize=(5, 4))
    for i, k in enumerate(["copy_cos", "rand_cos", "fore_cos"]):
        ax.scatter(np.full(len(rows), i) + np.random.default_rng(i).uniform(-.08, .08, len(rows)),
                   [r[k] for r in rows], s=12, alpha=.7)
    for r in rows:
        ax.plot([0, 1, 2], [r["copy_cos"], r["rand_cos"], r["fore_cos"]], color="0.8", lw=.5, zorder=0)
    ax.set_xticks([0, 1, 2], ["copy-forward", "random predictor", "V-JEPA 2 forecast"])
    ax.set_ylabel("cosine with actual next 2 s (token mean)")
    ax.set_title(f"Gate 1: {g['decision']}")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["dirs"]["figs"], "go_nogo.png"), dpi=200)
    plt.close(fig)


def fig_model_fit(cfg):
    path = os.path.join(cfg["dirs"]["stats"], "model_fit.csv")
    if not os.path.exists(path):
        return
    rows = read_csv(path)
    models = sorted({r["model"] for r in rows})
    subs = sorted({r["subject"] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for j, s in enumerate(subs):
        vals = [next((float(r["median_r"]) for r in rows if r["model"] == m and r["subject"] == s), np.nan) for m in models]
        ax.plot(range(len(models)), vals, "o-", label=s)
    ax.set_xticks(range(len(models)), models, rotation=30)
    ax.set_ylabel("median test r (1000 parcels)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["dirs"]["figs"], "model_fit.png"), dpi=200)
    plt.close(fig)


def fig_networks(cfg):
    path = os.path.join(cfg["dirs"]["stats"], "network_summary.csv")
    if not os.path.exists(path):
        return
    rows = read_csv(path)
    for name, cons in [("H1", H1), ("H2", H2)]:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        w = 0.8 / len(cons)
        for i, c in enumerate(cons):
            for j, n in enumerate(NETWORKS):
                v = [float(r["mean"]) for r in rows if r["contrast"] == c and r["network"] == n]
                sig = [float(r["q"]) < cfg["stats"]["alpha"] for r in rows if r["contrast"] == c and r["network"] == n]
                if not v:
                    continue
                x = j + (i - (len(cons) - 1) / 2) * w
                ax.bar(x, np.mean(v), w * .9, color=f"C{i}", label=c if j == 0 else None)
                ax.scatter(np.full(len(v), x), v, c=["k" if s else "0.6" for s in sig], s=10, zorder=3)
        ax.axhline(0, color="k", lw=.6)
        ax.set_xticks(range(len(NETWORKS)), NETWORKS)
        ax.set_ylabel("ΔR² (test season)")
        ax.set_title(f"{name}: bars = mean of subjects; black dots = subject q < {cfg['stats']['alpha']}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(cfg["dirs"]["figs"], f"network_{name}.png"), dpi=200)
        plt.close(fig)


def fig_maps(cfg):
    from nilearn import datasets, plotting, surface
    img = nib.load(atlas_path(cfg, cfg["subjects"][0]))
    A = np.asarray(img.dataobj).astype(int)
    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    files = [os.path.join(cfg["dirs"]["stats"], f"{s}_parcels.npz") for s in cfg["subjects"]]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        return
    for c in H1 + H2:
        vals = [np.load(f)[c][0] for f in files if c in np.load(f).files]
        if not vals:
            continue
        mean = np.mean(vals, 0)
        vol = nib.Nifti1Image(np.where(A > 0, np.r_[0.0, mean][A], 0.0).astype(np.float32), img.affine)
        vmax = float(np.percentile(np.abs(mean), 99)) or 1e-4
        fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(9, 3.6))
        for ax, hemi in zip(axes, ["left", "right"]):
            tex = surface.vol_to_surf(vol, getattr(fs, f"pial_{hemi}"), interpolation="nearest")
            plotting.plot_surf_stat_map(getattr(fs, f"infl_{hemi}"), tex, hemi=hemi, view="lateral",
                                        bg_map=getattr(fs, f"sulc_{hemi}"), axes=ax, colorbar=hemi == "right",
                                        vmax=vmax, threshold=vmax * 0.05, cmap="cold_hot", figure=fig)
        fig.suptitle(f"{c}: mean ΔR² across {len(vals)} subjects")
        fig.savefig(os.path.join(cfg["dirs"]["figs"], f"map_{c}.png"), dpi=200)
        plt.close(fig)


def fig_timecourse(cfg):
    sub = cfg["subjects"][0]
    tpath = os.path.join(cfg["dirs"]["results"], sub, "_test.npz")
    mpath = os.path.join(cfg["dirs"]["results"], sub, "surp.npz")
    ppath = os.path.join(cfg["dirs"]["stats"], f"{sub}_parcels.npz")
    if not (os.path.exists(tpath) and os.path.exists(mpath) and os.path.exists(ppath)):
        return
    T, M = np.load(tpath), np.load(mpath)
    if "pred" not in M.files:
        return
    ep = sorted(set(T["ep"].tolist()))[0]
    sel = T["ep"] == ep
    best = int(np.argmax(np.load(ppath)["surp_vs_none"][0]))
    d = np.load(os.path.join(cfg["dirs"]["vjepa"], f"{ep}.npz"))
    nu = np.load(os.path.join(cfg["dirs"]["nuisance"], f"{ep}.npz"))
    t = np.arange(len(d["surp_l1"])) * cfg["tr"]
    z = lambda x: (x - np.nanmean(x)) / np.nanstd(x)
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(t, z(d["surp_l1"]), label="surprise (L1)")
    ax[0].plot(t, z(d["chg_l1"]), label="latent change (copy-forward error)", alpha=.7)
    for c in nu["cut_times"]:
        ax[0].axvline(c, color="0.7", lw=.5)
    ax[0].legend(fontsize=7)
    ax[0].set_ylabel("z")
    rows = np.where(sel)[0]
    yy, pp = T["Y"][rows, best].astype(float), M["pred"][rows, best].astype(float)
    tt = T["k"][rows] * cfg["tr"]
    ax[1].plot(tt, yy, color="k", lw=.8, label=f"BOLD parcel {best + 1}")
    ax[1].plot(tt, pp, color="C3", lw=.8, label="'surp' model prediction")
    ax[1].legend(fontsize=7)
    ax[1].set_xlabel("time (s); grey lines = cuts")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["dirs"]["figs"], f"timecourse_{ep}.png"), dpi=200)
    plt.close(fig)


def main():
    args = config_arg().parse_args()
    cfg = load_config(args.config, args.exp)
    for f in [fig_go_nogo, fig_model_fit, fig_networks, fig_maps, fig_timecourse]:
        f(cfg)
        log(cfg, "figures", f"{f.__name__} done")


if __name__ == "__main__":
    main()
