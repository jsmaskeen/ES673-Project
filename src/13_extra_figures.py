"""Step 13 - figures for the extra experiments (CPU). PNGs in <work>/figs_<exp>/.

  --exp horizon : horizon_H1.png / horizon_H2.png  network-level dR2 vs forecast horizon
                  (lines = mean of subjects, dots = subjects, filled marker = group effect)
  --exp spatial | banded : compare_main.png  main analysis vs this experiment, per network and contrast
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import HORIZON_SLOTS, config_arg, load_config, log

NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
SIX = ["fore_vs_none", "rand_vs_none", "H1_fore_vs_rand", "surp_vs_none", "surpR_vs_none", "H2_surp_vs_rand"]


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fig_horizon(cfg):
    rows = read(os.path.join(cfg["dirs"]["stats"], "network_summary.csv"))
    grp = {(r["contrast"], r["network"]): r["group_effect"] == "True"
           for r in read(os.path.join(cfg["dirs"]["stats"], "group_summary.csv"))}
    hs = list(HORIZON_SLOTS)
    xs = [(j + 1) / 2 for j in HORIZON_SLOTS.values()]
    for name, cons in [("H1", ["fore_vs_none", "rand_vs_none", "H1_fore_vs_rand"]),
                       ("H2", ["surp_vs_none", "surpR_vs_none", "H2_surp_vs_rand"])]:
        fig, axes = plt.subplots(1, len(NETWORKS), figsize=(17, 3.2), sharey=True)
        for ax, n in zip(axes, NETWORKS):
            for i, c in enumerate(cons):
                means = []
                for x, h in zip(xs, hs):
                    v = [float(r["mean"]) for r in rows if r["contrast"] == f"{c}_{h}" and r["network"] == n]
                    means.append(np.mean(v) if v else np.nan)
                    ax.scatter(np.full(len(v), x + (i - 1) * 0.06), v, s=6, color=f"C{i}", alpha=.4)
                ax.plot(xs, means, color=f"C{i}", label=c, lw=1.2)
                for x, h, m in zip(xs, hs, means):
                    g = grp.get((f"{c}_{h}", n), False)
                    ax.scatter([x], [m], s=28, color=f"C{i}" if g else "white", edgecolor=f"C{i}", zorder=3)
            ax.axhline(0, color="k", lw=.6)
            ax.set_title(n)
            ax.set_xticks(xs)
            ax.set_xlabel("horizon (s)")
        axes[0].set_ylabel("ΔR² (test season)")
        axes[-1].legend(fontsize=7)
        fig.suptitle(f"{name} by forecast horizon; filled = group effect (q<.05 & ΔR²>0 in >= {cfg['stats']['min_subjects']} subjects)")
        fig.tight_layout()
        fig.savefig(os.path.join(cfg["dirs"]["figs"], f"horizon_{name}.png"), dpi=200)
        plt.close(fig)


def fig_compare(cfg):
    main_stats = os.path.join(cfg["paths"]["work"], "results", "stats", "group_summary.csv")
    if not os.path.exists(main_stats):
        return
    A = {(r["contrast"], r["network"]): r for r in read(main_stats)}
    B = {(r["contrast"], r["network"]): r for r in read(os.path.join(cfg["dirs"]["stats"], "group_summary.csv"))}
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
    for ax, c in zip(axes.ravel(), SIX):
        for i, (lab, D) in enumerate([("main", A), (cfg["experiment"], B)]):
            m = [float(D[(c, n)]["mean_across_subjects"]) if (c, n) in D else np.nan for n in NETWORKS]
            k = [D[(c, n)]["n_sig"] if (c, n) in D else "" for n in NETWORKS]
            x = np.arange(len(NETWORKS)) + (i - 0.5) * 0.38
            ax.bar(x, m, 0.36, color=f"C{i}", label=lab)
            for xi, mi, ki in zip(x, m, k):
                ax.text(xi, mi, str(ki), ha="center", va="bottom" if mi >= 0 else "top", fontsize=7)
        ax.axhline(0, color="k", lw=.6)
        ax.set_title(c)
        ax.set_xticks(range(len(NETWORKS)), NETWORKS, rotation=30, fontsize=8)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("mean ΔR² across subjects; numbers = subjects with q < .05")
    fig.tight_layout()
    fig.savefig(os.path.join(cfg["dirs"]["figs"], "compare_main.png"), dpi=200)
    plt.close(fig)


def main():
    args = config_arg().parse_args()
    cfg = load_config(args.config, args.exp)
    if cfg["experiment"] == "horizon":
        fig_horizon(cfg)
        log(cfg, "figures", "horizon figures done")
    else:
        fig_compare(cfg)
        log(cfg, "figures", "compare figure done")


if __name__ == "__main__":
    main()
