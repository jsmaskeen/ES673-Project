"""Step 05 - Gate 1: does the V-JEPA 2 forecast actually carry information about the next 2 s?

Uses the test-season feature files already written by step 04 (no extra GPU).
Unit = episode (mean over its valid TRs), so neighbouring TRs are not treated
as independent. Similarities are token-wise cosine with layer_norm(target).

Decision (exact):
  GO      if forecast > random-weight predictor: one-sided Wilcoxon p < 0.01 AND
          Cohen's d_z > 0.5 across episodes (requires >= 8 episodes).
  NO-GO   otherwise -> set `encoding.forecast_source: ridge` in config.yaml and
          run src/06_fallback_forecaster.py.
Forecast vs copy-forward is reported and must be described in the report, but
it does not gate: the encoding models already contain the current features
(= the copy-forward information), so H1 stays well-defined either way.
"""
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

from common import config_arg, ep_season, load_config, log


def main():
    args = config_arg().parse_args()
    cfg = load_config(args.config)
    files = sorted(glob.glob(os.path.join(cfg["dirs"]["vjepa"], "*.npz")))
    files = [f for f in files if ep_season(os.path.basename(f)[:-4]) == cfg["test_season"]]
    assert len(files) >= 8, f"need >= 8 test-season episodes extracted, have {len(files)}"
    rows = []
    for f in files:
        d = np.load(f)
        v = d["valid"]
        rows.append({k: float(np.mean(d[k][v])) for k in
                     ["fore_cos", "rand_cos", "copy_cos", "fore_l1", "rand_l1", "copy_l1"]} | {"ep": os.path.basename(f)[:-4]})
    fc, rc, cc = (np.array([r[k] for r in rows]) for k in ["fore_cos", "rand_cos", "copy_cos"])

    def test(a, b):
        d = a - b
        return {"mean_diff": float(d.mean()), "dz": float(d.mean() / d.std(ddof=1)),
                "p_one_sided": float(wilcoxon(a, b, alternative="greater").pvalue),
                "n_episodes_a_better": int((d > 0).sum()), "n_episodes": int(len(d))}
    vs_rand, vs_copy = test(fc, rc), test(fc, cc)
    go = vs_rand["p_one_sided"] < 0.01 and vs_rand["dz"] > 0.5
    res = {"decision": "GO" if go else "NO-GO", "forecast_vs_random": vs_rand, "forecast_vs_copy": vs_copy,
           "beats_copy_forward": bool(vs_copy["p_one_sided"] < 0.01 and vs_copy["dz"] > 0.5),
           "means": {"fore_cos": float(fc.mean()), "rand_cos": float(rc.mean()), "copy_cos": float(cc.mean())},
           "per_episode": rows}
    with open(os.path.join(cfg["dirs"]["results"], "go_nogo.json"), "w") as f:
        json.dump(res, f, indent=1)
    log(cfg, "go_nogo", f"decision={res['decision']}  forecast-vs-random {vs_rand}  forecast-vs-copy {vs_copy}")
    if not go:
        log(cfg, "go_nogo", "NO-GO: set encoding.forecast_source: ridge in config.yaml, then run 06_fallback_forecaster.py")


if __name__ == "__main__":
    main()
