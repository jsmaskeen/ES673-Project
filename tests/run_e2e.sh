#!/usr/bin/env bash
# End-to-end test on synthetic data with a tiny random V-JEPA 2 (CPU, ~5 min).
# Usage (from the repo root):  bash tests/run_e2e.sh /tmp/pos_e2e
set -euo pipefail
OUT="${1:-/tmp/pos_e2e}"
rm -rf "$OUT" && mkdir -p "$OUT"
python - "$OUT" <<'PY'
import sys, yaml
c = yaml.safe_load(open("config.yaml")); o = sys.argv[1]
c["paths"] = {"data_root": f"{o}/fake_data", "work": f"{o}/work"}
c["train_seasons"] = [4, 5]
c["video"]["size"] = 32
c["model"].update({"tiny_random": True, "dtype": "float32"})
c["extract"]["batch_trs"] = 8
c["encoding"].update({"pca_components": 8, "n_iter": 5, "backend": "numpy"})
c["stats"]["n_boot"] = 1000
yaml.safe_dump(c, open(f"{o}/config_e2e.yaml", "w"), sort_keys=False)
PY
C="$OUT/config_e2e.yaml"
python tests/test_units.py
python tests/test_timing.py
python tests/make_fake_dataset.py "$OUT/fake_data" --nilearn-dir "$OUT/nilearn"
python src/01_index_data.py --config "$C"
python src/02_reencode.py --config "$C" --jobs 2
python src/03_check_vjepa.py --config "$C" --n-clips 16
python src/04_extract_vjepa.py --config "$C"
python src/05_go_nogo.py --config "$C"
python src/07_nuisance.py --config "$C" --jobs 2
python src/08_prepare_features.py --config "$C"
python tests/plant_signal.py --config "$C"
python src/09_fit_encoding.py --config "$C"
python src/10_atlas.py --config "$C"
python src/11_stats.py --config "$C"
python src/12_figures.py --config "$C"
python src/09_fit_encoding.py --config "$C" --tag within_shot --models surp surp_noS surp_rand
python src/11_stats.py --config "$C" --tag within_shot
python - "$OUT" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(f"{sys.argv[1]}/work/results/stats/group_summary.csv")))
ge = {(r["contrast"], r["network"]) for r in rows if r["group_effect"] == "True"}
assert ("H2_surp_vs_rand", "SomMot") in ge and ("surp_vs_none", "SomMot") in ge, "planted surprise effect not recovered"
assert ("H1_fore_vs_rand", "Vis") in ge and ("fore_vs_none", "Vis") in ge, "planted forecast effect not recovered"
assert not any(c.startswith("H2") and n in ("Default", "Limbic", "Cont") for c, n in ge), "spurious H2 effect"
print("E2E PASSED: planted forecast (Vis) and surprise (SomMot) effects recovered")
PY

# ---------------- extra experiments
python tests/test_extra.py
python src/04b_extract_extra.py --config "$C"
# banded ridge on the same planted data must recover the same effects
python src/09_fit_encoding.py --config "$C" --exp banded
python src/11_stats.py --config "$C" --exp banded
python src/13_extra_figures.py --config "$C" --exp banded
# spatial: runs end to end (no planted spatial signal)
python src/08b_prepare_extra.py --config "$C" --exp spatial
python src/09_fit_encoding.py --config "$C" --exp spatial
python src/11_stats.py --config "$C" --exp spatial
python src/12_figures.py --config "$C" --exp spatial
python src/13_extra_figures.py --config "$C" --exp spatial
# horizon: plant a 1 s forecast signal (Vis) and a 3 s surprise signal (SomMot); they must be recovered at those horizons
python src/08b_prepare_extra.py --config "$C" --exp horizon
python tests/plant_signal.py --config "$C" --exp horizon --plant FORE_h10:0:100,SURP_h30:100:200,CUR:200:300
python src/09_fit_encoding.py --config "$C" --exp horizon
python src/11_stats.py --config "$C" --exp horizon
python src/13_extra_figures.py --config "$C" --exp horizon
python - "$OUT" <<'PY'
import csv, sys, os
w = f"{sys.argv[1]}/work"
ge = lambda p: {(r["contrast"], r["network"]) for r in csv.DictReader(open(p)) if r["group_effect"] == "True"}
b = ge(f"{w}/results_banded/stats/group_summary.csv")
assert ("H1_fore_vs_rand", "Vis") in b and ("H2_surp_vs_rand", "SomMot") in b, f"banded ridge lost the planted effects: {b}"
h = ge(f"{w}/results_horizon/stats/group_summary.csv")
assert ("H1_fore_vs_rand_h10", "Vis") in h and ("fore_vs_none_h10", "Vis") in h, f"horizon 1 s forecast not recovered: {sorted(h)}"
assert ("H2_surp_vs_rand_h30", "SomMot") in h and ("surp_vs_none_h30", "SomMot") in h, f"horizon 3 s surprise not recovered: {sorted(h)}"
for f in ["figs_horizon/horizon_H1.png", "figs_horizon/horizon_H2.png", "figs_spatial/compare_main.png", "figs_banded/compare_main.png"]:
    assert os.path.exists(f"{w}/{f}"), f
print("E2E EXTRA PASSED: banded ridge recovers planted effects; horizon experiment recovers them at the planted horizons")
PY
