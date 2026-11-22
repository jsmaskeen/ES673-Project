# Prediction or Surprise?

Pipeline: Algonauts 2025 / CNeuroMod Friends fMRI + frozen V-JEPA 2 ViT-L.
The question is whether cortex tracks the model's **forecast** (H1) or its **prediction error** (H2) beyond controls.


---

## 0. What runs where

| Step | Script | Machine | Rough time |
|---|---|---|---|
| Setup | `scripts/setup_env.sh` | once | 15 min |
| Download | `scripts/download_data.sh 1/2` | network | ~10 GB + ~15 GB |
| 01 index | `src/01_index_data.py` | CPU | 1 min |
| 02 re-encode | `src/02_reencode.py` | CPU | ~1 min per run file per job |
| 03 model checks + benchmark | `src/03_check_vjepa.py` | **GPU** | ~10 min |
| 04 V-JEPA features | `src/04_extract_vjepa.py` | **GPU** | from step-03 benchmark |
| 05 Gate 1 (go/no-go) | `src/05_go_nogo.py` | CPU | seconds |
| 06 fallback forecaster (only if NO-GO) | `src/06_fallback_forecaster.py` | CPU | ~10 min |
| 07 nuisance regressors | `src/07_nuisance.py` | CPU | ~1 min per run file per job |
| 08 PCA / bands | `src/08_prepare_features.py` | CPU | minutes |
| 09 encoding models | `src/09_fit_encoding.py` | **GPU** (or CPU) | from `--benchmark` |
| 10 atlas → Yeo-7 | `src/10_atlas.py` | CPU + internet once | 1 min |
| 11 statistics | `src/11_stats.py` | CPU | minutes |
| 12 figures | `src/12_figures.py` | CPU | 1–2 min |

- **Arguments:** every script takes `--config config.yaml` and is run from the repo root, e.g. `python src/04_extract_vjepa.py`.
- **Resuming:** every step that writes per-episode files skips finished episodes, so an interrupted run resumes when you rerun the same command.
- **Logs:** go to `~/pos/work/logs/`.

---

## 1. Setup (tonight)

```bash
# Linux or WSL2 Ubuntu (datalad does not run on Windows). Keep data under ~/, not /mnt/c.
mkdir -p ~/pos && cd ~/pos && unzip <path>/pos-pipeline.zip   # creates ~/pos/pos-pipeline
cd ~/pos/pos-pipeline
bash scripts/setup_env.sh          # ends with: OK torch 2.5.1+cu121 | transformers 4.57.1 | ... | GPU NVIDIA GeForce RTX 4070
source ~/pos/venv/bin/activate     # do this in every new terminal
bash tests/run_e2e.sh /tmp/pos_e2e # optional, 4 min, must end with "E2E PASSED"
```

Data and outputs go wherever `paths.data_root` / `paths.work` in config.yaml point (set to the portable SSD). Put the Hugging Face cache there too, because the model checkpoint (~1.3 GB) otherwise lands on the home disk: add `export HF_HOME=/media/nishuz/PortableSSD/NeuroAI/hf_cache` to your shell rc. On exFAT/NTFS drives datalad stores files unlocked (no symlinks); the pipeline detects not-yet-downloaded pointer files itself.

## 2. Download (start tonight; leave it running)

```bash
bash scripts/download_data.sh 1    # fMRI (4 subjects), atlases, transcripts S4-S6, Season 6 videos (~10 GB)
bash scripts/download_data.sh 2    # Season 4 + 5 videos (~15 GB)  - can run right after phase 1
```

**Why Seasons 4–5 for training:** the per-run video files of S1–S3 are about 23 GB per season, while S4–S7 are about 7.6 GB per season. The sizes were read from the dataset's git-annex keys. S4+S5 give about 96 runs (~19 h) of training data for a third of the download. Add S3 (`download_data.sh 3`, and `train_seasons: [3, 4, 5]`) only if time is left.

**Verified dataset facts:**
- Files are named `friends_s06e01a.mkv` / `.tsv` (not `friends_s-06...` as the README says).
- h5 keys are `ses-NNN_task-s06e01a`.
- The atlas is a NIfTI file in MNI152NLin2009cAsym space.
- fMRI starts exactly at movie onset (official tutorial).
- An fMRI run can be a few TRs longer than its movie.

## 3. Steps in order

### Person A (video / GPU)

```bash
python src/01_index_data.py                 # re-run after every download phase
python src/02_reencode.py --seasons 6 --jobs 4
python src/03_check_vjepa.py --episode s06e01a
```

**GATE 0: `03_check_vjepa.py` must pass A–D.**
- **A:** model config.
- **B:** the encoder path equals the model's own forward pass.
- **C:** the built-in forward pass leaks the future and ours does not.
- **D:** forecast tokens match their own location.

If D fails, stop and send `work/logs/check_vjepa.json`.

**Benchmark:** read `E_benchmark.est_gpu_hours_all_indexed`. It is the total GPU time for every indexed season. 

```bash
python src/04_extract_vjepa.py --seasons 6 --max-minutes 110     # repeat in each GPU session until done
python src/05_go_nogo.py
python src/02_reencode.py --seasons 4 5 --jobs 4                   # CPU, while the GPU works
python src/04_extract_vjepa.py --seasons 4 5 --max-minutes 110
```

**GATE 1 (`results/go_nogo.json`):**
- **GO:** the forecast beats the random-weight predictor across S6 runs (Wilcoxon p < 0.01, d_z > 0.5).
- **NO-GO:** set `encoding.forecast_source: ridge` in config.yaml and run `python src/06_fallback_forecaster.py` after the training seasons are extracted.

The forecast-vs-copy-forward comparison is reported but does not gate. The encoding models already contain the current features, which carry the copy-forward information. Say in the talk whether the forecast beats persistence.

### Person B (fMRI / CPU)

```bash
python src/07_nuisance.py --seasons 6 --jobs 4    # after S6 is re-encoded; later: --seasons 4 5
python src/10_atlas.py                            # once; downloads the Schaefer atlas from GitHub
```

Once S4, S5 and S6 features and nuisance files exist, run the following:

```bash
python src/08_prepare_features.py
python src/09_fit_encoding.py --benchmark          # prints estimated hours for all fits
python src/09_fit_encoding.py --subjects sub-01 --models base
python src/11_stats.py                             # writes model_fit.csv even with only 'base'
```

**GATE 2 (alignment sanity):** in `work/results/stats/model_fit.csv`, the `base` model must reach `meanr_Vis ≥ 0.15`. If it doesn't, frames and fMRI are misaligned. Check `logs/index_data.log` (excluded runs, fps, start_time) and `logs/reencode.log`, then rerun from step 02 on the affected runs.

If the benchmark estimate is too long, lower `encoding.n_iter` to 15–20, or `pca_components` to 64. The latter means rerunning step 08.

```bash
python src/09_fit_encoding.py                      # all 7 models x 4 subjects (resumable)
python src/11_stats.py
python src/12_figures.py
```

Results are in `work/results/stats/`:
- `group_summary.csv` holds the headline.
- `network_summary.csv` holds per-subject network effects with CIs, p and q.
- `<sub>_parcels.npz` holds parcel maps with ΔR², p and q.

Figures are in `work/figs/`.

---

## 4. What each number means (for the talk and report)

| Contrast | Models | Meaning |
|---|---|---|
| `fore_vs_none` | full − full_noF | Unique ΔR² of the forecast beyond nuisance, current, 6 s past and the actual future (oracle) |
| `rand_vs_none` | full_rand − full_noF | The same for an untrained predictor (capacity control) |
| **`H1_fore_vs_rand`** | full − full_rand | Forecast beats the random predictor |
| `surp_vs_none` | surp − surp_noS | Unique ΔR² of surprise beyond nuisance, current, past and **latent change** (copy-forward error) |
| `surpR_vs_none` | surp_rand − surp_noS | Same for the random predictor's error |
| **`H2_surp_vs_rand`** | surp − surp_rand | Trained surprise beats untrained surprise |

**Decision rule (pre-specify this in the talk):**
- H1 is supported in a Yeo-7 network if both `fore_vs_none` and `H1_fore_vs_rand` have network q < 0.05 with ΔR² > 0 in at least 3 of the 4 subjects.
- H2 is supported under the same rule, using `surp_vs_none` and `H2_surp_vs_rand`.

`11_stats.py` prints the supported networks.

**Statistics:**
- **Unit of inference:** the test episode. p-values come from a one-sided sign-flip test on per-episode ΔR² terms. This is exact under a symmetric null, unlike a bootstrap over TRs.
- **Confidence intervals:** bootstrap over episodes.
- **FDR:** BH across 1000 parcels, and separately across 7 networks.
- **Why small contrasts can be significant:** a model that shrinks harder predicts noise parcels slightly better. That is why each hypothesis needs its random-predictor contrast too. In testing, these artefacts were around 0.001 ΔR² and cancelled in the paired contrasts.

## 5. Methods facts to state in the report

- **Timing:**
  - fMRI sample k is acquired at t_k = 1.49·k s.
  - Video was re-encoded at 4 fps, the V-JEPA 2 pretraining rate.
  - Frame f shows the latest native frame at or before f/4 s. This uses ffmpeg `fps=4:start_time=0:round=up`; the default rounding would leak up to 0.125 s of future, as caught by `tests/test_timing.py`.
  - A feature at TR k uses only frames with f < floor(4·t_k). The oracle is the deliberate exception.
- **Forecast:** V-JEPA 2 ViT-L (`facebook/vjepa2-vitl-fpc64-256`, frozen).
  - The encoder runs on the past 2 s (8 frames, 1024 tokens).
  - The predictor is called directly with context indices 0–1023 and target indices 1024–2047, i.e. the next 2 s.
  - `VJEPA2Model.forward` is never used for forecasting. It encodes all frames before masking, so the future leaks into the context; this is demonstrated in tests and in step 03C.
- **Targets and errors:**
  - The target is the non-affine layer norm of the encoder output on the full 4 s clip, future-half tokens only. This is the V-JEPA 2 training target (`app/vjepa/train.py`).
  - Errors are mean |forecast − target| (the training loss) and 1 − token-mean cosine.
- **Surprise timing:** the error of the window [t_k, t_k+2 s) is assigned to TR k+2. Its last frame is at or before t_k + 1.75 s < t_{k+2} = t_k + 2.98 s, so it is causal.
- **Controls:**
  - A predictor re-initialised with the model's own init (seed 1234), with the encoder untouched.
  - The 6 s integration window.
  - The oracle future.
  - Latent change: copy-forward error.
  - Nuisance regressors: cuts (ffmpeg scene score > 0.3), luminance, frame difference, audio RMS, words and speech from the previous transcript chunk, and a cut-in-surprise-clip flag.
- **Framing:** 16:9 frames were letterboxed into 256×256, so the full field of view is kept. The HF processor's centre crop (292 px resize, 256 crop) drops about half the width; `video.mode: crop` is available as an ablation.
- **Encoding model:**
  - Z-score and PCA to 128 components per band, fitted on the training seasons only.
  - FIR delays of 1–6 TRs, built within each run.
  - Ridge regression without intercept, one penalty per parcel chosen from 21 values (10^-2 to 10^8) by GroupKFold(5) over runs; closed form via the eigendecomposition of X'X (`ridge_cv` in step 09, checked against scikit-learn in `tests/test_units.py`). Banded ridge (himalaya, one penalty per band) is available as `encoding.solver: banded` but was ~500x slower here.
  - The tested band of each model (FORE, RAND, SURP or SURPR) gets its own weight gamma in {0, 0.25, 0.5, 1, 2, 4} per parcel, chosen by the same CV as alpha (penalty on that band = alpha/gamma^2). Reason: with one shared penalty, adding a large, partly redundant band (FORE is 768 delayed columns) can *lower* R2 even if it carries signal, which biases the FORE and SURP contrasts toward null. gamma = 0 removes the band exactly, so `full` nests `full_noF` and `surp` nests `surp_noS` (tested against brute-force scikit-learn in `tests/test_units.py`, test 9). The log line of every tested model reports the share of parcels with gamma > 0; the per-parcel gammas are saved as `deltas` in the model's npz.
  - Train on S4–S5 and test on S6. fMRI is z-scored per run, and every model is scored on identical rows.

## 5b. Extra experiments (horizon curves, banded ridge, spatial)

Each experiment is an overlay in `experiments/<name>.yaml`, selected with `--exp <name>`. It writes
only its own folders (`features/reduced_<exp>`, `results_<exp>`, `figs_<exp>`); the main results are
never touched. The main analysis (steps 08-12) must have been run first.

| Experiment | Status | What changes | GPU |
|---|---|---|---|
| `banded` | planned in the proposal | every band gets its own penalty: 16 Dirichlet band weightings x tested-band gamma {0, .5, 1, 2}, chosen per parcel by CV (fast closed-form version of himalaya's random search; `--exp banded` reuses the main features) | ~1.5-2 h of fits |
| `horizon` | planned in the proposal | forecast / random predictor / oracle / surprise of the single 0.5-s slot ending 0.5, 1, 2, 3, 4 s ahead; the same 3+3 model design per horizon | new extraction (04b) |
| `spatial` | exploratory (chosen after the main result) | all V-JEPA bands pooled per 2x2 frame quadrant instead of over the whole frame (CUR_q keeps spatial layout; 256 PCA components) | same extraction (04b) |

```bash
python3 src/04b_extract_extra.py                       # GPU, resumable: S6 first, then S1-S5
python3 src/09_fit_encoding.py --exp banded            # can run while 04b runs (needs ~3 GB GPU memory)
python3 src/11_stats.py --exp banded && python3 src/13_extra_figures.py --exp banded
python3 src/08b_prepare_extra.py --exp horizon         # after 04b has finished
python3 src/09_fit_encoding.py --exp horizon && python3 src/11_stats.py --exp horizon && python3 src/13_extra_figures.py --exp horizon
python3 src/08b_prepare_extra.py --exp spatial
python3 src/09_fit_encoding.py --exp spatial && python3 src/11_stats.py --exp spatial
python3 src/12_figures.py --exp spatial && python3 src/13_extra_figures.py --exp spatial
```
Tests: `tests/test_extra.py` (no leakage in the new forecasts, pooling, per-slot surprise shifts,
banded candidates nest the reduced model, solver == scikit-learn) and the end of `tests/run_e2e.sh`
(banded ridge recovers the planted effects; the horizon experiment recovers a planted 1 s forecast
and 3 s surprise at exactly those horizons).
