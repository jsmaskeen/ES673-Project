"""Build a small fake Algonauts-2025-shaped dataset for end-to-end testing.

Same file names, h5 keys, atlas format and transcript columns as the real
dataset (verified against the datalad index of
github.com/courtois-neuromod/algonauts_2025.competitors and friends.stimuli):
  fmri/sub-0X/func/sub-0X_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5
      keys 'ses-0NN_task-sSSeEEx' -> float32 [N_TR, 1000]
  fmri/sub-0X/atlas/sub-0X_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz
  stimuli/movies/friends/s<season>/friends_sSSeEEx.mkv   (29.97 fps, 16:9, with audio)
  stimuli/transcripts/friends/s<season>/friends_sSSeEEx.tsv  (one row per 1.49 s)
fMRI here is noise; tests/plant_signal.py later replaces it with a planted signal.

    python tests/make_fake_dataset.py OUT_DIR --nilearn-dir DIR
"""
import argparse
import os
import subprocess

import h5py
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image

SUBS = ["sub-01", "sub-02", "sub-03", "sub-05"]
EPISODES = {4: ["s04e02a", "s04e02b", "s04e03a", "s04e03b"],
            5: ["s05e02a", "s05e02b", "s05e03a", "s05e03b"],
            6: ["s06e01a", "s06e01b", "s06e02a", "s06e02b", "s06e03a", "s06e03b", "s06e04a", "s06e04b"]}
DUR = 120.0
TR = 1.49
SRC = ["testsrc2=s=320x180:r=30000/1001", "mandelbrot=s=320x180:r=30000/1001",
       "cellauto=s=320x180:r=30000/1001:rule=110", "life=s=320x180:r=30000/1001:mold=10:ratio=0.2",
       "smptehdbars=s=320x180:r=30000/1001", "rgbtestsrc=s=320x180:r=30000/1001"]
WORDS = "so what are you doing here we were on a break could I be any more".split()


def make_video(path, seed):
    rng = np.random.default_rng(seed)
    durs, t = [], 0.0
    while t < DUR:
        d = min(float(rng.uniform(6, 20)), DUR - t)
        durs.append(d)
        t += d
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error"]
    for i, d in enumerate(durs):
        cmd += ["-f", "lavfi", "-t", f"{d:.3f}", "-i", SRC[int(rng.integers(len(SRC)))]]
    cmd += ["-f", "lavfi", "-t", f"{DUR}", "-i", f"sine=frequency={int(rng.integers(200, 800))}:beep_factor=4"]
    fc = "".join(f"[{i}:v]" for i in range(len(durs))) + f"concat=n={len(durs)}:v=1:a=0,format=yuv420p[v]"
    cmd += ["-filter_complex", fc, "-map", "[v]", "-map", f"{len(durs)}:a", "-c:v", "libx264", "-preset", "veryfast",
            "-c:a", "pcm_s16le", path]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--nilearn-dir", required=True)
    a = ap.parse_args()
    root = a.out
    rng = np.random.default_rng(0)
    n_tr = int(DUR // TR) + 2          # fMRI runs slightly longer than the movie (as in the real data)
    for s, eps in EPISODES.items():
        os.makedirs(f"{root}/stimuli/movies/friends/s{s}", exist_ok=True)
        os.makedirs(f"{root}/stimuli/transcripts/friends/s{s}", exist_ok=True)
        for i, ep in enumerate(eps):
            make_video(f"{root}/stimuli/movies/friends/s{s}/friends_{ep}.mkv", seed=100 * s + i)
            rows = []
            for k in range(n_tr - 2):
                if rng.random() < 0.6:
                    w = list(rng.choice(WORDS, size=int(rng.integers(1, 5))))
                    rows.append({"text_per_tr": " ".join(w), "words_per_tr": str(w),
                                 "onsets_per_tr": str([round(k * TR + .1 * j, 2) for j in range(len(w))]),
                                 "durations_per_tr": str([0.1] * len(w))})
                else:
                    rows.append({"text_per_tr": None, "words_per_tr": None, "onsets_per_tr": None, "durations_per_tr": None})
            pd.DataFrame(rows).to_csv(f"{root}/stimuli/transcripts/friends/s{s}/friends_{ep}.tsv", sep="\t", index=False)
    sch = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7, resolution_mm=2, data_dir=a.nilearn_dir)
    aff = np.array([[2, 0, 0, -96], [0, 2, 0, -132], [0, 0, 2, -78], [0, 0, 0, 1]], float)
    atl = image.resample_img(sch.maps, target_affine=aff, target_shape=(97, 115, 97), interpolation="nearest",
                             force_resample=True, copy_header=True)
    atl = nib.Nifti1Image(np.asarray(atl.dataobj).astype(np.int16), aff)
    for sub in SUBS:
        os.makedirs(f"{root}/fmri/{sub}/func", exist_ok=True)
        os.makedirs(f"{root}/fmri/{sub}/atlas", exist_ok=True)
        nib.save(atl, f"{root}/fmri/{sub}/atlas/{sub}_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz")
        fn = f"{root}/fmri/{sub}/func/{sub}_task-friends_space-MNI152NLin2009cAsym_atlas-Schaefer18_parcel-1000Par7Net_desc-s123456_bold.h5"
        with h5py.File(fn, "w") as h:
            ses = 1
            for s, eps in EPISODES.items():
                for ep in eps:
                    h.create_dataset(f"ses-{ses:03d}_task-{ep}", data=rng.standard_normal((n_tr, 1000)).astype(np.float32))
                    ses += 1
    print("fake dataset at", root)


if __name__ == "__main__":
    main()
