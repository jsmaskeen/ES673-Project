"""Shared helpers: config, paths, episode naming, fMRI loading, timing rules.

Timing convention (verified in the official Algonauts 2025 tutorial, cell 91:
"the fMRI timeseries onset is ALWAYS SYNCHRONIZED with movie onset"):
    fMRI sample k is acquired at t_k = k * TR seconds after movie onset.
Re-encoded video: frame f (0-based) shows the movie at time f / fps.
Leakage rule used everywhere: a feature attached to TR k may only use
frames f < e_k, where e_k = floor(fps * t_k)  (so every frame time < t_k).
"""
import glob
import json
import math
import os
import re
import sys
import time

import h5py
import numpy as np
import yaml

FRIENDS_KEY_RE = re.compile(r"^ses-(\d+)_task-(s\d{2}e\d{2}[a-z])$")
EP_RE = re.compile(r"^s(\d{2})e(\d{2})([a-z])$")


# ----------------------------------------------------------------------------- config
HORIZON_SLOTS = {"h05": 0, "h10": 1, "h20": 3, "h30": 5, "h40": 7}   # name -> future slot (0.5 s each); slot j ends (j+1)/2 s after t


def _deep_update(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = v
    return a


def horizon_models_contrasts(slots):
    """Models/contrasts for the horizon experiment: for every horizon the same 3+3 design as the main analysis."""
    models, contrasts, hyps = {"base": ["NUIS", "CUR"]}, {}, {}
    for h in slots:
        models[f"full_{h}"] = ["NUIS", "CUR", "LONG", f"ORAC_{h}", f"FORE_{h}"]
        models[f"full_noF_{h}"] = ["NUIS", "CUR", "LONG", f"ORAC_{h}"]
        models[f"full_rand_{h}"] = ["NUIS", "CUR", "LONG", f"ORAC_{h}", f"RAND_{h}"]
        models[f"surp_{h}"] = ["NUIS", "CUR", "LONG", f"CHG_{h}", f"SURP_{h}"]
        models[f"surp_noS_{h}"] = ["NUIS", "CUR", "LONG", f"CHG_{h}"]
        models[f"surp_rand_{h}"] = ["NUIS", "CUR", "LONG", f"CHG_{h}", f"SURPR_{h}"]
        contrasts.update({
            f"fore_vs_none_{h}": [f"full_{h}", f"full_noF_{h}"], f"rand_vs_none_{h}": [f"full_rand_{h}", f"full_noF_{h}"],
            f"H1_fore_vs_rand_{h}": [f"full_{h}", f"full_rand_{h}"], f"surp_vs_none_{h}": [f"surp_{h}", f"surp_noS_{h}"],
            f"surpR_vs_none_{h}": [f"surp_rand_{h}", f"surp_noS_{h}"], f"H2_surp_vs_rand_{h}": [f"surp_{h}", f"surp_rand_{h}"]})
        hyps[f"H1_{h}"] = [f"fore_vs_none_{h}", f"H1_fore_vs_rand_{h}"]
        hyps[f"H2_{h}"] = [f"surp_vs_none_{h}", f"H2_surp_vs_rand_{h}"]
    return models, contrasts, hyps


def load_config(path="config.yaml", exp=None):
    """exp: name of an overlay in experiments/<exp>.yaml (horizon | spatial | banded). An experiment reads
    and writes its own reduced features, results, stats and figures (suffix _<exp>), never the main ones."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if exp:
        ov = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", f"{exp}.yaml")
        assert os.path.exists(ov), f"no experiment overlay {ov}"
        with open(ov) as f:
            _deep_update(cfg, yaml.safe_load(f))
        if cfg.get("generate") == "horizon":
            cfg["models"], cfg["contrasts"], cfg["hypotheses"] = horizon_models_contrasts(list(HORIZON_SLOTS))
    cfg["experiment"] = exp or ""
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = os.path.abspath(os.path.expanduser(v))
    w = cfg["paths"]["work"]
    sfx = f"_{exp}" if exp else ""
    red = os.path.join(w, "features", "reduced") if (not exp or cfg.get("reuse_main_features")) \
        else os.path.join(w, "features", f"reduced{sfx}")
    cfg["dirs"] = {
        "frames": os.path.join(w, "frames4"),
        "audio": os.path.join(w, "audio16k"),
        "vjepa": os.path.join(w, "features", "vjepa"),
        "extra": os.path.join(w, "features", "extra"),
        "nuisance": os.path.join(w, "features", "nuisance"),
        "ridgefc": os.path.join(w, "features", "ridgefc"),
        "reduced": red,
        "reduced_main": os.path.join(w, "features", "reduced"),
        "results": os.path.join(w, f"results{sfx}"),
        "stats": os.path.join(w, f"results{sfx}", "stats"),
        "atlas": os.path.join(w, "atlas"),
        "figs": os.path.join(w, f"figs{sfx}"),
        "logs": os.path.join(w, "logs"),
    }
    for d in cfg["dirs"].values():
        os.makedirs(d, exist_ok=True)
    cfg["index_path"] = os.path.join(w, "index.json")
    return cfg


def config_arg(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--exp", default=None, help="experiment overlay: horizon | spatial | banded (see experiments/)")
    return p


def log(cfg, name, msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(cfg["dirs"]["logs"], f"{name}.log"), "a") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------------- naming
def present(path, min_bytes=1024):
    """True only if the file's content is downloaded. On exFAT/NTFS drives datalad
    uses unlocked mode, where a not-yet-downloaded file is a tiny pointer file
    (exists, ~100 bytes). Real files here are >= 27 kB (smallest transcript)."""
    return os.path.exists(path) and os.path.getsize(path) >= min_bytes



def ep_season(ep):
    m = EP_RE.match(ep)
    assert m, f"bad episode id {ep}"
    return int(m.group(1))


def video_path(cfg, ep):
    return os.path.join(cfg["paths"]["data_root"], "stimuli", "movies", "friends",
                        f"s{ep_season(ep)}", f"friends_{ep}.mkv")


def transcript_path(cfg, ep):
    return os.path.join(cfg["paths"]["data_root"], "stimuli", "transcripts", "friends",
                        f"s{ep_season(ep)}", f"friends_{ep}.tsv")


def friends_h5_path(cfg, sub):
    pat = os.path.join(cfg["paths"]["data_root"], "fmri", sub, "func", f"{sub}_task-friends_*_bold.h5")
    files = glob.glob(pat)
    assert len(files) == 1, f"expected 1 Friends h5 for {sub}, found {files}"
    assert present(files[0]), f"{files[0]} is not downloaded yet: run scripts/download_data.sh 1"
    return files[0]


def atlas_path(cfg, sub="sub-01"):
    files = glob.glob(os.path.join(cfg["paths"]["data_root"], "fmri", sub, "atlas", "*.nii.gz"))
    assert len(files) == 1 and present(files[0]), f"atlas missing for {sub}: {files}"
    return files[0]


def frames_path(cfg, ep):
    return os.path.join(cfg["dirs"]["frames"], f"{ep}.mp4")


def audio_path(cfg, ep):
    return os.path.join(cfg["dirs"]["audio"], f"{ep}.wav")


# ----------------------------------------------------------------------------- fMRI
def load_fmri_friends(cfg, sub, seasons=None):
    """Returns ({ep: float32 [N,1000]}, [duplicate keys skipped]).
    Keys look like 'ses-003_task-s01e01a' (tutorial strips key[13:])."""
    out, dups = {}, []
    with h5py.File(friends_h5_path(cfg, sub), "r") as h:
        keys = sorted(h.keys(), key=lambda k: int(FRIENDS_KEY_RE.match(k).group(1)) if FRIENDS_KEY_RE.match(k) else 10**9)
        for k in keys:
            m = FRIENDS_KEY_RE.match(k)
            assert m, f"unexpected h5 key {k} in {sub}"
            ep = m.group(2)
            if seasons is not None and ep_season(ep) not in seasons:
                continue
            if ep in out:          # repeated presentation: keep the first session
                dups.append(k)
                continue
            x = h[k][:].astype(np.float32)
            assert x.ndim == 2 and x.shape[1] == 1000, (k, x.shape)
            out[ep] = x
    return out, dups


def zscore_run(x):
    mu = x.mean(0, keepdims=True)
    sd = x.std(0, keepdims=True)
    return ((x - mu) / np.maximum(sd, 1e-6)).astype(np.float32)


# ----------------------------------------------------------------------------- timing
def frame_end(k, tr, fps):
    """e_k: frames with index < e_k are strictly before t_k = k*tr."""
    return int(math.floor(fps * k * tr))


def surprise_shift(tr, fps, fut_frames):
    """Smallest s such that the future window of TR k (frames e_k .. e_k+fut-1)
    lies strictly before t_{k+s} for every k. Uses floor(a+b) >= floor(a)+floor(b)."""
    s = 1
    while math.floor(s * tr * fps) < fut_frames:
        s += 1
    return s


# ----------------------------------------------------------------------------- index / io
def load_index(cfg):
    assert os.path.exists(cfg["index_path"]), "run src/01_index_data.py first"
    with open(cfg["index_path"]) as f:
        return json.load(f)


def episodes_for(index, seasons):
    return sorted(ep for ep in index["episodes"] if ep_season(ep) in seasons)


def save_npz_atomic(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def require(cond, msg):
    if not cond:
        print("ERROR:", msg, file=sys.stderr, flush=True)
        sys.exit(1)
