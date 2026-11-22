"""Step 01 - index the downloaded data and check alignment.

Writes <work>/index.json and <work>/logs/index_data.log.
Only episodes whose .mkv AND .tsv are actually downloaded are indexed, so
re-run this after every `datalad get` (it is cheap).

Decisions (exact):
* fMRI is z-scored per run later regardless of what this reports.
* n_video_tr = floor(video_duration / TR). The official tutorial states fMRI
  runs can be LONGER than the movie, never earlier-starting. For each
  subject x episode:  keep if  n_video_tr - 3 <= N_fmri <= n_video_tr + 10,
  otherwise exclude that subject's episode and log it.
"""
import json
import subprocess
import sys

import numpy as np

from common import (config_arg, episodes_for, ep_season, load_config, load_fmri_friends, log, present,
                    transcript_path, video_path)
import os


def ffprobe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration,start_time:stream=index,codec_type,codec_name,r_frame_rate,avg_frame_rate,width,height,sample_rate,channels",
         "-of", "json", path], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    v = [s for s in d["streams"] if s["codec_type"] == "video"][0]
    a = [s for s in d["streams"] if s["codec_type"] == "audio"]
    num, den = v["avg_frame_rate"].split("/")
    return {
        "duration": float(d["format"]["duration"]),
        "start_time": float(d["format"].get("start_time", 0.0)),
        "fps_native": float(num) / float(den),
        "width": int(v["width"]), "height": int(v["height"]),
        "video_codec": v["codec_name"],
        "audio_codec": a[0]["codec_name"] if a else None,
        "audio_rate": int(a[0]["sample_rate"]) if a else None,
    }


def main():
    args = config_arg().parse_args()
    cfg = load_config(args.config)
    TR = cfg["tr"]
    seasons = sorted(set(cfg["train_seasons"] + [cfg["test_season"]]))
    index = {"episodes": {}, "subjects": {}, "missing_media": [], "excluded": []}

    n_tr = {}
    for sub in cfg["subjects"]:
        fm, dups = load_fmri_friends(cfg, sub, seasons)
        mus = np.mean([np.abs(x.mean(0)).mean() for x in fm.values()])
        sds = np.mean([x.std(0).mean() for x in fm.values()])
        already_z = bool(mus < 0.05 and 0.95 < sds < 1.05)
        index["subjects"][sub] = {"duplicates_skipped": dups, "mean_abs_mean": float(mus),
                                  "mean_sd": float(sds), "already_zscored": already_z, "episodes": []}
        log(cfg, "index_data", f"{sub}: {len(fm)} runs in seasons {seasons}; mean|mean|={mus:.3f} "
                               f"mean sd={sds:.3f} -> already z-scored: {already_z}; duplicates skipped: {dups}")
        for ep, x in fm.items():
            n_tr.setdefault(ep, {})[sub] = int(x.shape[0])

    # README: sub-02 lacks s05e20a; sub-05 lacks s04e01a, s04e01b, s04e13b
    for sub, miss in {"sub-02": ["s05e20a"], "sub-05": ["s04e01a", "s04e01b", "s04e13b"]}.items():
        for ep in miss:
            if ep_season(ep) in seasons and sub in cfg["subjects"]:
                if sub in n_tr.get(ep, {}):   # the README list is out of date; just note it
                    log(cfg, "index_data", f"note: README lists {sub} {ep} as missing, but it is present (kept)")

    for ep in sorted(n_tr):
        vp, tp = video_path(cfg, ep), transcript_path(cfg, ep)
        if not (present(vp) and present(tp)):
            index["missing_media"].append(ep)
            continue
        info = ffprobe(vp)
        n_video_tr = int(np.floor(info["duration"] / TR))
        keep = {}
        for sub, N in n_tr[ep].items():
            if n_video_tr - 3 <= N <= n_video_tr + 10:
                keep[sub] = N
                index["subjects"][sub]["episodes"].append(ep)
            else:
                index["excluded"].append({"ep": ep, "sub": sub, "N_fmri": N, "n_video_tr": n_video_tr})
                log(cfg, "index_data", f"EXCLUDE {sub} {ep}: N_fmri={N} vs video {n_video_tr} TRs")
        if not keep:
            continue
        info.update({"video": vp, "transcript": tp, "n_tr": keep, "n_tr_max": max(keep.values()),
                     "n_video_tr": n_video_tr})
        index["episodes"][ep] = info

    with open(cfg["index_path"], "w") as f:
        json.dump(index, f, indent=1)

    eps = index["episodes"]
    fps_set = sorted({round(e["fps_native"], 3) for e in eps.values()})
    res_set = sorted({(e["width"], e["height"]) for e in eps.values()})
    starts = [e["start_time"] for e in eps.values()]
    log(cfg, "index_data", f"indexed {len(eps)} episodes; missing media for {len(index['missing_media'])}; "
                           f"excluded subject-runs {len(index['excluded'])}")
    log(cfg, "index_data", f"native fps values: {fps_set}; resolutions: {res_set}; "
                           f"start_time range: {min(starts) if starts else None}..{max(starts) if starts else None}")
    for s in seasons:
        n = len(episodes_for(index, [s]))
        log(cfg, "index_data", f"season {s}: {n} episodes ready")
    if starts and max(abs(s) for s in starts) > 0.5:
        log(cfg, "index_data", "WARNING: some videos have start_time > 0.5 s; frames are still indexed "
                               "from container time 0 (= fMRI onset). Report this in the methods.")


if __name__ == "__main__":
    main()
