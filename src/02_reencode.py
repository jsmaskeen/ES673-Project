"""Step 02 - re-encode every indexed episode to a fixed 4 fps, 256x256 clip + 16 kHz mono wav.

Frame f of <work>/frames4/<ep>.mp4 shows the latest native frame at or before
time f/4 s (ffmpeg fps filter, start_time=0, round=up; checked by
tests/test_timing.py). The default round=near would show content from up to
0.125 s AFTER f/4, i.e. leak future frames. CPU only; run it with --jobs.

    python src/02_reencode.py --seasons 6 --jobs 4
"""
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from common import audio_path, config_arg, ep_season, frames_path, load_config, load_index, log


def vf_string(cfg):
    fps, S = cfg["video"]["fps"], cfg["video"]["size"]
    if cfg["video"]["mode"] == "letterbox":
        geom = (f"scale={S}:{S}:force_original_aspect_ratio=decrease:flags=area,"
                f"pad={S}:{S}:(ow-iw)/2:(oh-ih)/2:color=black")
    elif cfg["video"]["mode"] == "crop":   # same geometry as the HF VJEPA2 video processor
        R = int(S * 256 / 224)
        geom = f"scale=-2:{R}:flags=area,crop={S}:{S}"
    else:
        raise ValueError(cfg["video"]["mode"])
    # 720x480 DVD frames have non-square pixels: first resample to square pixels (no-op if SAR = 1)
    return f"fps={fps}:start_time=0:round=up,scale=trunc(iw*sar/2)*2:ih,setsar=1,{geom},setsar=1"


def count_frames(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path],
                         capture_output=True, text=True, check=True).stdout.strip()
    return int(out.split(",")[0])


def one(cfg, ep, info):
    vout, aout = frames_path(cfg, ep), audio_path(cfg, ep)
    side = vout.replace(".mp4", ".json")
    if os.path.exists(vout) and os.path.exists(aout) and os.path.exists(side):
        return ep, "skip"
    tmp_v, tmp_a = vout + ".tmp.mp4", aout + ".tmp.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", info["video"], "-map", "0:v:0",
                    "-an", "-vf", vf_string(cfg), "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
                    "-pix_fmt", "yuv420p", "-threads", "2", tmp_v], check=True)
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", info["video"], "-map", "0:a:0",
                    "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", tmp_a], check=True)
    n = count_frames(tmp_v)
    fps = cfg["video"]["fps"]
    err = abs(n / fps - info["duration"])
    assert err <= 0.5, f"{ep}: {n} frames at {fps} fps vs duration {info['duration']:.2f}s (err {err:.2f}s)"
    os.replace(tmp_v, vout)
    os.replace(tmp_a, aout)
    with open(side, "w") as f:
        json.dump({"n_frames": n, "fps": fps, "duration_native": info["duration"],
                   "mode": cfg["video"]["mode"], "size": cfg["video"]["size"]}, f)
    return ep, f"ok {n} frames"


def main():
    p = config_arg()
    p.add_argument("--seasons", type=int, nargs="*", default=None)
    p.add_argument("--jobs", type=int, default=2)
    args = p.parse_args()
    cfg = load_config(args.config)
    index = load_index(cfg)
    eps = [ep for ep in sorted(index["episodes"]) if args.seasons is None or ep_season(ep) in args.seasons]
    log(cfg, "reencode", f"{len(eps)} episodes, vf = {vf_string(cfg)}")
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(one, cfg, ep, index["episodes"][ep]) for ep in eps]
        for fu in as_completed(futs):
            ep, msg = fu.result()
            log(cfg, "reencode", f"{ep}: {msg}")


if __name__ == "__main__":
    main()
