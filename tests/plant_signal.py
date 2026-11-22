"""Replace the fake fMRI with a planted signal built from the reduced bands, so the
encoding + stats steps must recover known effects:
  parcels   0- 99 : delayed FORE  (-> fore_vs_none > 0 and H1_fore_vs_rand > 0 there)
  parcels 100-199 : delayed SURP  (-> surp_vs_none > 0 and H2_surp_vs_rand > 0 there)
  parcels 200-299 : delayed CUR   (-> neither forecast nor surprise contrast)
  others          : noise
    python tests/plant_signal.py --config tests/config_test.yaml
"""
import glob
import importlib
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from common import config_arg, friends_h5_path, load_config, FRIENDS_KEY_RE  # noqa: E402
fit = importlib.import_module("09_fit_encoding")


def main():
    p = config_arg()
    p.add_argument("--plant", default="FORE:0:100,SURP:100:200,CUR:200:300", help="band:first:last parcels, comma-separated")
    args = p.parse_args()
    cfg = load_config(args.config, args.exp)
    plant = [(b, slice(int(i), int(j))) for b, i, j in (x.split(":") for x in args.plant.split(","))]
    delays = cfg["encoding"]["delays"]
    rng = np.random.default_rng(1)
    W = {}
    for sub in cfg["subjects"]:
        fn = friends_h5_path(cfg, sub)
        with h5py.File(fn, "r+") as h:
            for key in list(h.keys()):
                ep = FRIENDS_KEY_RE.match(key).group(2)
                R = np.load(os.path.join(cfg["dirs"]["reduced"], f"{ep}.npz"))
                N = h[key].shape[0]
                n = min(N, len(R["valid"]))
                Y = rng.standard_normal((N, 1000)).astype(np.float32)
                for band, sl in plant:
                    X = fit.delay_ep(R[band][:n], delays)
                    if band not in W:
                        W[band] = np.random.default_rng(7).standard_normal((X.shape[1], 100)).astype(np.float32)
                    s = X @ W[band]
                    s /= (s.std(0, keepdims=True) + 1e-6)
                    Y[:n, sl] += 1.0 * s
                del h[key]
                h.create_dataset(key, data=Y)
    print("planted signal into", len(cfg["subjects"]), "subjects")


if __name__ == "__main__":
    main()
