"""Step 10 - map the 1000 challenge parcels to the 7 Yeo networks (CPU, needs internet once).

The challenge atlas is fmri/sub-0X/atlas/*_atlas-Schaefer18_parcel-1000Par7Net_desc-dseg_parcellation.nii.gz
(MNI152NLin2009cAsym). nilearn's Schaefer-2018 1000-parcel / 7-network 2 mm map
(FSL MNI152) is resampled onto it; for every challenge parcel p the majority
nilearn label inside it is found.
Decision: if the majority label equals p for >= 90% of parcels, the orders are
identical and network(p) = network of nilearn label p; otherwise network(p) =
network of the majority label. Writes <work>/atlas/parcel_networks.json.
"""
import json
import os

import nibabel as nib
import numpy as np
from nilearn import datasets, image

from common import atlas_path, config_arg, load_config, log

NETWORKS = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]


def main():
    args = config_arg().parse_args()
    cfg = load_config(args.config)
    img = nib.load(atlas_path(cfg, cfg["subjects"][0]))
    A = np.asarray(img.dataobj).astype(int)
    assert set(np.unique(A)) <= set(range(0, 1001)) and A.max() == 1000, "unexpected atlas labels"
    sch = datasets.fetch_atlas_schaefer_2018(n_rois=1000, yeo_networks=7, resolution_mm=2,
                                             data_dir=os.path.join(cfg["dirs"]["atlas"], "nilearn"))
    labels = [l.decode() if isinstance(l, bytes) else str(l) for l in sch.labels]
    labels = [l for l in labels if l != "Background"]
    assert len(labels) == 1000, len(labels)
    ref = np.asarray(image.resample_to_img(sch.maps, img, interpolation="nearest",
                                           force_resample=True, copy_header=True).dataobj).astype(int)
    major = np.zeros(1001, int)
    for p in range(1, 1001):
        v = ref[(A == p) & (ref > 0)]
        major[p] = np.bincount(v).argmax() if v.size else 0
    agreement = float(np.mean(major[1:] == np.arange(1, 1001)))
    dice = float(2 * ((A > 0) & (ref > 0)).sum() / ((A > 0).sum() + (ref > 0).sum()))
    use = np.arange(1, 1001) if agreement >= 0.9 else major[1:]
    assert (use > 0).all(), "some parcels have no overlap with the reference atlas"
    nets = [labels[u - 1].split("_")[2] for u in use]
    assert set(nets) <= set(NETWORKS), set(nets)
    cent = []
    for p in range(1, 1001):
        ijk = np.argwhere(A == p).mean(0)
        cent.append((img.affine @ np.r_[ijk, 1])[:3].tolist())
    out = {"networks": nets, "labels": [labels[u - 1] for u in use], "order_agreement": agreement,
           "mask_dice": dice, "rule": "identity" if agreement >= 0.9 else "majority", "centroids_mm": cent}
    with open(os.path.join(cfg["dirs"]["atlas"], "parcel_networks.json"), "w") as f:
        json.dump(out, f)
    counts = {n: nets.count(n) for n in NETWORKS}
    log(cfg, "atlas", f"order agreement {agreement:.3f}, mask Dice {dice:.3f}, rule={out['rule']}, counts {counts}")


if __name__ == "__main__":
    main()
