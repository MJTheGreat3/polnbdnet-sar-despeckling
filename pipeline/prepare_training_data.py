"""Build train/val/test tiles for the NBDNet Wishart-noise effectiveness
experiment.

Train/val scenes get per-tile random look-count L in {1,2,4} (matching
the paper's own training setup and guide Step 9.4's suggestion to vary L
so the network generalizes across noise levels). The held-out test scene
gets a SEPARATE tile set per fixed L in {1,2,4,8}, same tile origins
across all L, so recovery quality can be compared per noise severity
(guide Step 13 / paper Table 1 style evaluation).

One scene per split (howlnd=train, bermsr=val, monda=test) -- distinct
real sites (Maine forest / Canadian boreal / Gabon tropical), so no
scene-leakage concern (guide Step 11).
"""
import json
import os
import time

import numpy as np

from .clean_reference import build_clean_reference
from .covariance import check_covariance_validity
from .masking import build_valid_mask
from .pol_nbdnet import compute_db_norm_stats, cov_to_repr, simulate_polsar_speckle
from .uavsar_io import load_scene

DATA_ROOT = "pipeline/data"
RAW_DIR = os.path.join(DATA_ROOT, "raw")
OUT_DIR = os.path.join(DATA_ROOT, "experiment")

REF_WINDOW = 15
TILE_SIZE = 64
STRIDE = 64
VALID_FRAC_THRESH = 0.95
TRAIN_L_CHOICES = (1, 2, 4)
TEST_L_VALUES = (1, 2, 4, 8)

SCENES = {"train": "howlnd", "val": "bermsr", "test": "monda"}


def enumerate_tile_origins(valid_mask, tile_size=TILE_SIZE, stride=STRIDE, thresh=VALID_FRAC_THRESH):
    H, W = valid_mask.shape
    origins = []
    for r in range(0, H - tile_size + 1, stride):
        for c in range(0, W - tile_size + 1, stride):
            if valid_mask[r:r + tile_size, c:c + tile_size].mean() >= thresh:
                origins.append((r, c))
    return origins


def build_scene_clean_ref(scene_id):
    scene = load_scene(os.path.join(RAW_DIR, scene_id))
    mask = build_valid_mask(scene["hh"], scene["hv"], scene["vh"], scene["vv"], nodata=0.0, edge_margin=4)
    C_ref, keep = build_clean_reference(scene["hh"], scene["hv"], scene["vh"], scene["vv"],
                                         mask=mask, window=REF_WINDOW, min_valid_frac=VALID_FRAC_THRESH)
    report = check_covariance_validity(C_ref)
    valid_mask = mask & keep
    return C_ref, valid_mask, report


def build_tiles(C_ref, origins, db_min, db_max, rng, fixed_L=None, L_choices=None):
    inputs, targets, Ls = [], [], []
    for (r, c) in origins:
        crop = C_ref[r:r + TILE_SIZE, c:c + TILE_SIZE]
        L = fixed_L if fixed_L is not None else int(rng.choice(L_choices))
        noisy = simulate_polsar_speckle(crop, L=L, rng=rng)
        inputs.append(cov_to_repr(noisy, db_min=db_min, db_max=db_max))
        targets.append(cov_to_repr(crop, db_min=db_min, db_max=db_max))
        Ls.append(L)
    return (np.stack(inputs).astype("float32"), np.stack(targets).astype("float32"), np.array(Ls, dtype="int8"))


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    print("building clean references...")
    refs = {}
    for split, sid in SCENES.items():
        C_ref, valid_mask, report = build_scene_clean_ref(sid)
        origins = enumerate_tile_origins(valid_mask)
        refs[split] = {"scene_id": sid, "C_ref": C_ref, "valid_mask": valid_mask, "origins": origins, "report": report}
        print(f"  {split}={sid}: {C_ref.shape[:2]}, {len(origins)} valid tile origins, "
              f"hermitian/psd={report['hermitian']}/{report['psd']}")

    train_diag = np.diagonal(refs["train"]["C_ref"], axis1=-2, axis2=-1).real
    db_min, db_max = compute_db_norm_stats(train_diag[refs["train"]["valid_mask"]])
    print(f"db_min={db_min:.2f} db_max={db_max:.2f} (fit on train scene only)")

    meta = {"db_min": db_min, "db_max": db_max, "ref_window": REF_WINDOW, "tile_size": TILE_SIZE,
            "stride": STRIDE, "scenes": SCENES, "train_L_choices": TRAIN_L_CHOICES, "test_L_values": TEST_L_VALUES}

    for split in ("train", "val"):
        r = refs[split]
        inputs, targets, Ls = build_tiles(r["C_ref"], r["origins"], db_min, db_max, rng, L_choices=TRAIN_L_CHOICES)
        out_path = os.path.join(OUT_DIR, f"{split}.npz")
        np.savez_compressed(out_path, inputs=inputs, targets=targets, Ls=Ls)
        print(f"  wrote {out_path}: {inputs.shape[0]} tiles, L distribution "
              f"{dict(zip(*np.unique(Ls, return_counts=True)))}")

    r = refs["test"]
    for L in TEST_L_VALUES:
        inputs, targets, Ls = build_tiles(r["C_ref"], r["origins"], db_min, db_max, rng, fixed_L=L)
        out_path = os.path.join(OUT_DIR, f"test_L{L}.npz")
        np.savez_compressed(out_path, inputs=inputs, targets=targets, Ls=Ls)
        print(f"  wrote {out_path}: {inputs.shape[0]} tiles, fixed L={L}")

    meta["elapsed_seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done in {meta['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
