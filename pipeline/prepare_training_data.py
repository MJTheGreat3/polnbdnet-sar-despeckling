"""Build train/val/test tiles for the NBDNet Wishart-noise effectiveness
experiment.

Train/val scenes get per-tile random look-count L in {1,2,4} (matching
the paper's own training setup and guide Step 9.4's suggestion to vary L
so the network generalizes across noise levels). The held-out test scene
gets a SEPARATE tile set per fixed L in {1,2,4,8}, same tile origins
across all L, so recovery quality can be compared per noise severity
(guide Step 13 / paper Table 1 style evaluation).

Each split takes a LIST of scene ids (not just one) -- add more flight
lines to "train" to grow/diversify the training set once you have the
compute/disk for it. No scene id may appear in more than one split
(guide Step 11: splitting crops from the same flight line across
train/val/test leaks neighboring-region texture/speckle correlation).

Each scene normalizes itself: db_min/db_max (1st/99th percentile of that
scene's own valid diagonal power) are computed PER SCENE, not fit on
train and reused everywhere. This is a radiometric calibration step
(every flight line has its own absolute brightness/dynamic range), not a
label statistic, so doing it per-scene isn't test-set leakage -- it's
what removed the cross-scene blob/colour artifacts (report.md Next
Steps #1) caused by applying one scene's brightness range to another.
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

# batch_pull_crops.py names each crop "<prefix>_r<row>_c<col>" -- any
# scene id starting with one of these prefixes is auto-added to train, so a
# 100+ scene pull (see pull_200gb_dataset.sh) doesn't need a hand-typed list.
# Keep val/test scenes' flight lines out of this list (leakage, guide Step 11).
AUTO_TRAIN_PREFIXES = ["bermsR_r", "bermsA1_r", "bermsA2_r", "bermsB1_r", "bermsB2_r"]


def discover_scenes(prefixes, raw_dir=RAW_DIR):
    if not os.path.isdir(raw_dir):
        return []
    found = []
    for name in sorted(os.listdir(raw_dir)):
        if any(name.startswith(p) for p in prefixes) and os.path.isfile(os.path.join(raw_dir, name, f"{name}.ann")):
            found.append(name)
    return found


# Add more scene ids here (one list entry per crop you pulled with
# nisar_to_legacy.py) to grow the training set. Keep val/test scenes
# distinct flight lines never touched during train.
SCENES = {"train": ["howlnd"] + discover_scenes(AUTO_TRAIN_PREFIXES), "val": ["bermsr"], "test": ["monda"]}


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


def _scene_done(split_dir, sid, split):
    """True if sid's shard(s) already exist AND carry the norm/count sidecar
    fields -- lets a re-run after a crash/interruption skip finished scenes
    entirely, without even rebuilding their (expensive, ~8GB+) C_ref."""
    if split == "test":
        paths = [os.path.join(split_dir, f"{sid}_L{L}.npz") for L in TEST_L_VALUES]
    else:
        paths = [os.path.join(split_dir, f"{sid}.npz")]
    if not all(os.path.exists(p) for p in paths):
        return None
    try:
        d = np.load(paths[0])
        return float(d["db_min"]), float(d["db_max"]), int(d["n_tiles"])
    except Exception:
        return None


def process_scene(sid, split, rng):
    """Build one scene's C_ref, tile it, write its shard(s), then let C_ref
    be garbage-collected before the caller moves to the next scene.

    build_covariance() materializes 2-3 dense (H,W,3,3) complex128 arrays
    (~470 bytes/pixel) before windowing -- at 8192x8192 that's ~32GB for
    ONE scene. Processing scene-by-scene and freeing C_ref immediately
    keeps peak memory bounded by the single largest scene, not by how
    many scenes (or their total size) you add to train."""
    split_dir = os.path.join(OUT_DIR, split)
    os.makedirs(split_dir, exist_ok=True)

    done = _scene_done(split_dir, sid, split)
    if done is not None:
        db_min, db_max, n_tiles = done
        print(f"  {split}={sid}: already done ({n_tiles} tiles, db_min={db_min:.2f} db_max={db_max:.2f}), skipping")
        return db_min, db_max, n_tiles

    C_ref, valid_mask, report = build_scene_clean_ref(sid)
    origins = enumerate_tile_origins(valid_mask)
    diag = np.diagonal(C_ref, axis1=-2, axis2=-1).real
    db_min, db_max = compute_db_norm_stats(diag[valid_mask])
    print(f"  {split}={sid}: {C_ref.shape[:2]}, {len(origins)} valid tile origins, "
          f"hermitian/psd={report['hermitian']}/{report['psd']}, "
          f"db_min={db_min:.2f} db_max={db_max:.2f} (own scene stats)")

    if split == "test":
        for L in TEST_L_VALUES:
            inputs, targets, Ls = build_tiles(C_ref, origins, db_min, db_max, rng, fixed_L=L)
            out_path = os.path.join(split_dir, f"{sid}_L{L}.npz")
            np.savez_compressed(out_path, inputs=inputs, targets=targets, Ls=Ls,
                                 db_min=db_min, db_max=db_max, n_tiles=len(origins))
            print(f"    wrote {out_path}: {inputs.shape[0]} tiles, fixed L={L}")
    else:
        inputs, targets, Ls = build_tiles(C_ref, origins, db_min, db_max, rng, L_choices=TRAIN_L_CHOICES)
        out_path = os.path.join(split_dir, f"{sid}.npz")
        np.savez_compressed(out_path, inputs=inputs, targets=targets, Ls=Ls,
                             db_min=db_min, db_max=db_max, n_tiles=len(origins))
        print(f"    wrote {out_path}: {inputs.shape[0]} tiles, "
              f"L distribution {dict(zip(*np.unique(Ls, return_counts=True)))}")

    return db_min, db_max, len(origins)  # C_ref/valid_mask/origins go out of scope here -> GC'd


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(0)

    all_ids = [sid for ids in SCENES.values() for sid in ids]
    dupes = {sid for sid in all_ids if all_ids.count(sid) > 1}
    if dupes:
        raise ValueError(f"scene id(s) {dupes} appear in more than one split -- leakage (guide Step 11)")

    per_scene_norm = {}
    tile_counts = {}
    for split, sids in SCENES.items():
        print(f"building {split} ({len(sids)} scene(s))...")
        for sid in sids:
            db_min, db_max, n_tiles = process_scene(sid, split, rng)
            per_scene_norm[sid] = [db_min, db_max]
            tile_counts[sid] = n_tiles

    meta = {"per_scene_norm": per_scene_norm, "tile_counts": tile_counts, "ref_window": REF_WINDOW,
            "tile_size": TILE_SIZE, "stride": STRIDE, "scenes": SCENES, "train_L_choices": TRAIN_L_CHOICES,
            "test_L_values": TEST_L_VALUES, "elapsed_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"done in {meta['elapsed_seconds']}s. Tiles are per-scene shards under "
          f"{OUT_DIR}/{{train,val,test}}/ -- train_nbdnet.py streams them instead of "
          f"loading everything into RAM at once.")


if __name__ == "__main__":
    main()
