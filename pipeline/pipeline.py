"""End-to-end orchestrator for PolNBDNet preprocessing (guide Steps 1-13).

Usage:
    python -m pipeline.pipeline synthetic --data-root data --window 5 \
        --ref-window 15 --tile-size 64 --stride 32
    python -m pipeline.pipeline noise2noise --data-root data --window 5 \
        --tile-size 64 --stride 32

Expects data/raw/<scene_id>/ to contain the .ann file and the four
*_HH_*.slc / *_HV_*.slc / *_VH_*.slc / *_VV_*.slc files per Step 1.
"""
import argparse
import json
import os

import numpy as np

from .checklist import run_checklist
from .clean_reference import build_clean_reference
from .covariance import build_covariance
from .masking import build_valid_mask
from .pol_nbdnet import check_validity, compute_db_norm_stats, cov_to_repr, simulate_polsar_speckle
from .splitting import split_scenes, verify_no_leakage
from .subaperture import split_scene_subapertures
from .tiling import tile_pairs, tiles_to_arrays
from .uavsar_io import load_scene


def list_scenes(raw_dir):
    return sorted(
        d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))
    )


def process_scene_synthetic(scene_dir, window, ref_window, nodata):
    scene = load_scene(scene_dir)
    mask = build_valid_mask(scene["hh"], scene["hv"], scene["vh"], scene["vv"], nodata=nodata)

    C, valid_frac = build_covariance(scene["hh"], scene["hv"], scene["vh"], scene["vv"], window=window, mask=mask)
    C_ref, ref_keep = build_clean_reference(scene["hh"], scene["hv"], scene["vh"], scene["vv"], mask=mask, window=ref_window)

    ok_c, _ = check_validity(C)
    if not ok_c:
        print(f"warning: {scene['scene_id']} target C failed Hermitian/PSD check")

    L = np.random.default_rng().integers(1, 9, size=1).item()
    C_noisy = simulate_polsar_speckle(C_ref, L=L)
    ok_n, _ = check_validity(C_noisy)
    if not ok_n:
        print(f"warning: {scene['scene_id']} simulated C_noisy failed validity check")

    valid_mask_out = (valid_frac >= 0.9) & ref_keep
    return scene["scene_id"], C_noisy, C_ref, valid_mask_out


def process_scene_noise2noise(scene_dir, window, nodata):
    scene = load_scene(scene_dir)
    mask = build_valid_mask(scene["hh"], scene["hv"], scene["vh"], scene["vv"], nodata=nodata)

    scene_a, scene_b = split_scene_subapertures(scene)
    C_a, valid_a = build_covariance(scene_a["hh"], scene_a["hv"], scene_a["vh"], scene_a["vv"], window=window, mask=mask)
    C_b, valid_b = build_covariance(scene_b["hh"], scene_b["hv"], scene_b["vh"], scene_b["vv"], window=window, mask=mask)

    valid_mask_out = (valid_a >= 0.9) & (valid_b >= 0.9)
    return scene["scene_id"], C_a, C_b, valid_mask_out


def run(mode, data_root, window, ref_window, tile_size, stride, valid_frac_thresh, nodata, seed):
    raw_dir = os.path.join(data_root, "raw")
    processed_dir = os.path.join(data_root, "processed")
    splits_dir = os.path.join(data_root, "splits")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    scene_ids = list_scenes(raw_dir)
    if not scene_ids:
        raise SystemExit(f"no scenes found under {raw_dir}")

    split_map = split_scenes(scene_ids, seed=seed)
    verify_no_leakage(split_map)
    with open(os.path.join(splits_dir, "scene_splits.json"), "w") as f:
        json.dump(split_map, f, indent=2)

    scene_to_split = {sid: name for name, ids in split_map.items() for sid in ids}

    all_diag_powers = []
    scene_results = {}
    for scene_id in scene_ids:
        scene_dir = os.path.join(raw_dir, scene_id)
        if mode == "synthetic":
            sid, C_input, C_target, valid_mask = process_scene_synthetic(scene_dir, window, ref_window, nodata)
        else:
            sid, C_input, C_target, valid_mask = process_scene_noise2noise(scene_dir, window, nodata)
        scene_results[sid] = (C_input, C_target, valid_mask)
        diag = np.diagonal(C_target, axis1=-2, axis2=-1).real
        all_diag_powers.append(diag[valid_mask])

    db_min, db_max = compute_db_norm_stats(np.concatenate([p.ravel() for p in all_diag_powers]))
    stats = {"db_min": db_min, "db_max": db_max, "window": window, "mode": mode, "tile_size": tile_size, "stride": stride}
    with open(os.path.join(processed_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    counts = {"train": 0, "val": 0, "test": 0}
    for scene_id, (C_input, C_target, valid_mask) in scene_results.items():
        input_repr = cov_to_repr(C_input, db_min=db_min, db_max=db_max)
        target_repr = cov_to_repr(C_target, db_min=db_min, db_max=db_max)

        tiles = list(tile_pairs(input_repr, target_repr, valid_mask, tile_size=tile_size, stride=stride, valid_frac_thresh=valid_frac_thresh))
        if not tiles:
            continue
        inputs, targets, origins = tiles_to_arrays(tiles)

        split_name = scene_to_split[scene_id]
        out_path = os.path.join(processed_dir, f"{scene_id}.npz")
        np.savez_compressed(out_path, inputs=inputs, targets=targets, origins=origins, split=split_name)

        with open(os.path.join(processed_dir, f"{scene_id}.metadata.json"), "w") as f:
            json.dump({"scene_id": scene_id, "split": split_name, "n_tiles": int(len(tiles))}, f, indent=2)

        counts[split_name] += len(tiles)

        report = run_checklist(inputs=inputs, targets=targets, splits=split_map)
        if not report["all_ok"]:
            print(f"warning: checklist failed for {scene_id}: {report}")

    print(f"done. tiles per split: {counts}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["synthetic", "noise2noise"])
    p.add_argument("--data-root", default="data")
    p.add_argument("--window", type=int, default=5, help="multilook window for target/independent-look covariance")
    p.add_argument("--ref-window", type=int, default=15, help="multilook window for the clean reference (synthetic mode only)")
    p.add_argument("--tile-size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--valid-frac-thresh", type=float, default=0.9)
    p.add_argument("--nodata", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    run(args.mode, args.data_root, args.window, args.ref_window, args.tile_size, args.stride, args.valid_frac_thresh, args.nodata, args.seed)


if __name__ == "__main__":
    main()
