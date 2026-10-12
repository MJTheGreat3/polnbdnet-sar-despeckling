"""End-to-end pipeline run on real public UAVSAR MLC data (adapted route,
since raw SLC isn't publicly downloadable -- see pipeline/download_scene.py
docstring). Runs guide Steps 3, 5-11, 13 (Steps 1-2, 4 replaced by direct
MLC->covariance assembly; Step 12 skipped, MLC has no phase history left).

Writes:
    pipeline/results/stats.json   -- per-scene + aggregate stats, all checklist results
    pipeline/results/pipeline_steps.png  -- Pauli RGB through each step, few scenes
"""
import json
import os
import time

import numpy as np

from .checklist import run_checklist
from .clean_reference import build_clean_reference_from_covariance
from .covariance import check_covariance_validity
from .masking import build_valid_mask_from_covariance
from .mlc_io import load_mlc_scene, mlc_to_covariance
from .pol_nbdnet import check_validity, compute_db_norm_stats, cov_to_repr, repr_to_cov, simulate_polsar_speckle
from .splitting import split_scenes, verify_no_leakage
from .tiling import tile_pairs, tiles_to_arrays
from .visualize import pauli_rgb

RAW_DIR = "pipeline/data/raw"
PROCESSED_DIR = "pipeline/data/processed"
RESULTS_DIR = "pipeline/results"

REF_WINDOW = 9
TILE_SIZE = 64
STRIDE = 32
VALID_FRAC_THRESH = 0.9
SEED = 0


def main():
    t0 = time.time()
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    scene_ids = sorted(
        d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))
    )
    print(f"scenes found: {scene_ids}")

    split_map = split_scenes(scene_ids, ratios=(0.34, 0.33, 0.33), seed=SEED)
    verify_no_leakage(split_map)
    scene_to_split = {sid: name for name, ids in split_map.items() for sid in ids}
    print(f"split: {split_map}")

    per_scene = {}
    for sid in scene_ids:
        scene_dir = os.path.join(RAW_DIR, sid)
        mlc = load_mlc_scene(scene_dir)
        C = mlc_to_covariance(mlc)
        mask = build_valid_mask_from_covariance(C, nodata=0.0, edge_margin=2, erode_px=1)

        report_C = check_covariance_validity(C)

        C_ref, ref_keep = build_clean_reference_from_covariance(C, mask=mask, window=REF_WINDOW)
        report_ref = check_covariance_validity(C_ref)

        rng = np.random.default_rng(hash(sid) % (2**32))
        L = int(rng.integers(2, 9))
        C_noisy = simulate_polsar_speckle(C_ref, L=L, rng=rng)
        ok_noisy, report_noisy = check_validity(C_noisy)

        valid_mask = (mask) & ref_keep
        per_scene[sid] = {
            "rows": int(C.shape[0]),
            "cols": int(C.shape[1]),
            "split": scene_to_split[sid],
            "valid_pixel_fraction": float(mask.mean()),
            "looks_used_for_synthetic_noise": L,
            "C_report": report_C,
            "C_ref_report": report_ref,
            "C_noisy_report": report_noisy,
            "_C": C,
            "_C_ref": C_ref,
            "_C_noisy": C_noisy,
            "_valid_mask": valid_mask,
        }
        print(f"  {sid}: {C.shape[0]}x{C.shape[1]}, valid={mask.mean():.3f}, L={L}, "
              f"C hermitian/psd={report_C['hermitian']}/{report_C['psd']}, "
              f"C_noisy hermitian/psd={report_noisy['hermitian']}/{report_noisy['psd']}")

    # db normalization stats from TRAIN scenes only (guide mistake #6: never
    # compute normalization stats on val/test)
    train_diag_powers = []
    for sid in split_map["train"]:
        s = per_scene[sid]
        diag = np.diagonal(s["_C_ref"], axis1=-2, axis2=-1).real
        train_diag_powers.append(diag[s["_valid_mask"]])
    db_min, db_max = compute_db_norm_stats(np.concatenate([p.ravel() for p in train_diag_powers]))
    print(f"db_min={db_min:.2f} db_max={db_max:.2f} (fit on train scenes only)")

    tile_counts = {"train": 0, "val": 0, "test": 0}
    checklist_reports = {}
    for sid, s in per_scene.items():
        target_repr = cov_to_repr(s["_C_ref"], db_min=db_min, db_max=db_max)
        input_repr = cov_to_repr(s["_C_noisy"], db_min=db_min, db_max=db_max)

        tiles = list(tile_pairs(
            input_repr, target_repr, s["_valid_mask"],
            tile_size=TILE_SIZE, stride=STRIDE, valid_frac_thresh=VALID_FRAC_THRESH,
        ))
        s["n_tiles"] = len(tiles)
        tile_counts[s["split"]] += len(tiles)
        s["_example_tile"] = tiles[len(tiles) // 2] if tiles else None

        if tiles:
            inputs, targets, origins = tiles_to_arrays(tiles)
            np.savez_compressed(
                os.path.join(PROCESSED_DIR, f"{sid}.npz"),
                inputs=inputs, targets=targets, origins=origins, split=s["split"],
            )
            report = run_checklist(inputs=inputs, targets=targets, splits=split_map)
        else:
            report = {"all_ok": False, "note": "no tiles produced"}
        checklist_reports[sid] = report
        print(f"  {sid}: {len(tiles)} tiles ({s['split']}), checklist all_ok={report['all_ok']}")

    stats = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "route": "MLC (multilooked complex) -- SLC not publicly available, see download_scene.py docstring",
        "scenes": {
            sid: {k: v for k, v in s.items() if not k.startswith("_")}
            for sid, s in per_scene.items()
        },
        "split": split_map,
        "db_min": db_min,
        "db_max": db_max,
        "ref_window": REF_WINDOW,
        "tile_size": TILE_SIZE,
        "stride": STRIDE,
        "valid_frac_thresh": VALID_FRAC_THRESH,
        "tile_counts_by_split": tile_counts,
        "checklist_by_scene": checklist_reports,
        "all_scenes_all_ok": all(r["all_ok"] for r in checklist_reports.values()),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    stats_path = os.path.join(RESULTS_DIR, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote {stats_path}")

    png_path = os.path.join(RESULTS_DIR, "pipeline_steps.png")
    make_figure(per_scene, db_min, db_max, png_path)
    print(f"wrote {png_path}")

    return per_scene, stats


def make_figure(per_scene, db_min, db_max, out_path):
    import matplotlib.pyplot as plt

    scene_ids = list(per_scene.keys())
    n = len(scene_ids)
    cols = ["Raw MLC -> C\n(Pauli RGB)", "Valid mask", "Clean reference C_ref\n(large-window multilook)",
            "Synthetic noisy C_noisy\n(Wishart, L looks)", "Example tile: target", "Example tile: input"]
    fig, axes = plt.subplots(n, len(cols), figsize=(4 * len(cols), 4 * n))
    if n == 1:
        axes = axes[None, :]

    for row, sid in enumerate(scene_ids):
        s = per_scene[sid]
        vm = s["_valid_mask"]
        axes[row, 0].imshow(pauli_rgb(s["_C"], mask=vm))
        axes[row, 1].imshow(vm, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(pauli_rgb(s["_C_ref"], mask=vm))
        axes[row, 3].imshow(pauli_rgb(s["_C_noisy"], mask=vm))

        tile = s.get("_example_tile")
        if tile is not None:
            target_C = repr_to_cov(tile["target"], db_min=db_min, db_max=db_max)
            input_C = repr_to_cov(tile["input"], db_min=db_min, db_max=db_max)
            axes[row, 4].imshow(pauli_rgb(target_C))
            axes[row, 5].imshow(pauli_rgb(input_C))
        else:
            axes[row, 4].text(0.5, 0.5, "no tile", ha="center", va="center")
            axes[row, 5].text(0.5, 0.5, "no tile", ha="center", va="center")

        axes[row, 0].set_ylabel(f"{sid}\n({s['split']})", fontsize=9)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    for col, title in enumerate(cols):
        axes[0, col].set_title(title, fontsize=10)

    fig.suptitle("PolNBDNet preprocessing pipeline -- real UAVSAR MLC data, per step", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
