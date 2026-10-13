"""Pull many non-overlapping crops out of ONE flight line file in a single
pass -- opens the HDF5 file once (vs. once per crop with nisar_to_legacy.py
directly), which matters when the file is a remote URL (each open is a new
HTTP session) and when you're generating dozens of crops per flight line to
scale up the training set (see report.md Next Steps / the "go higher"
conversation).

Download the file locally first for anything beyond a handful of crops --
dozens of chunked HTTP range-reads against the same remote file is slower
and flakier than reading a local file once. The flight lines are listed in
DATA_SOURCE_AND_INSTRUCTIONS.md; `wget` them to the remote box's disk, then
point --h5 at the local path.

Usage:
    python -m pipeline.batch_pull_crops \
        --h5 /data/bermsA_09041_22031_008_220814_L090_CX_RSLC_02.h5 \
        --prefix bermsA --out-root pipeline/data/raw \
        --crop-size 4096 --max-crops 20
"""
import argparse
import os

from .nisar_to_legacy import _find_frequencyA, _open, crop_from_open_file

# Real UAVSAR flight lines commonly have a triangular no-data region near
# the first/last few hundred lines (rotated swath framing) -- masking.py
# handles it fine per-tile, but skipping it here avoids wasting whole crops
# on mostly-invalid data.
EDGE_MARGIN_ROWS = 2000


def plan_grid(rows_total, cols_total, crop_size, max_crops):
    usable_rows = rows_total - 2 * EDGE_MARGIN_ROWS
    n_row_steps = max(1, usable_rows // crop_size)
    n_col_steps = max(1, cols_total // crop_size)
    origins = []
    for ri in range(n_row_steps):
        for ci in range(n_col_steps):
            if len(origins) >= max_crops:
                return origins
            row_start = EDGE_MARGIN_ROWS + ri * crop_size
            col_start = ci * crop_size
            origins.append((row_start, col_start))
    return origins


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", required=True, help="local path (strongly preferred for many crops) or https:// URL")
    p.add_argument("--prefix", required=True, help="scene id prefix, e.g. bermsA -> bermsA_r<row>_c<col>")
    p.add_argument("--out-root", default="pipeline/data/raw")
    p.add_argument("--crop-size", type=int, default=4096)
    p.add_argument("--max-crops", type=int, default=20)
    args = p.parse_args()

    hf = _open(args.h5)
    freq, band = _find_frequencyA(hf)
    pols = [p.decode() if isinstance(p, bytes) else p for p in freq["listOfPolarizations"][:]]
    print(f"band={band} polarizations={pols}")
    for pol in ("HH", "HV", "VH", "VV"):
        if pol not in pols:
            raise ValueError(f"file is missing {pol} -- not full quad-pol")

    rows_total, cols_total = freq["HH"].shape
    origins = plan_grid(rows_total, cols_total, args.crop_size, args.max_crops)
    print(f"scene is {rows_total}x{cols_total}; planning {len(origins)} non-overlapping "
          f"{args.crop_size}x{args.crop_size} crops (all from this one flight line -- "
          f"fine to put them all in train, never split across train/val/test)")

    for row_start, col_start in origins:
        scene_id = f"{args.prefix}_r{row_start}_c{col_start}"
        out_dir = os.path.join(args.out_root, scene_id)
        ann_path = os.path.join(out_dir, f"{scene_id}.ann")
        if os.path.exists(ann_path):
            print(f"  skip {scene_id} (already pulled)")
            continue
        print(f"-- {scene_id} --")
        crop_from_open_file(freq, band, args.h5, out_dir, scene_id,
                             row_start, args.crop_size, col_start, args.crop_size)

    print(f"done: {len(origins)} crop(s) under {args.out_root}/{args.prefix}_*/ "
          f"-- add their scene ids to SCENES['train'] in prepare_training_data.py")


if __name__ == "__main__":
    main()
