"""Convert a NISAR-format RSLC HDF5 file (the real, publicly downloadable
raw quad-pol SLC source -- see DATA_SOURCE.md) into the legacy flat-binary
.ann + 4x .slc layout that pipeline/uavsar_io.py and ann_parser.py expect,
so the rest of the pipeline (written against the guide's assumed format)
needs no changes at all.

A full RSLC scene is enormous (58GB+ uncompressed across 4 polarizations,
~1.8 billion pixels) -- far too large to hold in RAM for the covariance/
multilook/Wishart steps downstream, which are dense-array operations.
This converter always extracts a spatial CROP, never the full scene;
pick a crop size your machine can hold comfortably in RAM (a few thousand
pixels per side is already a large, realistic test area -- see the
memory-budget guidance in DATA_SOURCE.md).

Usage:
    python -m pipeline.nisar_to_legacy \
        --h5 /path/or/url/to/scene.h5 \
        --out-dir data/raw/my_crop \
        --scene-id my_crop \
        --row-start 80000 --row-size 4096 \
        --col-start 3000 --col-size 4096
"""
import argparse
import os

import h5py
import numpy as np


def _open(h5_path_or_url):
    if h5_path_or_url.startswith("http://") or h5_path_or_url.startswith("https://"):
        import fsspec

        return h5py.File(fsspec.open(h5_path_or_url, mode="rb").open(), "r")
    return h5py.File(h5_path_or_url, "r")


def _find_frequencyA(hf):
    # NISAR RSLC layout: science/<band>/RSLC/swaths/frequencyA, band is LSAR or SSAR
    for band in ("LSAR", "SSAR"):
        path = f"science/{band}/RSLC/swaths/frequencyA"
        if path in hf:
            return hf[path], band
    raise KeyError("could not find science/{LSAR,SSAR}/RSLC/swaths/frequencyA in this file")


def crop_from_open_file(freq, band, h5_path_or_url, out_dir, scene_id, row_start, row_size, col_start, col_size):
    """Write one crop's .ann+.slc given an already-open NISAR HDF5 handle --
    lets a caller pull many crops from the same file without reopening/
    re-authenticating the HTTP connection for each one (see
    batch_pull_crops.py)."""
    os.makedirs(out_dir, exist_ok=True)
    rows_total, cols_total = freq["HH"].shape
    row_end = min(row_start + row_size, rows_total)
    col_end = min(col_start + col_size, cols_total)
    actual_rows = row_end - row_start
    actual_cols = col_end - col_start
    print(f"scene is {rows_total}x{cols_total}; cropping to "
          f"rows [{row_start}:{row_end}) x cols [{col_start}:{col_end}) = {actual_rows}x{actual_cols}")

    for pol in ("HH", "HV", "VH", "VV"):
        crop = freq[pol][row_start:row_end, col_start:col_end].astype("<c8")
        out_path = os.path.join(out_dir, f"{scene_id}_{pol}_01.slc")
        crop.tofile(out_path)
        print(f"  wrote {out_path}  ({crop.nbytes / 1e6:.1f} MB)")

    slant_range_spacing = float(freq["slantRangeSpacing"][()])
    swaths = freq.parent
    az_spacing = float(swaths["zeroDopplerTimeSpacing"][()]) if "zeroDopplerTimeSpacing" in swaths else 0.0

    ann_path = os.path.join(out_dir, f"{scene_id}.ann")
    with open(ann_path, "w") as f:
        f.write(f"; Converted crop from NISAR RSLC file: {h5_path_or_url}\n")
        f.write(f"; Original scene shape: {rows_total} x {cols_total}\n")
        f.write(f"; Crop origin (row, col): ({row_start}, {col_start})\n")
        f.write(f"slc_1_1x1 rows (pixels) = {actual_rows}\n")
        f.write(f"slc_1_1x1 columns (pixels) = {actual_cols}\n")
        f.write("slc_no_data (real) = 0.0\n")
        f.write(f"Slant range spacing (m) = {slant_range_spacing}\n")
        if az_spacing:
            f.write(f"Azimuth time spacing (s) = {az_spacing}\n")
    print(f"  wrote {ann_path}")


def convert(h5_path_or_url, out_dir, scene_id, row_start, row_size, col_start, col_size):
    hf = _open(h5_path_or_url)
    freq, band = _find_frequencyA(hf)

    pols = [p.decode() if isinstance(p, bytes) else p for p in freq["listOfPolarizations"][:]]
    print(f"band={band} polarizations in file={pols}")
    for pol in ("HH", "HV", "VH", "VV"):
        if pol not in pols:
            raise ValueError(f"file is missing {pol} -- not full quad-pol, can't build the covariance matrix")

    crop_from_open_file(freq, band, h5_path_or_url, out_dir, scene_id, row_start, row_size, col_start, col_size)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5", required=True, help="local path or https:// URL to the NISAR RSLC .h5 file")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--scene-id", required=True)
    p.add_argument("--row-start", type=int, default=0)
    p.add_argument("--row-size", type=int, default=4096)
    p.add_argument("--col-start", type=int, default=0)
    p.add_argument("--col-size", type=int, default=4096)
    args = p.parse_args()
    convert(args.h5, args.out_dir, args.scene_id, args.row_start, args.row_size, args.col_start, args.col_size)


if __name__ == "__main__":
    main()
