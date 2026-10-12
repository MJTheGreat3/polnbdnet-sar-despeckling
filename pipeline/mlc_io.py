"""Read UAVSAR MLC (multilooked complex) products and assemble the 3x3
polarimetric covariance matrix C directly -- no custom multilook needed,
the MLC processor already did that averaging (guide Step 4 is a no-op
here; this module stands in for guide Steps 1-4 when SLC isn't available).

MLC file layout per scene (6 files):
    HHHH, HVHV, VVVV  -- REAL*4, little-endian, one float per pixel (power)
    HHHV, HHVV, HVVV  -- complex64, little-endian (cross-products)
"""
import glob
import os

import numpy as np

from .ann_parser import parse_ann

MLC_CHANNELS = ("HHHH", "HVHV", "VVVV", "HHHV", "HHVV", "HVVV")
_POWER_CHANNELS = ("HHHH", "HVHV", "VVVV")


def find_mlc_dims(ann):
    rows = ann.get("mlc_pwr.set_rows") or ann.get("mlc_mag.set_rows")
    cols = ann.get("mlc_pwr.set_cols") or ann.get("mlc_mag.set_cols")
    if rows is None or cols is None:
        raise ValueError("could not find mlc_pwr.set_rows/set_cols in .ann")
    return int(rows), int(cols)


def find_channel_file(scene_dir, channel):
    matches = glob.glob(os.path.join(scene_dir, f"*{channel}*.mlc"))
    if not matches:
        raise FileNotFoundError(f"no *{channel}*.mlc file found in {scene_dir}")
    if len(matches) > 1:
        raise ValueError(f"multiple {channel} MLC files found in {scene_dir}: {matches}")
    return matches[0]


def read_mlc_channel(path, rows, cols, channel, memmap=True):
    dtype = "<f4" if channel in _POWER_CHANNELS else "<c8"
    if memmap:
        return np.memmap(path, dtype=dtype, mode="r", shape=(rows, cols))
    return np.fromfile(path, dtype=dtype).reshape(rows, cols)


def load_mlc_scene(scene_dir, memmap=True):
    ann_files = glob.glob(os.path.join(scene_dir, "*.ann"))
    if not ann_files:
        raise FileNotFoundError(f"no .ann file in {scene_dir}")
    ann = parse_ann(ann_files[0])
    rows, cols = find_mlc_dims(ann)

    data = {}
    for ch in MLC_CHANNELS:
        path = find_channel_file(scene_dir, ch)
        data[ch] = read_mlc_channel(path, rows, cols, ch, memmap=memmap)

    scene_id = os.path.basename(os.path.normpath(scene_dir))
    return {"scene_id": scene_id, "rows": rows, "cols": cols, "ann": ann, **data}


def mlc_to_covariance(mlc):
    """Assemble the 3x3 Hermitian covariance matrix C directly from the
    6 MLC channels: C11=HHHH, C22=HVHV, C33=VVVV (real diagonal power),
    C12=HHHV, C13=HHVV, C23=HVVV (complex cross-products), lower
    triangle filled by conjugate symmetry.
    """
    rows, cols = mlc["HHHH"].shape
    C = np.zeros((rows, cols, 3, 3), dtype=np.complex64)
    C[..., 0, 0] = mlc["HHHH"]
    C[..., 1, 1] = mlc["HVHV"]
    C[..., 2, 2] = mlc["VVVV"]
    C[..., 0, 1] = mlc["HHHV"]
    C[..., 1, 0] = np.conj(mlc["HHHV"])
    C[..., 0, 2] = mlc["HHVV"]
    C[..., 2, 0] = np.conj(mlc["HHVV"])
    C[..., 1, 2] = mlc["HVVV"]
    C[..., 2, 1] = np.conj(mlc["HVVV"])
    return C
