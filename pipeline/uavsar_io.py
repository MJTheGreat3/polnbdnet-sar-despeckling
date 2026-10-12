"""Step 1-2: read raw UAVSAR SLC binaries into complex arrays."""
import glob
import os

import numpy as np

from .ann_parser import find_slc_segments, parse_ann

POLARIZATIONS = ("HH", "HV", "VH", "VV")


def read_uavsar_slc(path, rows, cols, memmap=True):
    """Raw little-endian complex64 SLC binary -> (rows, cols) complex array."""
    if memmap:
        return np.memmap(path, dtype="<c8", mode="r", shape=(rows, cols))
    return np.fromfile(path, dtype="<c8").reshape(rows, cols)


def find_scene_files(scene_dir, pol):
    matches = glob.glob(os.path.join(scene_dir, f"*_{pol}_*.slc"))
    if not matches:
        raise FileNotFoundError(f"no *_{pol}_*.slc file found in {scene_dir}")
    if len(matches) > 1:
        raise ValueError(f"multiple {pol} SLC files found in {scene_dir}: {matches}")
    return matches[0]


def load_scene(scene_dir, segment_index=0, memmap=True):
    """Load one UAVSAR SLC scene: 4 polarizations + parsed .ann metadata.

    Returns dict with keys hh, hv, vh, vv (complex arrays), rows, cols,
    ann (raw parsed metadata), scene_id.
    """
    ann_files = glob.glob(os.path.join(scene_dir, "*.ann"))
    if not ann_files:
        raise FileNotFoundError(f"no .ann file in {scene_dir}")
    ann = parse_ann(ann_files[0])

    segments = find_slc_segments(ann)
    if not segments:
        raise ValueError(f"could not find slc rows/columns keys in {ann_files[0]}")
    if len(segments) > 1:
        print(
            f"warning: {len(segments)} range segments found in {scene_dir}, "
            f"using segment index {segment_index} ({segments[segment_index]})"
        )
    rows, cols, _prefix = segments[segment_index]

    data = {}
    for pol in POLARIZATIONS:
        path = find_scene_files(scene_dir, pol)
        data[pol.lower()] = read_uavsar_slc(path, rows, cols, memmap=memmap)

    scene_id = os.path.basename(os.path.normpath(scene_dir))
    return {
        "scene_id": scene_id,
        "rows": rows,
        "cols": cols,
        "ann": ann,
        **data,
    }
