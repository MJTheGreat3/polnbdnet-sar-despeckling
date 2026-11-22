"""Pull a crop from a remote NATIVE legacy-format UAVSAR SLC source (flat
little-endian complex64 binaries served directly over HTTP, not a NISAR
HDF5 file) -- the 6 terrain-diversity sites picked in report.md Week 3
(S3.3) ship this way: a small .ann file plus 4x huge .slc files, one per
polarization, no HDF5 conversion step needed (unlike nisar_to_legacy.py,
which exists for the NISAR-format source).

Pulls via HTTP Range requests, chunked a few hundred rows at a time (one
request per chunk, full row width -- the crop's columns are a strict
sub-range of the full row, so rows aren't byte-contiguous across a
column-cropped span; fetching full width and slicing in memory avoids
needing one request per row). First version of this script did one
request per OUTPUT row (4096 tiny sequential requests per polarization)
and that reliably killed the connection (RemoteDisconnected / read
timeout) partway through on the real remote box -- too chatty for
whatever's on the other end (likely a connection/rate limit, not a
Range-support issue: Range itself is confirmed working, see _pol_url).
Writes a local .ann + 4x .slc crop in the exact layout uavsar_io.load_scene
already expects (same convention nisar_to_legacy.py produces), so the rest
of the pipeline needs no changes to consume it.

Usage:
    python -m pipeline.pull_legacy_crop \
        --ann-url https://downloaduav2.jpl.nasa.gov/Release32/rockmt_14107_01/rockmt_14107_20005_009_200212_L090HH_01_BU.ann \
        --out-dir pipeline/data/raw/rockmt_r0_c0 \
        --scene-id rockmt_r0_c0 \
        --row-start 0 --row-size 4096 --col-start 0 --col-size 4096
"""
import argparse
import os
import re
import time

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from .ann_parser import find_slc_segments, parse_ann

BYTES_PER_PIXEL = 8  # complex64: 2x float32
POLS = ("HH", "HV", "VH", "VV")
ROWS_PER_REQUEST = 256  # ~20MB/request at a typical ~10000-col scene width -- big
                        # enough to keep request count sane (16 per pol for a
                        # 4096-row crop, not 4096), small enough that one dropped
                        # connection only costs one chunk's retry, not the whole crop.


def _session():
    s = requests.Session()
    # connect AND read retries -- the read timeout seen on the real box happens
    # mid-transfer, not just on connect, so both need covering.
    retry = Retry(total=6, connect=6, read=6, backoff_factor=2,
                  status_forcelist=[500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    return s


def _download_text(session, url):
    r = session.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def _pol_url(ann_url, pol):
    """Swap the polarization token in the .ann URL/filename to get the
    matching .slc URL for a different polarization, e.g. ..._L090HH_01_BC.ann
    -> ..._L090HV_01_BC_s1_1x1.slc. All 4 pols share the same path/suffix,
    differing only in this one token -- but the .slc filename is NOT just
    the .ann name with the extension swapped: it's '.ann' replaced by
    '_s1_1x1.slc' (report.md S3.3: "the matching 4 .slc files sit next to
    it with the same filename stem ... _s1_1x1.slc instead of .ann").
    Confirmed via curl -I: the naive .ann-with-swapped-extension URL
    302-redirects to the JPL homepage (no such file); adding _s1_1x1 before
    .slc returns a real 200 with Accept-Ranges: bytes."""
    base = re.sub(r"\.ann$", "", ann_url)
    if f"L090{ann_url_pol(ann_url)}" not in base:
        raise ValueError(f"could not find a polarization token in {ann_url} to swap")
    swapped = base.replace(f"L090{ann_url_pol(ann_url)}", f"L090{pol}")
    return swapped + "_s1_1x1.slc"


def ann_url_pol(ann_url):
    m = re.search(r"L090(HH|HV|VH|VV)", ann_url)
    if not m:
        raise ValueError(f"could not find L090<pol> token in {ann_url}")
    return m.group(1)


def _range_get(session, url, offset, length, attempts=3):
    headers = {"Range": f"bytes={offset}-{offset + length - 1}"}
    last_exc = None
    for attempt in range(attempts):
        try:
            r = session.get(url, headers=headers, timeout=120)
            if r.status_code != 206:
                raise RuntimeError(f"expected 206 Partial Content from {url}, got {r.status_code} "
                                    f"-- server may not support Range requests")
            return r.content
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as exc:
            last_exc = exc
            wait = 2 ** attempt
            print(f"    chunk failed ({exc.__class__.__name__}), retrying in {wait}s "
                  f"(attempt {attempt + 1}/{attempts})...")
            time.sleep(wait)
    raise last_exc


def pull_crop_rows(session, slc_url, cols_total, row_start, row_size, col_start, col_size):
    """Fetch [row_start:row_start+row_size) x [col_start:col_start+col_size)
    in chunks of ROWS_PER_REQUEST full-width rows -- each chunk is one Range
    GET covering the chunk's full row width (unavoidable since the crop's
    columns aren't a contiguous byte range across multiple rows), then
    sliced down to the wanted columns in memory."""
    out = np.empty((row_size, col_size), dtype="<c8")
    row_nbytes = cols_total * BYTES_PER_PIXEL
    n_chunks = -(-row_size // ROWS_PER_REQUEST)
    for ci in range(n_chunks):
        r0 = ci * ROWS_PER_REQUEST
        r1 = min(r0 + ROWS_PER_REQUEST, row_size)
        chunk_rows = r1 - r0
        offset = (row_start + r0) * row_nbytes
        length = chunk_rows * row_nbytes
        print(f"    chunk {ci + 1}/{n_chunks} ({chunk_rows} rows, {length / 1e6:.1f} MB)...")
        raw = _range_get(session, slc_url, offset, length)
        full_width = np.frombuffer(raw, dtype="<c8").reshape(chunk_rows, cols_total)
        out[r0:r1] = full_width[:, col_start:col_start + col_size]
    return out


def pull_crop(ann_url, out_dir, scene_id, row_start, row_size, col_start, col_size):
    os.makedirs(out_dir, exist_ok=True)
    session = _session()

    ann_text = _download_text(session, ann_url)
    local_ann_src = os.path.join(out_dir, "_source.ann")
    with open(local_ann_src, "w") as f:
        f.write(ann_text)
    ann = parse_ann(local_ann_src)
    os.remove(local_ann_src)

    segments = find_slc_segments(ann)
    if not segments:
        raise ValueError(f"could not find slc rows/columns keys in {ann_url}")
    rows_total, cols_total, _ = segments[0]
    row_end = min(row_start + row_size, rows_total)
    col_end = min(col_start + col_size, cols_total)
    actual_rows = row_end - row_start
    actual_cols = col_end - col_start
    print(f"scene is {rows_total}x{cols_total}; cropping to "
          f"rows [{row_start}:{row_end}) x cols [{col_start}:{col_end}) = {actual_rows}x{actual_cols}")

    for pol in POLS:
        slc_url = _pol_url(ann_url, pol)
        print(f"  pulling {pol} from {slc_url}...")
        crop = pull_crop_rows(session, slc_url, cols_total, row_start, actual_rows, col_start, actual_cols)
        out_path = os.path.join(out_dir, f"{scene_id}_{pol}_01.slc")
        crop.tofile(out_path)
        print(f"    wrote {out_path} ({crop.nbytes / 1e6:.1f} MB)")

    ann_path = os.path.join(out_dir, f"{scene_id}.ann")
    with open(ann_path, "w") as f:
        f.write(f"; Cropped from remote legacy-format UAVSAR source: {ann_url}\n")
        f.write(f"; Original scene shape: {rows_total} x {cols_total}\n")
        f.write(f"; Crop origin (row, col): ({row_start}, {col_start})\n")
        f.write(f"slc_1_1x1 rows (pixels) = {actual_rows}\n")
        f.write(f"slc_1_1x1 columns (pixels) = {actual_cols}\n")
        f.write("slc_no_data (real) = 0.0\n")
    print(f"  wrote {ann_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ann-url", required=True, help="the HH .ann URL (report.md S3.3 table), "
                                                      "e.g. .../rockmt_14107_20005_009_200212_L090HH_01_BU.ann")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--scene-id", required=True)
    p.add_argument("--row-start", type=int, default=0)
    p.add_argument("--row-size", type=int, default=4096)
    p.add_argument("--col-start", type=int, default=0)
    p.add_argument("--col-size", type=int, default=4096)
    args = p.parse_args()
    pull_crop(args.ann_url, args.out_dir, args.scene_id,
               args.row_start, args.row_size, args.col_start, args.col_size)


if __name__ == "__main__":
    main()
