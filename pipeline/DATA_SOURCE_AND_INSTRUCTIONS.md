# PolNBDNet Raw SLC Preprocessing — Handoff

This package implements the preprocessing pipeline described in
`PreProc_Guide.pdf` ("UAVSAR Data Preprocessing for Polarimetric
Despeckling (PolNBDNet)") — turning raw polarimetric SAR SLC data into
training pairs for a despeckling network. It's Python, tested, and works
end-to-end on real data — it just needs to run on a machine with more
RAM/disk than what I had available.

## What's in this package

| File | Guide step(s) | What it does |
|---|---|---|
| `ann_parser.py` | 1–2 | Parses UAVSAR `.ann` metadata files (handles both the old `slc_1_1x1 rows` key style and the current `slc_mag.set_rows` style) |
| `uavsar_io.py` | 1–2 | Reads raw little-endian complex64 `.slc` binaries |
| `masking.py` | 3 | Builds the valid-pixel mask (no-data, edges, optional erosion) |
| `covariance.py` | 4 | Builds the 3×3 polarimetric covariance matrix `C` via windowed multilook of the scattering vector; symmetrizes HV/VH; Hermitian/PSD validity checks |
| `radiometry.py` | 5 | Linear-power vs dB handling, incidence-angle trend check, layover/shadow flagging |
| `pol_nbdnet.py` | 7, 9 | `cov_to_repr`/`repr_to_cov` (3×3 complex `C` ↔ 9-channel real representation), `compute_db_norm_stats`, `simulate_polsar_speckle` (Wishart noise draw), `check_validity`, `wishart_distance` |
| `clean_reference.py` | 8 | Heavily-multilooked "clean" reference `C_ref` |
| `subaperture.py` | 12 | Splits SLC into two independent azimuth sub-looks (real Noise2Noise pairs, no synthetic noise needed) |
| `tiling.py` | 10 | Cuts matched (input, target) pairs into pixel-aligned tiles |
| `splitting.py` | 11 | Scene-level train/val/test split, leakage check |
| `checklist.py` | 13 | Automated version of the guide's final pre-training checklist |
| `pipeline.py` | all | CLI orchestrator: `python -m pipeline.pipeline synthetic \| noise2noise --data-root data` |
| `visualize.py` | — | Pauli RGB rendering, for the visual sanity checks the guide asks for at several steps |
| `nisar_to_legacy.py` | — | **New, not from the guide** — converts a real NISAR-format RSLC HDF5 file (see below) into the flat-binary `.ann`+`.slc` layout the rest of this pipeline expects |

Everything except `nisar_to_legacy.py` is a direct implementation of what
the guide describes, unmodified by the data-source problem below.

## Where to actually get raw SLC data (read this first)

This turned out to be the hard part. Two things I confirmed directly,
not from documentation:

**ASF (Alaska Satellite Facility), the standard public UAVSAR archive,
does not host raw SLC for UAVSAR at all.** I scanned all 12,521 UAVSAR
granules on ASF programmatically (via `asf_search`) — zero have a `.slc`
file attached. ASF only carries MLC/GRD-level (already multilooked)
UAVSAR products.

**Real raw quad-pol SLC does exist publicly, but in NISAR-format HDF5,
not the legacy flat-binary format the guide assumes**, at JPL's own
download page:

```
https://uavsar.jpl.nasa.gov/cgi-bin/download-files.pl
```

Look under the **"LBand RSLC"** and **"PBand RSLC"** sections. No login
required — plain HTTPS. I verified these are real (HDF5 magic bytes,
actual complex radar data, not placeholder files) by opening them
directly over the network.

Each file is a full flight line: ~183,000 azimuth lines × 9,900 range
samples × 4 polarizations, complex64. That's **41–118 GB per scene**.
Smallest available right now:

| Scene | Size | URL |
|---|---|---|
| bermsR_09037 | 41.15 GB | `https://uavsar.jpl.nasa.gov/DOWNLOAD_FILES/LBand-RSLC/bermsR_09037_22031_000_220814_L090_CX_RSLC_02.h5` |
| bermsR_27037 | 41.15 GB | `https://uavsar.jpl.nasa.gov/DOWNLOAD_FILES/LBand-RSLC/bermsR_27037_22031_001_220814_L090_CX_RSLC_02.h5` |
| bermsA_09041 | 41.15 GB | `https://uavsar.jpl.nasa.gov/DOWNLOAD_FILES/LBand-RSLC/bermsA_09041_22031_008_220814_L090_CX_RSLC_02.h5` |
| bermsA_27041 | 41.15 GB | `https://uavsar.jpl.nasa.gov/DOWNLOAD_FILES/LBand-RSLC/bermsA_27041_22031_009_220814_L090_CX_RSLC_02.h5` |
| bermsB_09043 | 41.15 GB | `https://uavsar.jpl.nasa.gov/DOWNLOAD_FILES/LBand-RSLC/bermsB_09043_22031_012_220814_L090_CX_RSLC_02.h5` |
| bermsB_27043 | 41.15 GB | `https://uavsar.jpl.nasa.gov/DOWNLOAD_FILES/LBand-RSLC/bermsB_27043_22031_013_220814_L090_CX_RSLC_02.h5` |
| Howlnd_16701 / _34701 | 58.0 GB each | `.../LBand-RSLC/Howlnd_16701_22025_000_220721_L090_CX_RSLC_02.h5` |
| pongaA..F, pongar | 32–43 GB each | `.../PBand-RSLC/pongaA_27542_24021_004_240602_PL09040080_CX_RSLC_02.h5` etc |
| mondaA..E, mondah | 48.4 GB each (LBand) / 32 GB (PBand) | `.../LBand-RSLC/mondaA_09137_24038_004_240706_L090_CX_RSLC_02.h5` etc |
| ogoouA/B, ogooue (x2) | 118 GB each | largest, avoid unless you specifically need this site |

Full listing is on the page itself — it also gives `wget` commands per
category if you'd rather script the download than click through.

Internally, each file stores the four scattering channels at:

```
science/LSAR/RSLC/swaths/frequencyA/HH   (rows, cols) complex64
science/LSAR/RSLC/swaths/frequencyA/HV   (rows, cols) complex64
science/LSAR/RSLC/swaths/frequencyA/VH   (rows, cols) complex64
science/LSAR/RSLC/swaths/frequencyA/VV   (rows, cols) complex64
```

— exactly the raw SLC scattering vector the guide's Step 1–2 wants, just
packaged as HDF5 instead of the classic UAVSAR flat-binary `.ann`+`.slc`.

## Memory budget — read before you run anything

A full scene is on the order of **1.8 billion pixels**. Each
polarization channel alone is ~14.5 GB as complex64; four channels is
~58 GB; the dense 3×3 covariance array built from them (`covariance.py`,
guide Step 4) would be on the order of a few hundred GB. This pipeline —
like the guide itself — processes a scene as one dense in-memory array
per step (masking, covariance, clean reference, Wishart simulation all
operate on the whole array at once). **That will not fit in RAM at full
scene size on typical hardware.**

Work on a spatial **crop**, not the full flight line. A few thousand
pixels per side (e.g. 4096×4096 ≈ 16.8M pixels) is already a large,
realistic test region, fits comfortably in RAM, and is genuinely useful
training data — you don't need the whole flight line for this.

## Getting a crop into the pipeline

`nisar_to_legacy.py` pulls a crop directly from the HDF5 file and writes
it out in the exact `.ann` + `.slc` layout the rest of the pipeline
expects, so nothing else needs to change.

The datasets inside these files are **chunked** (512×512 in the ones I
checked), so `h5py`+`fsspec` can fetch just the chunks overlapping your
requested crop over HTTP — you don't have to download the full 41–118 GB
file first if you don't want to (though for repeated experiments it's
faster to download once locally and point `--h5` at the local path).

```bash
pip install -r requirements.txt

python -m pipeline.nisar_to_legacy \
    --h5 https://uavsar.jpl.nasa.gov/DOWNLOAD_FILES/LBand-RSLC/bermsR_09037_22031_000_220814_L090_CX_RSLC_02.h5 \
    --out-dir data/raw/bermsR_crop1 \
    --scene-id bermsR_crop1 \
    --row-start 80000 --row-size 4096 \
    --col-start 3000  --col-size 4096
```

This writes `data/raw/bermsR_crop1/bermsR_crop1_{HH,HV,VH,VV}_01.slc`
and `bermsR_crop1.ann`. Pick different `--row-start`/`--col-start` (or a
different `--h5` scene entirely) to make more crops — you'll want a few
for a meaningful train/val/test split (Step 11). **Don't split crops
from the same flight line across train/val/test** — guide Step 11
explicitly warns this causes leakage (neighboring regions share real
texture and partially correlated speckle); keep each flight line's crops
in one split.

## Running the pipeline

Two training-pair routes, per guide Steps 6–12:

```bash
# Recommended when SLC is available (guide Step 12): real independent
# looks from splitting the synthetic aperture, no synthetic noise model.
python -m pipeline.pipeline noise2noise --data-root data \
    --window 5 --tile-size 64 --stride 32

# Alternative (guide Steps 8-9): synthetic Wishart-simulated speckle on
# top of a heavily-averaged "clean" reference.
python -m pipeline.pipeline synthetic --data-root data \
    --window 5 --ref-window 15 --tile-size 64 --stride 32
```

Output: `data/processed/<scene>.npz` (`inputs`, `targets`, `origins`,
`split`), `data/processed/stats.json`, `data/splits/scene_splits.json`.

## What I actually verified

I don't have 58 GB+ of disk/RAM to spare here, so I tested this on a
real but small crop (512×512, pulled live over HTTP from the Howlnd
LBand RSLC file above, no local download) rather than a full scene:

- `nisar_to_legacy.py` correctly pulls the crop and writes valid
  `.ann`+`.slc` files
- Both `noise2noise` and `synthetic` routes ran clean end-to-end on that
  real crop — produced valid tiles, all Hermitian/PSD checks passed
- The full checklist (`checklist.py`, guide Step 13) reported all-clear

So the code itself is real and tested against real UAVSAR data, not
just synthetic test fixtures — it just needs to be run at a bigger scale
than I could do here. At 4096×4096 or similar it should behave the same
way, just slower and with more tiles.

## A few things worth knowing going in

- `pol_nbdnet.py`'s `simulate_polsar_speckle` had two real bugs that
  only showed up on actual (non-idealized) data: it could OOM on a
  full-scene array (fixed with internal chunking), and it could crash
  on `Cholesky` for low-confidence edge pixels that aren't perfectly
  PSD (fixed with an eigenvalue-floor projection before decomposition).
  Both are already fixed in this package — mentioning it so you know
  the robustness work is done, not something to redo.
- If a crop straddles the edge of the flight swath (rotated rectangular
  framing — common near the first/last few hundred lines), expect a
  triangular no-data region; `masking.py` handles it correctly, just
  don't be surprised by it in a Pauli RGB preview.
