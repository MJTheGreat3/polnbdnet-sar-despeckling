#!/usr/bin/env bash
# Pulls a ~200GB training set: re-pulls the 3 existing scenes (gitignored,
# not in the repo) + 5 new UAVSAR LBand RSLC flight lines, each cut into 25
# non-overlapping 4096x4096 crops (125 new train scenes total). URLs below
# are confirmed-real (from pipeline/DATA_SOURCE_AND_INSTRUCTIONS.md), not
# guessed -- these 5 are genuinely different flight passes from the 3
# scenes already in use (no leakage across train/val/test, guide Step 11).
#
# Run from the repo root with the polnbdnet conda env active. Safe to
# re-run: wget -c resumes partial downloads, and both nisar_to_legacy.py
# and batch_pull_crops.py skip scenes/crops that already exist on disk.
#
# After this finishes: python -m pipeline.prepare_training_data
# (also resumable -- see prepare_training_data.py's _scene_done)
set -euo pipefail

H5_CACHE="$HOME/uavsar_h5"
RAW_DIR=pipeline/data/raw
mkdir -p "$H5_CACHE"

echo "=== re-pulling the 3 existing scenes (train anchor / val / test) ==="

[ -f "$RAW_DIR/howlnd/howlnd.ann" ] || python -m pipeline.nisar_to_legacy \
    --h5 https://uavsar.jpl.nasa.gov/DOWNLOAD_FILES/LBand-RSLC/Howlnd_16701_22025_000_220721_L090_CX_RSLC_02.h5 \
    --out-dir "$RAW_DIR/howlnd" --scene-id howlnd \
    --row-start 60000 --row-size 4096 --col-start 2500 --col-size 4096

[ -f "$RAW_DIR/monda/monda.ann" ] || python -m pipeline.nisar_to_legacy \
    --h5 https://uavsar.jpl.nasa.gov/DOWNLOAD_FILES/LBand-RSLC/mondaA_09137_24038_004_240706_L090_CX_RSLC_02.h5 \
    --out-dir "$RAW_DIR/monda" --scene-id monda \
    --row-start 50000 --row-size 4096 --col-start 2500 --col-size 4096

[ -f "$RAW_DIR/bermsr/bermsr.ann" ] || python -m pipeline.nisar_to_legacy \
    --h5 https://uavsar.jpl.nasa.gov/DOWNLOAD_FILES/LBand-RSLC/bermsR_09037_22031_000_220814_L090_CX_RSLC_02.h5 \
    --out-dir "$RAW_DIR/bermsr" --scene-id bermsr \
    --row-start 40000 --row-size 4096 --col-start 2500 --col-size 4096

echo "=== downloading 5 new flight lines (~205GB total, this is the slow part -- run under tmux/nohup) ==="

declare -A FILES=(
  [bermsR_27037]="https://uavsar.jpl.nasa.gov/DOWNLOAD_FILES/LBand-RSLC/bermsR_27037_22031_001_220814_L090_CX_RSLC_02.h5"
  [bermsA_09041]="https://uavsar.jpl.nasa.gov/DOWNLOAD_FILES/LBand-RSLC/bermsA_09041_22031_008_220814_L090_CX_RSLC_02.h5"
  [bermsA_27041]="https://uavsar.jpl.nasa.gov/DOWNLOAD_FILES/LBand-RSLC/bermsA_27041_22031_009_220814_L090_CX_RSLC_02.h5"
  [bermsB_09043]="https://uavsar.jpl.nasa.gov/DOWNLOAD_FILES/LBand-RSLC/bermsB_09043_22031_012_220814_L090_CX_RSLC_02.h5"
  [bermsB_27043]="https://uavsar.jpl.nasa.gov/DOWNLOAD_FILES/LBand-RSLC/bermsB_27043_22031_013_220814_L090_CX_RSLC_02.h5"
)

for name in "${!FILES[@]}"; do
    echo "--- $name ---"
    wget -c -O "$H5_CACHE/${name}.h5" "${FILES[$name]}"
done

echo "=== cutting 25 non-overlapping 4096x4096 crops from each (125 new train scenes) ==="

python -m pipeline.batch_pull_crops --h5 "$H5_CACHE/bermsR_27037.h5" --prefix bermsR  --crop-size 4096 --max-crops 25
python -m pipeline.batch_pull_crops --h5 "$H5_CACHE/bermsA_09041.h5" --prefix bermsA1 --crop-size 4096 --max-crops 25
python -m pipeline.batch_pull_crops --h5 "$H5_CACHE/bermsA_27041.h5" --prefix bermsA2 --crop-size 4096 --max-crops 25
python -m pipeline.batch_pull_crops --h5 "$H5_CACHE/bermsB_09043.h5" --prefix bermsB1 --crop-size 4096 --max-crops 25
python -m pipeline.batch_pull_crops --h5 "$H5_CACHE/bermsB_27043.h5" --prefix bermsB2 --crop-size 4096 --max-crops 25

echo ""
echo "done. Next: python -m pipeline.prepare_training_data"
echo "(SCENES['train'] auto-discovers these via AUTO_TRAIN_PREFIXES, no manual edit needed)"
echo ""
echo "optional cleanup once tiling has succeeded, NOT run automatically -- reclaims ~205GB:"
echo "  rm -rf $H5_CACHE"
