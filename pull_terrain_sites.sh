#!/usr/bin/env bash
# Pulls one 4096x4096 crop from each of the 6 terrain-diversity sites
# selected in report.md Week 3 S3.3 (report.md Open item #6: val/test were
# each a single scene -- this is step 1 of fixing that). URLs are the
# verified HH .ann paths from that table, relative to downloaduav2.jpl.nasa.gov/
# (confirmed real via curl -I in that session, not guessed).
#
# These sites ship the native legacy flat-binary format directly (.ann +
# 4x .slc), unlike the NISAR-HDF5 source used for howlnd/bermsr/monda/berms*
# -- pulled here with pull_legacy_crop.py (HTTP Range reads), not
# nisar_to_legacy.py/batch_pull_crops.py.
#
# Allocation (first pass -- just enough to make val/test multi-scene):
#   rockmt, cvalle, atchaf, dorado -> train (4 new terrain types)
#   barrow                        -> val   (2nd val scene)
#   stlake                        -> test  (2nd test scene)
# Add more crops per site later (same pull_legacy_crop.py call, different
# --row-start/--col-start) to grow train further.
#
# Run from the repo root with the polnbdnet conda env active.
# After this finishes: add the new scene ids to SCENES in
# prepare_training_data.py (see that file's comment), then
# python -m pipeline.prepare_training_data
set -euo pipefail

BASE="https://downloaduav2.jpl.nasa.gov"
RAW_DIR=pipeline/data/raw
ROW_START=2000   # skip the triangular no-data edge region near row 0
ROW_SIZE=4096
COL_START=0
COL_SIZE=4096

pull() {
    local ann_path="$1" scene_id="$2"
    local out_dir="$RAW_DIR/$scene_id"
    if [ -f "$out_dir/$scene_id.ann" ]; then
        echo "skip $scene_id (already pulled)"
        return
    fi
    echo "--- $scene_id ---"
    python -m pipeline.pull_legacy_crop \
        --ann-url "$BASE/$ann_path" \
        --out-dir "$out_dir" --scene-id "$scene_id" \
        --row-start "$ROW_START" --row-size "$ROW_SIZE" \
        --col-start "$COL_START" --col-size "$COL_SIZE"
}

echo "=== train: 4 new terrain types ==="
pull "Release32/rockmt_14107_01/rockmt_14107_20005_009_200212_L090HH_01_BU.ann" rockmt_r2000_c0
pull "Release33/cvalle_32524_01/cvalle_32524_23007_005_230629_L090HH_01_BC.ann" cvalle_r2000_c0
pull "Release27/atchaf_06309_02/atchaf_06309_21023_001_210327_L090HH_02_BU.ann" atchaf_r2000_c0
pull "Release28/dorado_34017_01/dorado_34017_20002_002_200131_L090HH_01_BU.ann" dorado_r2000_c0

echo "=== val: 2nd val scene ==="
pull "Release34/barrow_15018_01/barrow_15018_17069_003_170621_L090HH_01_BC.ann" barrow_val1

echo "=== test: 2nd test scene ==="
pull "Release31/stlake_27129_01/stlake_27129_20007_004_200213_L090HH_01_BC.ann" stlake_test1

echo ""
echo "done. Next steps:"
echo "1. Add to prepare_training_data.py: AUTO_TRAIN_PREFIXES += [\"rockmt_r\", \"cvalle_r\", \"atchaf_r\", \"dorado_r\"]"
echo "   (train scenes auto-discover via that prefix list, same as bermsA/B)"
echo "2. Add to SCENES in prepare_training_data.py: val -> + \"barrow_val1\", test -> + \"stlake_test1\""
echo "3. python -m pipeline.prepare_training_data"
echo "4. evaluate_nbdnet.py already loops over all test scenes (report.md Open item #6 fix)"
