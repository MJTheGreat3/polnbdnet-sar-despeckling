# PolNBDNet: PolSAR Despeckling

Project-elective work on polarimetric SAR (PolSAR) speckle reduction using **NBDNet**, a non-blind despeckling network (noise estimation + denoising subnetworks with Squeeze-Excitation and CBAM attention), implemented in Keras/TensorFlow. This is an ongoing project — the current state is a working baseline (real-data pipeline, trained model, first evaluation), not a finished result. See `report.md` / `report.pdf` for the full write-up, including an honest list of current limitations and next steps.

## What this does

1. Pulls real UAVSAR quad-pol SLC data (sourced from JPL's NISAR-format HDF5 RSLC products, since the public ASF archive only hosts multilooked/ground-projected products, not raw SLC).
2. Builds the 3x3 polarimetric covariance matrix per pixel, and a large-window-multilooked "clean" reference (`C_ref`) standing in for the unobservable true signal.
3. Simulates speckle at controlled look-counts (L = 1, 2, 4, 8) using the complex Wishart distribution — the statistically correct way to inject correlated, phase-preserving noise into polarimetric data.
4. Converts noisy/clean covariance pairs into a 9-channel representation and tiles them (64x64) for training.
5. Trains NBDNet to recover the clean covariance from the noisy one.
6. Evaluates recovery quality (PSNR, SSIM, RMSE, CC) across look-counts on a held-out scene.

## Status

Baseline results (see `report.md` for details and diagrams):

| L (looks) | Noisy PSNR | Denoised PSNR | Gain |
|---|---|---|---|
| 1 (severe) | 19.03 dB | 25.59 dB | +6.56 dB |
| 2 | 21.78 dB | 27.62 dB | +5.84 dB |
| 4 | 24.58 dB | 28.63 dB | +4.05 dB |
| 8 (mild) | 27.41 dB | 28.88 dB | +1.47 dB |

Known limitation: the model oversmooths and loses fine texture (low correlation coefficient despite strong PSNR/SSIM gains), matching a weakness documented in the original NBDNet paper itself. Planned next steps (report §8): diagnose a possible train/test normalization domain shift, add classical-filter baselines (boxcar, Lee) for a fair comparison, and extend to full-scene stitched reconstruction.

## Repo layout

```
pipeline/
  ann_parser.py, uavsar_io.py, mlc_io.py    -- read UAVSAR/NISAR SLC formats
  nisar_to_legacy.py                        -- convert NISAR HDF5 crop -> legacy .ann+.slc
  masking.py, covariance.py, clean_reference.py, radiometry.py
  pol_nbdnet.py                             -- 9-channel representation <-> covariance conversion
  subaperture.py, tiling.py, splitting.py   -- preprocessing steps per the guide
  prepare_training_data.py                  -- builds Wishart-noisy training/val/test tiles
  model.py                                  -- NBDNet architecture (Keras)
  train_nbdnet.py                           -- training CLI
  evaluate_nbdnet.py                        -- evaluation across look-counts
  make_before_after.py                      -- before/after figure generation
  visualize.py                              -- Pauli RGB rendering, plotting helpers
  DATA_SOURCE_AND_INSTRUCTIONS.md           -- where the raw data comes from and how to pull it
  results/                                  -- trained metrics, example figures (model weights gitignored)
report.md / report.pdf                      -- full write-up with diagrams and results
NBDNetPaper.pdf, PreProc_Guide.pdf, ReviewPaper.pdf  -- reference papers
```

`pipeline/data/` (raw SLC crops + generated tiles) and the trained `.keras` weights are gitignored — they're large and regeneratable from the scripts above.

## Setup

```bash
conda create -n polnbdnet python=3.11
conda activate polnbdnet
pip install -r pipeline/requirements.txt
```

On Apple Silicon, also install `tensorflow-metal` for GPU acceleration (requires Python 3.11 + TensorFlow 2.16.2 for a working Metal plugin as of this writing).

## Usage

```bash
# 1. pull and convert raw SLC data -- see pipeline/DATA_SOURCE_AND_INSTRUCTIONS.md first
python -m pipeline.nisar_to_legacy ...

# 2. build training/val/test tiles (Wishart-noisy + clean pairs)
python -m pipeline.prepare_training_data

# 3. train NBDNet
python -m pipeline.train_nbdnet --epochs 20 --batch-size 32

# 4. evaluate across look-counts L = 1, 2, 4, 8
python -m pipeline.evaluate_nbdnet

# 5. generate before/after example figures
python -m pipeline.make_before_after
```
