"""Evaluate the trained NBDNet on the held-out test scene, per look-count L.

For each L, compares (a) raw noisy input vs clean reference [baseline]
and (b) model output vs clean reference [despeckled], using PSNR, SSIM,
RMSE and CC computed on the reconstructed total-power ("span") image in
dB -- the standard single-channel SAR representative image, same idea as
the paper's own Table 1 evaluation. Answers: how effective is the
pipeline's Wishart-noise route, and does effectiveness hold up as L
(noise severity) changes.
"""
import json
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from . import model as _model  # noqa: F401 -- registers custom layers before load_model
from .pol_nbdnet import repr_to_cov
from .visualize import pauli_rgb

DATA_DIR = "pipeline/data/experiment"
RESULTS_DIR = "pipeline/results"
TEST_L_VALUES = (1, 2, 4, 8)


def span_db(repr_arr, db_min, db_max):
    """Total power (trace of C) in dB, reconstructed from the 9-channel
    representation -- the representative single-channel image metrics are
    computed on."""
    C = repr_to_cov(repr_arr, db_min=db_min, db_max=db_max)
    span = np.diagonal(C, axis1=-2, axis2=-1).real.sum(axis=-1)
    return 10 * np.log10(np.clip(span, 1e-12, None))


def compute_metrics(clean_db, other_db, data_range):
    """data_range must be a fixed value shared across the whole batch, not
    computed per-tile -- a single near-flat tile (zero local variance,
    common in homogeneous regions like calm water) would otherwise divide
    by ~0 and produce -inf/nan that poisons the batch average."""
    psnr = peak_signal_noise_ratio(clean_db, other_db, data_range=data_range)
    ssim = structural_similarity(clean_db, other_db, data_range=data_range)
    rmse = float(np.sqrt(np.mean((clean_db - other_db) ** 2)))
    if clean_db.std() < 1e-8 or other_db.std() < 1e-8:
        cc = float("nan")
    else:
        cc = float(np.corrcoef(clean_db.ravel(), other_db.ravel())[0, 1])
    return {"psnr": float(psnr), "ssim": float(ssim), "rmse": rmse, "cc": cc}


def main():
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    db_min, db_max = meta["db_min"], meta["db_max"]

    model = keras.models.load_model(os.path.join(RESULTS_DIR, "nbdnet_model.keras"))

    results = {}
    example_tiles = {}
    for L in TEST_L_VALUES:
        d = np.load(os.path.join(DATA_DIR, f"test_L{L}.npz"))
        noisy, clean = d["inputs"], d["targets"]

        denoised = model.predict(noisy, batch_size=64, verbose=0)

        clean_db = span_db(clean, db_min, db_max)
        noisy_db = span_db(noisy, db_min, db_max)
        denoised_db = span_db(denoised, db_min, db_max)

        data_range = float(clean_db.max() - clean_db.min())

        n = clean_db.shape[0]
        noisy_metrics = [compute_metrics(clean_db[i], noisy_db[i], data_range) for i in range(n)]
        denoised_metrics = [compute_metrics(clean_db[i], denoised_db[i], data_range) for i in range(n)]

        def avg(ms, key):
            return float(np.nanmean([m[key] for m in ms]))

        results[f"L={L}"] = {
            "n_tiles": n,
            "noisy_vs_clean": {k: avg(noisy_metrics, k) for k in ("psnr", "ssim", "rmse", "cc")},
            "denoised_vs_clean": {k: avg(denoised_metrics, k) for k in ("psnr", "ssim", "rmse", "cc")},
        }
        improvement = results[f"L={L}"]["denoised_vs_clean"]["psnr"] - results[f"L={L}"]["noisy_vs_clean"]["psnr"]
        results[f"L={L}"]["psnr_improvement_db"] = improvement

        print(f"L={L}: noisy PSNR={results[f'L={L}']['noisy_vs_clean']['psnr']:.2f}dB  "
              f"denoised PSNR={results[f'L={L}']['denoised_vs_clean']['psnr']:.2f}dB  "
              f"(+{improvement:.2f}dB)  SSIM {results[f'L={L}']['noisy_vs_clean']['ssim']:.3f} -> "
              f"{results[f'L={L}']['denoised_vs_clean']['ssim']:.3f}")

        mid = n // 2
        example_tiles[L] = {
            "clean": repr_to_cov(clean[mid], db_min=db_min, db_max=db_max),
            "noisy": repr_to_cov(noisy[mid], db_min=db_min, db_max=db_max),
            "denoised": repr_to_cov(denoised[mid], db_min=db_min, db_max=db_max),
        }

    with open(os.path.join(RESULTS_DIR, "evaluation.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {RESULTS_DIR}/evaluation.json")

    make_figure(example_tiles, os.path.join(RESULTS_DIR, "evaluation_examples.png"))
    print(f"wrote {RESULTS_DIR}/evaluation_examples.png")


def make_figure(example_tiles, out_path):
    Ls = sorted(example_tiles.keys())
    fig, axes = plt.subplots(len(Ls), 3, figsize=(9, 3 * len(Ls)))
    if len(Ls) == 1:
        axes = axes[None, :]
    for row, L in enumerate(Ls):
        t = example_tiles[L]
        axes[row, 0].imshow(pauli_rgb(t["clean"]))
        axes[row, 1].imshow(pauli_rgb(t["noisy"]))
        axes[row, 2].imshow(pauli_rgb(t["denoised"]))
        axes[row, 0].set_ylabel(f"L={L}", fontsize=11)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    for col, title in enumerate(["Clean (C_ref)", "Noisy (Wishart input)", "NBDNet output"]):
        axes[0, col].set_title(title, fontsize=11)
    fig.suptitle("NBDNet despeckling, test scene, by look-count L", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
