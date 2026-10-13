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
from scipy.ndimage import uniform_filter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from . import model as _model  # noqa: F401 -- registers custom layers before load_model
from .pol_nbdnet import repr_to_cov
from .visualize import pauli_rgb

DATA_DIR = "pipeline/data/experiment"
RESULTS_DIR = "pipeline/results"
TEST_L_VALUES = (1, 2, 4, 8)
BASELINE_WINDOW = 7


def boxcar_filter_cov(C, size=BASELINE_WINDOW):
    """Classical baseline 1: plain spatial moving-average of every
    covariance-matrix element over a size x size window, independently
    per tile (C has shape (n, H, W, 3, 3)). Caveat (report.md Next Steps
    #3): this is close to how C_ref itself was generated (also a
    large-window multilook average), so it has a structural home-field
    advantage here -- Lee below is the fairer baseline to beat."""
    out = np.empty_like(C)
    fsize = (1, size, size)
    for i in range(3):
        for j in range(3):
            out[..., i, j] = (uniform_filter(C[..., i, j].real, size=fsize)
                               + 1j * uniform_filter(C[..., i, j].imag, size=fsize))
    return out


def lee_filter_cov(C, L, size=BASELINE_WINDOW):
    """Classical baseline 2: Lee filter generalized to a covariance-matrix
    image -- a single per-pixel adaptive weight b in [0,1], derived from
    the local coefficient of variation of the total-power (span) image,
    applied uniformly to every matrix element (edge/texture-preserving,
    unlike boxcar, because b -> 1 i.e. "keep the noisy value" in
    high-variance/textured regions and b -> 0 i.e. "use the local mean"
    in homogeneous regions)."""
    span = np.diagonal(C, axis1=-2, axis2=-1).real.sum(axis=-1)
    fsize = (1, size, size)
    mean_span = uniform_filter(span, size=fsize)
    mean_sq_span = uniform_filter(span ** 2, size=fsize)
    var_span = np.clip(mean_sq_span - mean_span ** 2, 0, None)

    ci2 = var_span / np.clip(mean_span ** 2, 1e-12, None)
    cu2 = 1.0 / L
    b = np.clip(1.0 - cu2 / np.clip(ci2, 1e-12, None), 0.0, 1.0)

    mean_C = boxcar_filter_cov(C, size=size)
    return mean_C + b[..., None, None] * (C - mean_C)


def cov_span_db(C):
    span = np.diagonal(C, axis1=-2, axis2=-1).real.sum(axis=-1)
    return 10 * np.log10(np.clip(span, 1e-12, None))


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
    test_scenes = meta["scenes"]["test"]
    if len(test_scenes) != 1:
        raise NotImplementedError("evaluate_nbdnet assumes a single test scene; "
                                   "per-tile scene tracking would be needed for more than one")
    db_min, db_max = meta["per_scene_norm"][test_scenes[0]]

    model = keras.models.load_model(os.path.join(RESULTS_DIR, "nbdnet_model.keras"))

    results = {}
    example_tiles = {}
    for L in TEST_L_VALUES:
        d = np.load(os.path.join(DATA_DIR, "test", f"{test_scenes[0]}_L{L}.npz"))
        noisy, clean = d["inputs"], d["targets"]

        denoised = model.predict(noisy, batch_size=64, verbose=0)

        noisy_cov = repr_to_cov(noisy, db_min=db_min, db_max=db_max)
        boxcar_db = cov_span_db(boxcar_filter_cov(noisy_cov))
        lee_db = cov_span_db(lee_filter_cov(noisy_cov, L=L))

        clean_db = span_db(clean, db_min, db_max)
        noisy_db = span_db(noisy, db_min, db_max)
        denoised_db = span_db(denoised, db_min, db_max)

        data_range = float(clean_db.max() - clean_db.min())

        n = clean_db.shape[0]

        def avg(ms, key):
            return float(np.nanmean([m[key] for m in ms]))

        def metrics_vs_clean(other_db):
            ms = [compute_metrics(clean_db[i], other_db[i], data_range) for i in range(n)]
            return {k: avg(ms, k) for k in ("psnr", "ssim", "rmse", "cc")}

        results[f"L={L}"] = {
            "n_tiles": n,
            "noisy_vs_clean": metrics_vs_clean(noisy_db),
            "boxcar_vs_clean": metrics_vs_clean(boxcar_db),
            "lee_vs_clean": metrics_vs_clean(lee_db),
            "denoised_vs_clean": metrics_vs_clean(denoised_db),
        }
        improvement = results[f"L={L}"]["denoised_vs_clean"]["psnr"] - results[f"L={L}"]["noisy_vs_clean"]["psnr"]
        results[f"L={L}"]["psnr_improvement_db"] = improvement

        r = results[f"L={L}"]
        print(f"L={L}: noisy {r['noisy_vs_clean']['psnr']:.2f}dB  "
              f"boxcar {r['boxcar_vs_clean']['psnr']:.2f}dB  "
              f"lee {r['lee_vs_clean']['psnr']:.2f}dB  "
              f"NBDNet {r['denoised_vs_clean']['psnr']:.2f}dB  (+{improvement:.2f}dB vs noisy)  "
              f"SSIM noisy {r['noisy_vs_clean']['ssim']:.3f} -> NBDNet {r['denoised_vs_clean']['ssim']:.3f}")

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
