"""Full-scene stitched reconstruction (report.md Next Steps #2).

Training/eval only ever score non-overlapping TILE_SIZE=64/STRIDE=64 tiles
in isolation -- there's no mosaic output, so no way to tell whether the
model's per-tile boundary behaves differently from its interior (grid-line
artifacts at tile seams). This script runs NBDNet on a full scene with
OVERLAPPING tiles (stride 48, 16px overlap) and keeps only each tile's
center crop when stitching, discarding the 8px border ring -- the standard
fix for seam artifacts in tiled CNN inference, since a tile's own edges are
exactly where it has the least receptive-field context.
"""
import argparse
import json
import os

import keras
import matplotlib.pyplot as plt
import numpy as np

from . import model as _model  # noqa: F401 -- registers custom layers
from .evaluate_nbdnet import span_db
from .pol_nbdnet import cov_to_repr, simulate_polsar_speckle
from .prepare_training_data import build_scene_clean_ref
from .visualize import pauli_rgb

DATA_DIR = "pipeline/data/experiment"
RESULTS_DIR = "pipeline/results"
TILE_SIZE = 64
STRIDE = 48  # 16px overlap -> BORDER=8px trimmed from each side per tile
BORDER = (TILE_SIZE - STRIDE) // 2


def tile_origins_full_coverage(H, W, tile_size=TILE_SIZE, stride=STRIDE):
    """Like prepare_training_data.enumerate_tile_origins, but the last
    row/col is snapped flush to the bottom/right edge instead of dropped --
    stitching needs every pixel covered by some tile, unlike training
    tiling, which can discard a leftover partial strip."""
    def positions(size):
        if size <= tile_size:
            return [0]
        pos = list(range(0, size - tile_size + 1, stride))
        if pos[-1] != size - tile_size:
            pos.append(size - tile_size)
        return pos

    rows, cols = positions(H), positions(W)
    return [(r, c) for r in rows for c in cols]


def stitch_predict(model, repr_full, tile_size=TILE_SIZE, stride=STRIDE, border=BORDER, batch_size=64):
    """Run model on overlapping tiles, keep each tile's center crop (full
    crop at scene edges, where there's no neighboring tile to hand off to)."""
    H, W, ch = repr_full.shape
    origins = tile_origins_full_coverage(H, W, tile_size, stride)

    tiles = np.stack([repr_full[r:r + tile_size, c:c + tile_size] for (r, c) in origins])
    preds = model.predict(np.nan_to_num(tiles), batch_size=batch_size, verbose=0)

    out = np.zeros((H, W, ch), dtype=np.float32)
    covered = np.zeros((H, W), dtype=bool)
    for (r, c), pred in zip(origins, preds):
        top = 0 if r == 0 else border
        bottom = 0 if r + tile_size >= H else border
        left = 0 if c == 0 else border
        right = 0 if c + tile_size >= W else border
        out[r + top:r + tile_size - bottom, c + left:c + tile_size - right] = \
            pred[top:tile_size - bottom, left:tile_size - right]
        covered[r + top:r + tile_size - bottom, c + left:c + tile_size - right] = True

    assert covered.all(), "stitching left gaps -- tile_origins_full_coverage bug"
    return out


def seam_score(image_db, stride=STRIDE):
    """Ratio of mean gradient energy AT tile-stitch boundaries (multiples of
    `stride`, where two tiles' outputs meet) vs everywhere else. ~1.0 means
    no detectable seam; notably >1 means a visible grid-line artifact."""
    H, W = image_db.shape
    row_grad = np.abs(np.diff(image_db, axis=0))  # (H-1, W)
    col_grad = np.abs(np.diff(image_db, axis=1))  # (H, W-1)

    boundary_rows = [r for r in range(stride, H - 1, stride)]
    boundary_cols = [c for c in range(stride, W - 1, stride)]
    interior_rows = sorted(set(range(H - 1)) - set(boundary_rows))
    interior_cols = sorted(set(range(W - 1)) - set(boundary_cols))

    boundary_energy = np.concatenate([row_grad[boundary_rows].ravel(), col_grad[:, boundary_cols].ravel()])
    interior_energy = np.concatenate([row_grad[interior_rows].ravel(), col_grad[:, interior_cols].ravel()])
    return float(boundary_energy.mean() / max(interior_energy.mean(), 1e-8))


def make_figure(clean_cov, noisy_cov, denoised_cov, valid_mask, stride, out_path):
    """Full-scene Pauli RGB triptych, plus a zoomed-in panel around the
    first interior tile boundary with reference lines at the stitch seam --
    the fastest way to visually spot a grid-line artifact if one exists."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for col, (name, C) in enumerate([("Clean (C_ref)", clean_cov), ("Noisy (Wishart input)", noisy_cov),
                                      ("NBDNet stitched output", denoised_cov)]):
        axes[0, col].imshow(pauli_rgb(C, mask=valid_mask))
        axes[0, col].set_title(name, fontsize=10)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])

    H, W = valid_mask.shape
    zr0, zr1 = max(stride - 64, 0), min(stride + 64, H)
    zc0, zc1 = max(stride - 64, 0), min(stride + 64, W)
    for col, (name, C) in enumerate([("Clean", clean_cov), ("Noisy", noisy_cov), ("Denoised (zoom)", denoised_cov)]):
        crop = C[zr0:zr1, zc0:zc1]
        axes[1, col].imshow(pauli_rgb(crop, mask=valid_mask[zr0:zr1, zc0:zc1]))
        axes[1, col].axhline(stride - zr0, color="red", lw=0.8, alpha=0.7)
        axes[1, col].axvline(stride - zc0, color="red", lw=0.8, alpha=0.7)
        axes[1, col].set_title(f"{name}, zoom @ first seam", fontsize=9)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default=None, help="scene id (default: the test scene from meta.json)")
    p.add_argument("--L", type=int, default=1, choices=(1, 2, 4, 8))
    p.add_argument("--crop-size", type=int, default=1024,
                    help="test on a crop_size x crop_size top-left crop instead of the full scene "
                         "(default 1024 -- plenty of stride=48 tile boundaries to check for seams, "
                         "far cheaper than the full 4096x4096 scene). Ignored if --full is passed.")
    p.add_argument("--full", action="store_true",
                    help="run on the full scene instead of a crop. WARNING: main() keeps several full-scene "
                         "complex128 covariance arrays alive at once (C_ref, C_noisy, plus three more "
                         "reconstructed via repr_to_cov for the figure) -- on a 4096x4096 scene that's "
                         "~40GB+ of simultaneous arrays, enough to OOM-kill the process (and in one observed "
                         "case, the whole tmux session) even with tens of GB nominally free, since the box's "
                         "free RAM wasn't the full peak the script needs at once. Only pass this if you've "
                         "confirmed headroom for that burst.")
    args = p.parse_args()

    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    scene_id = args.scene or meta["scenes"]["test"][0]
    db_min, db_max = meta["per_scene_norm"][scene_id]

    print(f"building full-scene C_ref for {scene_id} ...")
    C_ref, valid_mask, report = build_scene_clean_ref(scene_id)
    H, W = C_ref.shape[:2]
    print(f"  {H}x{W}, hermitian/psd={report['hermitian']}/{report['psd']}")

    if not args.full:
        c = min(args.crop_size, H, W)
        C_ref, valid_mask = C_ref[:c, :c], valid_mask[:c, :c]
        H, W = c, c
        print(f"  cropped to {H}x{W} top-left (pass --full for the whole scene; see --help for why that risks OOM)")

    rng = np.random.default_rng(0)
    print(f"simulating L={args.L} speckle over the full scene ...")
    C_noisy = simulate_polsar_speckle(C_ref, L=args.L, rng=rng)

    clean_repr = cov_to_repr(C_ref, db_min=db_min, db_max=db_max).astype("float32")
    noisy_repr = cov_to_repr(C_noisy, db_min=db_min, db_max=db_max).astype("float32")

    model = keras.models.load_model(os.path.join(RESULTS_DIR, "nbdnet_model.keras"))

    print(f"stitched inference (tile={TILE_SIZE}, stride={STRIDE}, border={BORDER}px trimmed per tile)...")
    denoised_repr = stitch_predict(model, noisy_repr)

    clean_db = span_db(clean_repr, db_min, db_max)
    noisy_db = span_db(noisy_repr, db_min, db_max)
    denoised_db = span_db(denoised_repr, db_min, db_max)

    seam = seam_score(denoised_db[valid_mask.any(axis=1)][:, valid_mask.any(axis=0)], stride=STRIDE)
    psnr_gain = None
    data_range = float(clean_db[valid_mask].max() - clean_db[valid_mask].min())
    rmse_noisy = float(np.sqrt(np.mean((clean_db[valid_mask] - noisy_db[valid_mask]) ** 2)))
    rmse_denoised = float(np.sqrt(np.mean((clean_db[valid_mask] - denoised_db[valid_mask]) ** 2)))

    print(f"seam score (boundary-row/col gradient energy / interior gradient energy): {seam:.3f} "
          f"(~1.0 = no detectable seam)")
    print(f"full-scene RMSE(dB): noisy={rmse_noisy:.3f} denoised={rmse_denoised:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {
        "scene": scene_id, "L": args.L, "shape": [int(H), int(W)],
        "tile_size": TILE_SIZE, "stride": STRIDE, "border": BORDER,
        "seam_score": seam, "rmse_db_noisy": rmse_noisy, "rmse_db_denoised": rmse_denoised,
        "data_range_db": data_range,
    }
    out_json = os.path.join(RESULTS_DIR, f"stitch_{scene_id}_L{args.L}.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_json}")

    out_png = os.path.join(RESULTS_DIR, f"stitch_{scene_id}_L{args.L}.png")
    from .pol_nbdnet import repr_to_cov
    make_figure(
        repr_to_cov(clean_repr, db_min=db_min, db_max=db_max),
        repr_to_cov(noisy_repr, db_min=db_min, db_max=db_max),
        repr_to_cov(denoised_repr, db_min=db_min, db_max=db_max),
        valid_mask, STRIDE, out_png,
    )
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
