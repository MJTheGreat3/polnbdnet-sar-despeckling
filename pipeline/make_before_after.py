"""A few varied before/after examples from the test set (noisy input vs
NBDNet output vs clean ground truth), at L=1 -- the most illustrative
noise level since that's where despeckling matters most."""
import json
import os

import keras
import matplotlib.pyplot as plt
import numpy as np

from . import model as _model  # noqa: F401 -- registers custom layers
from .pol_nbdnet import repr_to_cov
from .visualize import pauli_rgb

DATA_DIR = "pipeline/data/experiment"
RESULTS_DIR = "pipeline/results"
L = 1
N_EXAMPLES = 5


def main():
    with open(os.path.join(DATA_DIR, "meta.json")) as f:
        meta = json.load(f)
    test_scenes = meta["scenes"]["test"]
    db_min, db_max = meta["per_scene_norm"][test_scenes[0]]

    model = keras.models.load_model(os.path.join(RESULTS_DIR, "nbdnet_model.keras"))

    d = np.load(os.path.join(DATA_DIR, f"test_L{L}.npz"))
    noisy, clean = d["inputs"], d["targets"]
    n = noisy.shape[0]

    rng = np.random.default_rng(7)
    idx = rng.choice(n, size=N_EXAMPLES, replace=False)
    idx.sort()

    denoised = model.predict(noisy[idx], batch_size=N_EXAMPLES, verbose=0)

    fig, axes = plt.subplots(N_EXAMPLES, 3, figsize=(9, 3 * N_EXAMPLES))
    for row, i in enumerate(idx):
        c_noisy = repr_to_cov(noisy[i], db_min=db_min, db_max=db_max)
        c_denoised = repr_to_cov(denoised[row], db_min=db_min, db_max=db_max)
        c_clean = repr_to_cov(clean[i], db_min=db_min, db_max=db_max)

        axes[row, 0].imshow(pauli_rgb(c_noisy))
        axes[row, 1].imshow(pauli_rgb(c_denoised))
        axes[row, 2].imshow(pauli_rgb(c_clean))
        axes[row, 0].set_ylabel(f"tile #{i}", fontsize=10)
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    for col, title in enumerate(["Before (noisy input, L=1)", "After (NBDNet output)", "Ground truth (C_ref)"]):
        axes[0, col].set_title(title, fontsize=11)

    fig.suptitle(f"Before / after examples, test scene (monda), L={L}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(RESULTS_DIR, "before_after_examples.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
