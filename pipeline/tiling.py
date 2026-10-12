"""Step 10: cut matched (input, target) scenes into pixel-aligned tiles."""
import numpy as np


def tile_pairs(input_repr, target_repr, valid_mask, tile_size=64, stride=32, valid_frac_thresh=0.9):
    """Slide a tile_size x tile_size window over pixel-aligned (H, W, 9)
    input/target representations, keeping only tiles whose valid-pixel
    fraction meets the threshold.

    Yields dict: input, target, origin (row, col), valid_frac.
    """
    H, W = valid_mask.shape
    for r in range(0, H - tile_size + 1, stride):
        for c in range(0, W - tile_size + 1, stride):
            m = valid_mask[r : r + tile_size, c : c + tile_size]
            frac = m.mean()
            if frac < valid_frac_thresh:
                continue
            yield {
                "input": input_repr[r : r + tile_size, c : c + tile_size, :],
                "target": target_repr[r : r + tile_size, c : c + tile_size, :],
                "origin": (r, c),
                "valid_frac": float(frac),
            }


def tiles_to_arrays(tiles):
    tiles = list(tiles)
    inputs = np.stack([t["input"] for t in tiles], axis=0)
    targets = np.stack([t["target"] for t in tiles], axis=0)
    origins = np.array([t["origin"] for t in tiles])
    return inputs, targets, origins
