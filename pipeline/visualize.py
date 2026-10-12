"""Pauli RGB rendering for visual QC, matching the guide's repeated
"check visually: a Pauli RGB should look like a recognizable scene" advice
(Steps 4.5, 8.4, 9.6, 10.5)."""
import numpy as np


def pauli_powers(C):
    """|HH+VV|^2 -> R (double-bounce+surface), |2*HV|^2 -> G (volume),
    |HH-VV|^2 -> B (double-bounce), derived from the covariance matrix in
    the lexicographic basis k=[HH, sqrt(2)*HV_sym, VV]."""
    c11 = C[..., 0, 0].real
    c22 = C[..., 1, 1].real
    c33 = C[..., 2, 2].real
    c13 = C[..., 0, 2]
    surface = c11 + c33 + 2 * c13.real       # |HH+VV|^2
    volume = 2 * c22                          # |2*HV|^2
    double_bounce = c11 + c33 - 2 * c13.real  # |HH-VV|^2
    return double_bounce, volume, surface


def pauli_rgb(C, percentile=98, gamma=0.5, mask=None):
    """mask (bool, same H,W as C): pixels outside it are excluded from the
    percentile color-stretch and rendered black. Without it, a handful of
    edge/no-data pixels (partial multilook windows at the scene boundary)
    can dominate the stretch and wash out or discolor the whole image."""
    r, g, b = pauli_powers(C)
    valid = np.isfinite(r) & np.isfinite(g) & np.isfinite(b)
    if mask is not None:
        valid &= mask
    out = np.zeros(C.shape[:2] + (3,), dtype=np.uint8)
    for i, ch in enumerate((r, g, b)):
        ch = np.clip(ch, 0, None) ** gamma
        sample = ch[valid]
        vmax = np.percentile(sample, percentile) if sample.size else 1.0
        vmax = vmax if vmax > 0 else 1.0
        scaled = np.clip(ch / vmax, 0, 1)
        scaled = np.nan_to_num(scaled)
        scaled = np.where(valid, scaled, 0.0)
        out[..., i] = (scaled * 255).astype(np.uint8)
    return out
