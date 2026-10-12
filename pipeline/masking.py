"""Step 3: mask invalid pixels."""
import numpy as np
from scipy.ndimage import binary_erosion


def build_valid_mask(hh, hv, vh, vv, nodata=0.0, edge_margin=0, erode_px=0):
    channels = (hh, hv, vh, vv)
    mask = np.ones(hh.shape, dtype=bool)
    for ch in channels:
        ch = np.asarray(ch)
        mask &= np.isfinite(ch.real) & np.isfinite(ch.imag)
        mask &= ~np.isclose(ch, nodata)

    if edge_margin > 0:
        mask[:edge_margin, :] = False
        mask[-edge_margin:, :] = False
        mask[:, :edge_margin] = False
        mask[:, -edge_margin:] = False

    if erode_px > 0:
        mask = binary_erosion(mask, iterations=erode_px, border_value=0)

    return mask


def build_valid_mask_from_covariance(C, nodata=0.0, edge_margin=0, erode_px=0):
    """Same idea as build_valid_mask, starting from an already-formed C
    (e.g. MLC data) instead of raw SLC channels."""
    diag = np.diagonal(C, axis1=-2, axis2=-1)
    mask = np.ones(C.shape[:2], dtype=bool)
    for i in range(3):
        ch = diag[..., i]
        mask &= np.isfinite(ch.real) & np.isfinite(ch.imag)
        mask &= ~np.isclose(ch.real, nodata)

    if edge_margin > 0:
        mask[:edge_margin, :] = False
        mask[-edge_margin:, :] = False
        mask[:, :edge_margin] = False
        mask[:, -edge_margin:] = False

    if erode_px > 0:
        mask = binary_erosion(mask, iterations=erode_px, border_value=0)

    return mask
