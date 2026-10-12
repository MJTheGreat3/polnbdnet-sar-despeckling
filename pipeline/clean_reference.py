"""Step 8: build the heavily-multilooked 'clean' reference C_ref.

Skip this module entirely if using the real-independent-looks route
(Step 12 / pipeline.subaperture) instead.
"""
import numpy as np

from .covariance import build_covariance, multilook_covariance


def build_clean_reference(hh, hv, vh, vv, mask=None, window=15, min_valid_frac=0.9):
    """Large-window multilook average as a stand-in for speckle-free C.

    Caller is responsible for choosing spatially homogeneous patches
    (large fields, stable forest, calm water) -- averaging over a real
    edge teaches the network to blur edges. `keep` flags pixels below
    min_valid_frac (too many masked pixels in their window -- guide Step
    8.3 "discard"); it's advisory, not baked into C_ref as NaN, since
    tiling (Step 10.3) tolerates up to ~10% low-confidence pixels per
    tile and a hard NaN would poison the whole tile instead.
    """
    C_ref, valid_frac = build_covariance(hh, hv, vh, vv, window=window, mask=mask)
    keep = valid_frac >= min_valid_frac
    return C_ref, keep


def build_clean_reference_from_covariance(C_in, mask=None, window=15, min_valid_frac=0.9):
    """Same idea, starting from an already-formed C (e.g. MLC data) instead
    of raw SLC channels -- further multilooks C_in with a larger window.
    """
    C_ref, valid_frac = multilook_covariance(C_in, window=window, mask=mask)
    keep = valid_frac >= min_valid_frac
    return C_ref, keep
