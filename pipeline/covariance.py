"""Step 4: multilooked polarimetric covariance matrix C (3x3, complex)."""
import numpy as np
from scipy.ndimage import uniform_filter


def symmetrize_hv(hv, vh):
    return (np.asarray(hv) + np.asarray(vh)) / 2.0


def multilook_covariance(C_in, window=5, mask=None):
    """Windowed average of an existing (per-pixel or already-multilooked)
    covariance array, respecting a validity mask. Averages the complex
    matrix elements directly (never dB/coherence). Used both to build C
    from raw per-pixel outer products (build_covariance) and to further
    multilook an already-formed C, e.g. MLC data, for a clean reference.

    Returns (C, valid_frac).
    """
    C_in = np.asarray(C_in)
    if mask is None:
        mask = np.ones(C_in.shape[:2], dtype=bool)
    weight = mask.astype(np.float64)

    weighted = C_in * weight[..., None, None]
    C = np.zeros_like(weighted)
    for i in range(3):
        for j in range(3):
            C[..., i, j] = uniform_filter(
                weighted[..., i, j].real, size=window, mode="reflect"
            ) + 1j * uniform_filter(weighted[..., i, j].imag, size=window, mode="reflect")

    valid_frac = uniform_filter(weight, size=window, mode="reflect")
    safe = np.where(valid_frac == 0, np.nan, valid_frac)
    with np.errstate(invalid="ignore"):
        C = C / safe[..., None, None]
    return C, valid_frac


def build_covariance(hh, hv, vh, vv, window=5, mask=None):
    """C = windowed average of k k^H, respecting a validity mask.

    Returns (C, valid_frac) where C has shape (H, W, 3, 3) complex and
    valid_frac is the fraction of the window that was valid data,
    per output pixel.
    """
    hh = np.asarray(hh)
    hv = np.asarray(hv)
    vh = np.asarray(vh)
    vv = np.asarray(vv)
    hv_sym = symmetrize_hv(hv, vh)
    k = np.stack([hh, np.sqrt(2) * hv_sym, vv], axis=-1)  # (H, W, 3)
    kk = k[..., :, None] * np.conj(k[..., None, :])  # (H, W, 3, 3)
    return multilook_covariance(kk, window=window, mask=mask)


def is_hermitian(C, atol=1e-4):
    return np.allclose(C, np.conj(np.swapaxes(C, -1, -2)), atol=atol, equal_nan=True)


def check_covariance_validity(C, tol=1e-6):
    """Step 4.5 / Step 13 checks: Hermitian, real+nonneg diagonal, PSD."""
    diag = np.diagonal(C, axis1=-2, axis2=-1)
    report = {
        "diag_real": bool(np.nanmax(np.abs(diag.imag)) < 1e-3) if diag.size else True,
        "diag_nonneg": bool(np.nanmin(diag.real) >= -tol) if diag.size else True,
        "hermitian": is_hermitian(C),
    }
    flat = C.reshape(-1, C.shape[-2], C.shape[-1])
    valid = ~np.any(np.isnan(flat), axis=(-1, -2))
    if valid.any():
        eigvals = np.linalg.eigvalsh(flat[valid])
        report["min_eigenvalue"] = float(np.min(eigvals))
        report["psd"] = bool(np.min(eigvals) >= -tol)
    else:
        report["min_eigenvalue"] = float("nan")
        report["psd"] = True
    return report
