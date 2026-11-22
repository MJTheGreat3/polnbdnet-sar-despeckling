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
    # float32, not float64 -- mask values are exactly 0.0/1.0 either way (no
    # precision lost), but multiplying a complex64 C_in by a float64 weight
    # silently upcasts the whole (H,W,3,3) result to complex128, doubling
    # memory for no reason. Matters at 4096x4096: that's the difference
    # between a ~1.2GB and ~2.4GB `weighted` array (report.md: real-box OOM
    # investigation for prepare_training_data.py).
    weight = mask.astype(np.float32)

    weighted = C_in * weight[..., None, None]
    C = np.zeros_like(weighted)
    for i in range(3):
        for j in range(3):
            C[..., i, j] = uniform_filter(
                weighted[..., i, j].real, size=window, mode="reflect"
            ) + 1j * uniform_filter(weighted[..., i, j].imag, size=window, mode="reflect")
    del weighted

    valid_frac = uniform_filter(weight, size=window, mode="reflect")
    safe = np.where(valid_frac == 0, np.nan, valid_frac)
    with np.errstate(invalid="ignore"):
        C /= safe[..., None, None]  # in-place: no second full-size array alongside the first
    return C, valid_frac


def build_covariance(hh, hv, vh, vv, window=5, mask=None, dtype=None):
    """C = windowed average of k k^H, respecting a validity mask.

    dtype defaults to preserving hh's own dtype (complex128 for raw SLC
    read via uavsar_io, unchanged behavior for every existing caller).
    Pass dtype=np.complex64 for a ~2x memory cut on the dense (H,W,3,3)
    intermediates at the cost of complex64 precision -- fine whenever the
    result ultimately gets cast to float32 anyway (prepare_training_data.py:
    cov_to_repr's 9-channel output is float32, so complex128 precision in
    this intermediate step is wasted upstream of that cast).

    Returns (C, valid_frac) where C has shape (H, W, 3, 3) complex and
    valid_frac is the fraction of the window that was valid data,
    per output pixel.
    """
    hh = np.asarray(hh)
    hv = np.asarray(hv)
    vh = np.asarray(vh)
    vv = np.asarray(vv)
    if dtype is None:
        dtype = hh.dtype
    hv_sym = symmetrize_hv(hv, vh)
    k = np.stack([hh, np.sqrt(2) * hv_sym, vv], axis=-1).astype(dtype)  # (H, W, 3)
    kk = k[..., :, None] * np.conj(k[..., None, :])  # (H, W, 3, 3)
    del k, hv_sym
    return multilook_covariance(kk, window=window, mask=mask)


def is_hermitian(C, atol=1e-4, chunk_size=500_000):
    """Chunked, not one np.allclose(C, conj(swapaxes(C))) over the whole
    array -- that conj()+swapaxes() pair each materialize a full-size copy
    of C on top of whatever's already alive in the caller, which at a
    4096x4096 scene (~16.7M pixels) is a multi-GB burst allocation/free in
    a few milliseconds. Fine on a quiet box; on this one (shared, other
    tenants already keeping the box's memory pressure nonzero) that burst
    was enough to trip systemd-oomd's per-cgroup pressure kill on
    prepare_training_data.py (report.md). Chunking bounds the temp copy to
    chunk_size regardless of scene size -- exact same result (every pixel
    still compared, same tolerance), just not all at once. Early-exits on
    the first non-Hermitian chunk, which also can't change the answer
    (allclose needs ALL elements within tolerance)."""
    C = np.asarray(C)
    flat = C.reshape(-1, C.shape[-2], C.shape[-1])
    n = flat.shape[0]
    for start in range(0, n, chunk_size):
        chunk = flat[start:start + chunk_size]
        if not np.allclose(chunk, np.conj(np.swapaxes(chunk, -1, -2)), atol=atol, equal_nan=True):
            return False
    return True


def check_covariance_validity(C, tol=1e-6, chunk_size=500_000):
    """Step 4.5 / Step 13 checks: Hermitian, real+nonneg diagonal, PSD.

    The eigenvalue check runs in chunks too, for the same reason as
    is_hermitian above: flat[valid] (needed to drop NaN pixels before
    eigvalsh, which NaNs would poison) is a boolean fancy-index, which
    always copies -- at scene scale that's another multi-GB burst. Chunked,
    this checks the exact same pixels and returns the exact same
    min_eigenvalue/psd verdict, just without ever holding more than
    chunk_size pixels' worth of temporaries at once.
    """
    diag = np.diagonal(C, axis1=-2, axis2=-1)
    report = {
        "diag_real": bool(np.nanmax(np.abs(diag.imag)) < 1e-3) if diag.size else True,
        "diag_nonneg": bool(np.nanmin(diag.real) >= -tol) if diag.size else True,
        "hermitian": is_hermitian(C, chunk_size=chunk_size),
    }
    flat = C.reshape(-1, C.shape[-2], C.shape[-1])
    n = flat.shape[0]
    min_eigenvalue = None
    for start in range(0, n, chunk_size):
        chunk = flat[start:start + chunk_size]
        valid = ~np.any(np.isnan(chunk), axis=(-1, -2))
        if not valid.any():
            continue
        eigvals = np.linalg.eigvalsh(chunk[valid])
        chunk_min = float(np.min(eigvals))
        min_eigenvalue = chunk_min if min_eigenvalue is None else min(min_eigenvalue, chunk_min)
    if min_eigenvalue is not None:
        report["min_eigenvalue"] = min_eigenvalue
        report["psd"] = bool(min_eigenvalue >= -tol)
    else:
        report["min_eigenvalue"] = float("nan")
        report["psd"] = True
    return report
