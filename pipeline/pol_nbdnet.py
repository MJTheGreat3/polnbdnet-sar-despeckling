"""Step 7 & 9: 9-channel input representation, Wishart speckle simulation,
covariance validity checks, and the Wishart distance used as a training loss.

This is the shared utility module the preprocessing guide refers to as
pol_nbdnet.py (cov_to_repr / repr_to_cov / simulate_polsar_speckle /
check_validity / wishart_distance).
"""
import numpy as np
import tensorflow as tf

# Placeholders -- overwrite from compute_db_norm_stats() on real training
# data (valid pixels only) and reuse the same constants everywhere.
DB_MIN = -30.0
DB_MAX = 10.0

_PAIRS = ((0, 1), (0, 2), (1, 2))


def cov_to_repr(C, db_min=DB_MIN, db_max=DB_MAX, eps=1e-12):
    """3x3 covariance -> 9 channels: 3 normalized-dB diag powers +
    6 real/imag parts of the complex coherences g_ij = C_ij / sqrt(C_ii C_jj).
    """
    C = np.asarray(C)
    diag = np.diagonal(C, axis1=-2, axis2=-1).real  # (..., 3)
    db = 10.0 * np.log10(np.clip(diag, eps, None))
    db_norm = np.clip((db - db_min) / (db_max - db_min), 0.0, 1.0)

    coh = []
    with np.errstate(invalid="ignore"):
        for i, j in _PAIRS:
            denom = np.sqrt(np.clip(diag[..., i] * diag[..., j], eps, None))
            g = C[..., i, j] / denom
            coh.append(g.real)
            coh.append(g.imag)
    coh = np.stack(coh, axis=-1)  # (..., 6)

    return np.concatenate([db_norm, coh], axis=-1)  # (..., 9)


def repr_to_cov(repr_arr, db_min=DB_MIN, db_max=DB_MAX):
    """Inverse of cov_to_repr: 9 channels -> 3x3 Hermitian covariance."""
    repr_arr = np.asarray(repr_arr)
    db_norm = repr_arr[..., :3]
    coh = repr_arr[..., 3:]

    db = db_norm * (db_max - db_min) + db_min
    diag = 10.0 ** (db / 10.0)

    shape = repr_arr.shape[:-1]
    C = np.zeros(shape + (3, 3), dtype=np.complex128)
    for idx in range(3):
        C[..., idx, idx] = diag[..., idx]

    for k, (i, j) in enumerate(_PAIRS):
        g = coh[..., 2 * k] + 1j * coh[..., 2 * k + 1]
        cij = g * np.sqrt(diag[..., i] * diag[..., j])
        C[..., i, j] = cij
        C[..., j, i] = np.conj(cij)

    return C


def compute_db_norm_stats(diag_powers, low_pct=1.0, high_pct=99.0, eps=1e-12):
    """DB_MIN/DB_MAX from actual training-set diagonal power statistics
    (valid pixels only). Reuse the returned constants for val/test/inference.
    """
    db = 10.0 * np.log10(np.clip(np.asarray(diag_powers), eps, None))
    db_min = float(np.nanpercentile(db, low_pct))
    db_max = float(np.nanpercentile(db, high_pct))
    return db_min, db_max


def _simulate_polsar_speckle_chunk(flat, L, rng, eig_floor_frac):
    """Core per-chunk math for simulate_polsar_speckle -- kept memory-bounded
    by the caller's chunking, see simulate_polsar_speckle."""
    n = flat.shape[0]
    flat = flat.copy()

    # invalid (NaN) input pixels -- e.g. no-data / masked regions -- would
    # otherwise crash cholesky; substitute identity so those positions decode
    # to garbage instead of erroring. Caller's valid_mask excludes them anyway.
    invalid = np.any(np.isnan(flat), axis=(-1, -2))
    flat[invalid] = np.eye(3)

    # Real data is never perfectly PSD: low-valid-fraction edge pixels
    # (partial multilook windows) and float32 roundoff both produce
    # matrices with tiny negative eigenvalues that a flat epsilon jitter
    # can't reliably fix (also meaningless across the huge dynamic range
    # of raw radar power units). So eigendecompose and floor eigenvalues
    # relative to each matrix's own power before Cholesky.
    flat = 0.5 * (flat + np.conj(np.swapaxes(flat, -1, -2)))  # exact Hermitian
    avg_power = np.clip(np.diagonal(flat, axis1=-2, axis2=-1).real.mean(axis=-1), 1e-30, None)
    eigvals, eigvecs = np.linalg.eigh(flat)  # ascending, (n,3) / (n,3,3)
    floor = np.maximum(eig_floor_frac * avg_power, 1e-9 * np.clip(eigvals[:, -1], 0, None))
    eigvals = np.maximum(eigvals, floor[:, None])
    flat_psd = (eigvecs * eigvals[:, None, :]) @ np.conj(np.swapaxes(eigvecs, -1, -2))
    flat_psd = 0.5 * (flat_psd + np.conj(np.swapaxes(flat_psd, -1, -2)))
    L_chol = np.linalg.cholesky(flat_psd)  # (n, 3, 3)

    max_L = int(L.max())
    noise = (rng.standard_normal((n, max_L, 3)) + 1j * rng.standard_normal((n, max_L, 3))) / np.sqrt(2)
    z = np.einsum("nij,nlj->nli", L_chol, noise)  # (n, max_L, 3)

    look_mask = (np.arange(max_L)[None, :] < L[:, None]).astype(z.dtype)  # (n, max_L)
    z_masked = z * look_mask[..., None]
    # sum_l z_l z_l^H without materializing the (n, max_L, 3, 3) tensor
    C_noisy = np.einsum("nla,nlb->nab", z_masked, np.conj(z)) / L[:, None, None]
    C_noisy[invalid] = np.nan
    return C_noisy


def simulate_polsar_speckle(C_ref, L=1, rng=None, eig_floor_frac=1e-6, chunk_size=200_000):
    """Draw C_noisy from the complex Wishart distribution with mean C_ref
    and L looks, jointly across the 3 channels (preserves cross-channel
    phase -- never add noise to each covariance element independently).

    Processes pixels in chunks: the naive vectorized form materializes an
    (n_pixels, max_looks, 3, 3) tensor, which is fine for a tile but OOMs
    on a full scene (millions of pixels) -- chunk_size bounds peak memory
    regardless of input size.
    """
    rng = np.random.default_rng() if rng is None else rng
    C_ref = np.asarray(C_ref)
    shape = C_ref.shape[:-2]
    n = int(np.prod(shape)) if shape else 1
    flat_full = C_ref.reshape(n, 3, 3)

    L_arr = np.asarray(L)
    L_full = np.full((n,), int(L_arr)) if L_arr.ndim == 0 else L_arr.reshape(n)

    out = np.empty((n, 3, 3), dtype=np.complex128)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        out[start:end] = _simulate_polsar_speckle_chunk(
            flat_full[start:end], L_full[start:end], rng, eig_floor_frac
        )

    return out.reshape(C_ref.shape)


def check_validity(C, tol=1e-6):
    """Hermitian + PSD check for a batch of 3x3 covariance matrices.
    Returns (all_valid: bool, n_violations: int, report: dict).
    """
    from .covariance import check_covariance_validity

    report = check_covariance_validity(C, tol=tol)
    all_valid = report["hermitian"] and report["psd"] and report["diag_nonneg"]
    return all_valid, report


def wishart_distance(C1, C2, eps=1e-8):
    """Symmetric Wishart/KL divergence between two covariance matrices,
    used as the Noise2Noise training loss between prediction and target
    (Step 12). Pure-numpy reference implementation.
    """
    C1 = np.asarray(C1)
    C2 = np.asarray(C2)
    n = C1.shape[-1]
    eye = np.eye(n) * eps

    C1_inv = np.linalg.inv(C1 + eye)
    C2_inv = np.linalg.inv(C2 + eye)

    term = np.trace(C1_inv @ C2, axis1=-2, axis2=-1) + np.trace(C2_inv @ C1, axis1=-2, axis2=-1)
    return 0.5 * (term.real - 2 * n)


def repr_to_cov_tf(repr_tensor, db_min, db_max):
    """TF-differentiable counterpart of repr_to_cov, for use inside a Keras
    loss (polarimetric roadmap Step 3). db_min/db_max broadcast against the
    (..., 3) diagonal channels -- pass per-tile values (not the module-level
    placeholders) since normalization is fit per scene, not globally."""
    repr_tensor = tf.cast(repr_tensor, tf.float32)
    db_min = tf.cast(db_min, tf.float32)
    db_max = tf.cast(db_max, tf.float32)

    db_norm = repr_tensor[..., :3]
    coh = repr_tensor[..., 3:]
    db = db_norm * (db_max - db_min) + db_min
    diag = tf.cast(tf.pow(10.0, db / 10.0), tf.complex64)  # (..., 3)

    def gamma(k):
        return tf.complex(coh[..., 2 * k], coh[..., 2 * k + 1])

    c01 = gamma(0) * tf.sqrt(diag[..., 0] * diag[..., 1])
    c02 = gamma(1) * tf.sqrt(diag[..., 0] * diag[..., 2])
    c12 = gamma(2) * tf.sqrt(diag[..., 1] * diag[..., 2])

    row0 = tf.stack([diag[..., 0], c01, c02], axis=-1)
    row1 = tf.stack([tf.math.conj(c01), diag[..., 1], c12], axis=-1)
    row2 = tf.stack([tf.math.conj(c02), tf.math.conj(c12), diag[..., 2]], axis=-1)
    return tf.stack([row0, row1, row2], axis=-2)  # (..., 3, 3)


def _inv3x3(M):
    """Closed-form 3x3 complex matrix inverse (adjugate/determinant),
    batched over any leading dims. tf.linalg.inv has no XLA GPU kernel for
    complex64 (MatrixInverse unsupported on XLA_GPU_JIT, hit when this is
    used inside a training loss -- Keras 3 jit-compiles by default) --
    plain arithmetic ops like this are XLA-compatible."""
    a, b, c = M[..., 0, 0], M[..., 0, 1], M[..., 0, 2]
    d, e, f = M[..., 1, 0], M[..., 1, 1], M[..., 1, 2]
    g, h, i = M[..., 2, 0], M[..., 2, 1], M[..., 2, 2]

    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    row0 = tf.stack([e * i - f * h, c * h - b * i, b * f - c * e], axis=-1)
    row1 = tf.stack([f * g - d * i, a * i - c * g, c * d - a * f], axis=-1)
    row2 = tf.stack([d * h - e * g, b * g - a * h, a * e - b * d], axis=-1)
    adj_t = tf.stack([row0, row1, row2], axis=-2)  # already transposed (= adjugate)
    return adj_t / det[..., None, None]


def wishart_distance_tf(C1, C2, eps=1e-6):
    """TF-differentiable counterpart of wishart_distance, batched over any
    leading dims. Returns per-pixel distance (..., ), caller reduces."""
    n = C1.shape[-1]
    eye = tf.eye(n, dtype=tf.complex64) * eps

    C1_inv = _inv3x3(C1 + eye)
    C2_inv = _inv3x3(C2 + eye)

    term = tf.linalg.trace(C1_inv @ C2) + tf.linalg.trace(C2_inv @ C1)
    return 0.5 * (tf.math.real(term) - 2 * n)
