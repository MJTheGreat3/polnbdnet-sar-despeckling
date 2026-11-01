"""Polarimetric evaluation suite (report.md Step 1 of the polarimetric
improvements roadmap, docs/PolNBDNet_Polarimetric_Improvements_Step_by_Step-1.pdf).

evaluate_nbdnet.py only scores the reconstructed total-power (span) image --
that can't tell whether a model preserves the cross-channel relationships
that make the data polarimetric in the first place. This module adds:
H/A/alpha (entropy/anisotropy/mean-alpha from the coherency-matrix eigen-
decomposition), HH-VV coherence (magnitude + phase), a PSD-validity
fraction, and the ratio-image self-consistency check (I_noisy / I_candidate
should look like pure speckle: mean ~= 1, var ~= 1/L, no scene structure).
"""
import numpy as np

# Lexicographic -> Pauli basis unitary transform. Our covariance C is built
# from k_lex = [S_hh, sqrt(2) S_hv, S_vv]^T (pol_nbdnet.py convention), and
# the Pauli scattering vector is k_pauli = (1/sqrt2)[S_hh+S_vv, S_hh-S_vv, 2 S_hv]^T.
# T = U C U^H has the same eigenvalues as C (U unitary) -- H/A only need
# those eigenvalues; alpha additionally needs the eigenvectors in this basis.
_U = np.array([
    [1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
    [1 / np.sqrt(2), 0, -1 / np.sqrt(2)],
    [0, 1, 0],
], dtype=np.complex128)


def cov_to_coherency(C):
    """3x3 lexicographic covariance -> 3x3 coherency matrix T = U C U^H."""
    return np.einsum("ij,...jk,kl->...il", _U, C, _U.conj().T)


def h_a_alpha(C):
    """Entropy, anisotropy, mean alpha (radians) from C's eigendecomposition.
    Returns (H, A, alpha), each shaped like C.shape[:-2]."""
    T = cov_to_coherency(C)
    T = 0.5 * (T + np.conj(np.swapaxes(T, -1, -2)))  # exact Hermitian
    eigvals, eigvecs = np.linalg.eigh(T)  # ascending
    eigvals = eigvals[..., ::-1]  # descending: lambda1 >= lambda2 >= lambda3
    eigvecs = eigvecs[..., ::-1]

    total = np.clip(eigvals.sum(axis=-1, keepdims=True), 1e-30, None)
    p = np.clip(eigvals / total, 1e-12, 1.0)

    H = -(p * (np.log(p) / np.log(3))).sum(axis=-1)
    A = (eigvals[..., 1] - eigvals[..., 2]) / np.clip(eigvals[..., 1] + eigvals[..., 2], 1e-30, None)

    alpha_i = np.arccos(np.clip(np.abs(eigvecs[..., 0, :]), 0.0, 1.0))  # first (Pauli) component of each eigenvector
    alpha = (p * alpha_i).sum(axis=-1)
    return H, A, alpha


def hhvv_coherence(C):
    """HH-VV complex coherence gamma = C13 / sqrt(C11 C33); returns (magnitude, phase_rad)."""
    c13 = C[..., 0, 2]
    denom = np.sqrt(np.clip(C[..., 0, 0].real * C[..., 2, 2].real, 1e-30, None))
    gamma = c13 / denom
    return np.abs(gamma), np.angle(gamma)


def psd_fraction(C, tol=1e-6):
    """Fraction of pixels with min eigenvalue >= -tol (valid covariance matrix)."""
    flat = C.reshape(-1, 3, 3)
    valid = ~np.any(np.isnan(flat), axis=(-1, -2))
    if not valid.any():
        return float("nan")
    eigvals = np.linalg.eigvalsh(flat[valid])
    return float(np.mean(eigvals.min(axis=-1) >= -tol))


def enl(span):
    """Equivalent number of looks: (mean I)^2 / var(I), over all given pixels."""
    span = np.asarray(span)
    m = np.nanmean(span)
    v = np.nanvar(span)
    return float(m ** 2 / v) if v > 1e-30 else float("inf")


def ratio_image_stats(noisy_span, candidate_span):
    """noisy / candidate ratio: should look like pure speckle (mean~=1,
    var~=1/L) if candidate removed only noise and no real structure."""
    r = noisy_span / np.clip(candidate_span, 1e-30, None)
    return float(np.nanmean(r)), float(np.nanvar(r))


def _circular_mae_deg(a, b):
    diff = np.angle(np.exp(1j * (a - b)))
    return float(np.degrees(np.nanmean(np.abs(diff))))


def polarimetric_errors(C_candidate, C_ref):
    """Mean errors of H, A, alpha (deg), and HH-VV coherence (mag, phase deg)
    of C_candidate against C_ref -- the headline "did it preserve polarimetry"
    numbers, not just span PSNR."""
    H_c, A_c, alpha_c = h_a_alpha(C_candidate)
    H_r, A_r, alpha_r = h_a_alpha(C_ref)
    mag_c, phase_c = hhvv_coherence(C_candidate)
    mag_r, phase_r = hhvv_coherence(C_ref)

    return {
        "H_mae": float(np.nanmean(np.abs(H_c - H_r))),
        "A_mae": float(np.nanmean(np.abs(A_c - A_r))),
        "alpha_mae_deg": float(np.degrees(np.nanmean(np.abs(alpha_c - alpha_r)))),
        "hhvv_coh_mag_mae": float(np.nanmean(np.abs(mag_c - mag_r))),
        "hhvv_coh_phase_mae_deg": _circular_mae_deg(phase_c, phase_r),
    }


def evaluate_candidate(C_candidate, C_ref, noisy_span=None, L=None, tol=1e-6):
    """Full per-candidate suite: validity, polarimetric parameter errors
    against the clean reference, and (if noisy_span given) the ratio-image
    self-consistency check against the raw noisy input."""
    out = {"psd_valid_fraction": psd_fraction(C_candidate, tol=tol)}
    out.update(polarimetric_errors(C_candidate, C_ref))

    if noisy_span is not None:
        span = np.diagonal(C_candidate, axis1=-2, axis2=-1).real.sum(axis=-1)
        ratio_mean, ratio_var = ratio_image_stats(noisy_span, span)
        out["ratio_mean"] = ratio_mean
        out["ratio_var"] = ratio_var
        out["ratio_var_expected"] = 1.0 / L if L else None
        out["enl"] = enl(span)

    return out
