"""Step 12: split SLC synthetic aperture into two independent sub-looks
for real Noise2Noise training pairs (no synthetic noise model needed)."""
import numpy as np


def split_subaperture(slc, azimuth_axis=0, guard_frac=0.0):
    """Split one SLC channel's azimuth spectrum into first/second half.

    Returns (slc_a, slc_b): same scene, statistically independent speckle.
    guard_frac drops a fraction of the spectrum at the split boundary
    from both halves to reduce cross-talk between the two sub-looks.
    """
    slc = np.asarray(slc)
    spec = np.fft.fftshift(np.fft.fft(slc, axis=azimuth_axis), axes=azimuth_axis)
    n = spec.shape[azimuth_axis]
    mid = n // 2
    guard = int(n * guard_frac / 2)

    idx_a = [slice(None)] * spec.ndim
    idx_b = [slice(None)] * spec.ndim
    idx_a[azimuth_axis] = slice(0, max(mid - guard, 0))
    idx_b[azimuth_axis] = slice(min(mid + guard, n), n)

    spec_a = np.zeros_like(spec)
    spec_b = np.zeros_like(spec)
    spec_a[tuple(idx_a)] = spec[tuple(idx_a)]
    spec_b[tuple(idx_b)] = spec[tuple(idx_b)]

    slc_a = np.fft.ifft(np.fft.ifftshift(spec_a, axes=azimuth_axis), axis=azimuth_axis)
    slc_b = np.fft.ifft(np.fft.ifftshift(spec_b, axes=azimuth_axis), axis=azimuth_axis)
    return slc_a, slc_b


def split_scene_subapertures(scene, azimuth_axis=0, guard_frac=0.0):
    """Apply split_subaperture to all four polarizations of a loaded scene."""
    out_a, out_b = {}, {}
    for pol in ("hh", "hv", "vh", "vv"):
        a, b = split_subaperture(scene[pol], azimuth_axis=azimuth_axis, guard_frac=guard_frac)
        out_a[pol] = a
        out_b[pol] = b
    return out_a, out_b
