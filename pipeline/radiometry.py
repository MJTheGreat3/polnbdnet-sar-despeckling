"""Step 5: keep math in linear power, dB only for visualization/input transform."""
import numpy as np


def to_db(power, eps=1e-12):
    return 10.0 * np.log10(np.clip(power, eps, None))


def incidence_trend(power_db, incidence_deg, mask):
    """Least-squares slope of power_db vs incidence angle, valid pixels only."""
    m = mask & np.isfinite(power_db) & np.isfinite(incidence_deg)
    if m.sum() < 2:
        return {"slope_db_per_deg": float("nan"), "n": int(m.sum())}
    x = incidence_deg[m].ravel()
    y = power_db[m].ravel()
    slope, intercept = np.polyfit(x, y, 1)
    return {"slope_db_per_deg": float(slope), "intercept_db": float(intercept), "n": int(m.sum())}


def flag_layover_shadow(slope_deg, incidence_deg, mask=None):
    """Flag pixels where local terrain slope makes them unreliable.

    Layover: local slope >= incidence angle (terrain tilts toward radar
    faster than the look angle). Shadow: local slope <= -(90 - incidence).
    Both are coarse heuristics per Step 5.3 -- use as a QC flag, not a hard mask.
    """
    layover = slope_deg >= incidence_deg
    shadow = slope_deg <= -(90.0 - incidence_deg)
    flagged = layover | shadow
    if mask is not None:
        flagged &= mask
    return flagged
