"""Step 13: final checklist before training, runnable on a sample batch."""
import numpy as np

from .pol_nbdnet import check_validity


def run_checklist(C=None, C_noisy=None, C_ref=None, C_a=None, C_b=None,
                   inputs=None, targets=None, splits=None):
    results = {}

    if C is not None:
        ok, report = check_validity(C)
        results["C_hermitian_psd"] = ok
        results["C_report"] = report

    if C_noisy is not None and C_ref is not None:
        ok_n, rep_n = check_validity(C_noisy)
        ok_r, rep_r = check_validity(C_ref)
        results["synthetic_route_valid"] = ok_n and ok_r
        results["C_noisy_report"] = rep_n
        results["C_ref_report"] = rep_r

    if C_a is not None and C_b is not None:
        results["independent_looks_shapes_match"] = C_a.shape == C_b.shape

    if inputs is not None and targets is not None:
        results["tiles_aligned_shape"] = inputs.shape == targets.shape
        results["no_nan_inputs"] = bool(np.isfinite(inputs).all())
        results["no_nan_targets"] = bool(np.isfinite(targets).all())

    if splits is not None:
        from .splitting import verify_no_leakage

        try:
            verify_no_leakage(splits)
            results["no_scene_leakage"] = True
        except ValueError as e:
            results["no_scene_leakage"] = False
            results["leakage_error"] = str(e)

    results["all_ok"] = all(v for k, v in results.items() if isinstance(v, bool))
    return results
