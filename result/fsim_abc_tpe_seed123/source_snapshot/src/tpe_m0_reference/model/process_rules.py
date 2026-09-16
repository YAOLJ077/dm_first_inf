"""Shared definitions for leaf moisture and normalized incubation completion."""

import numpy as np
from model.parameter_table import TABLE_CONFIG


INCUBATION_COMPLETION = TABLE_CONFIG['incubation']['spl_end']


def leaf_moisture_from_vpd(vpd_hpa, threshold_hpa):
    """Return the leaf-moisture proxy using physical VPD alone (hPa)."""
    if not np.isfinite(threshold_hpa) or threshold_hpa < 0:
        raise ValueError("The leaf-moisture VPD threshold must be finite and non-negative")
    values = np.asarray(vpd_hpa, dtype=float)
    return np.isfinite(values) & (values <= threshold_hpa)


def validate_incubation_completion(config):
    """Reject obsolete fitted completion values instead of silently using them."""
    value = config.get("incubation", {}).get("spl_end", INCUBATION_COMPLETION)
    if value != INCUBATION_COMPLETION:
        raise ValueError("incubation.spl_end is fixed at 1; it is not a calibration parameter")
