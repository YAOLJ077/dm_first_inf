# utils/sobol_param_meta.py
# ======================================================
# Sobol parameter meta information
# Centralized mapping for plotting & reporting
# ======================================================

PARAM_ABBR_MAP = {
    "mp_tmp__rh_wet_night": "MTWN",
    "sporulation__MP_min": "SMMI",
    "incubation__spl_end": "INCE",
    "infection__infection_condition": "IC",
    "deposition__rain_min": "DERM",
    "dispersal__wind_min": "DIWM",
    "dispersal__rain_min": "DIRM",
    "sporulation__Tmin": "STMI",
    "sporulation__Topt": "STOP",
    "sporulation__Tmax": "STMA",
    "zoospore_alive__vpd": "ZVPD",
    "zoospore_alive__rain": "ZARA",
}


def param_to_abbr(param: str) -> str:
    """
    Convert full parameter name to abbreviation.
    Fallback to original name if not defined.
    """
    return PARAM_ABBR_MAP.get(param, param)
