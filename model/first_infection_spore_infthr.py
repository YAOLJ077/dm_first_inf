"""Compatibility name for the structurally aligned FSIM-S engine.

FSIM-S now uses the external dose provider passed to ``FirstInfection``.  No
spore pressure is applied to infection thresholds or endogenous production.
"""
from model.first_infection import FirstInfection


class FirstInfectionSpore(FirstInfection):
    """FSIM core with optional observed-background dose cohorts."""
    pass
