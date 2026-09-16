#!/usr/bin/env python3
"""Calibrate only beta in [0,1] with training-only constrained TPE."""
from pressure_experiment_runner import entry

if __name__ == '__main__':
    entry('beta_tpe')
