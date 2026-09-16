"""Defaults read from config/fsim_runtime_parameters.csv.

load_config(path) creates a fresh dictionary from an alternative table.
The legacy config dictionary remains available to existing callers.
"""
from model.parameter_table import DEFAULT_TABLE, load_config

config = load_config(DEFAULT_TABLE)
