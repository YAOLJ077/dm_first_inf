"""Validated runtime table, independent of the model and scientific libraries."""
from __future__ import annotations
import copy
import csv
import json
import math
from pathlib import Path

DEFAULT_TABLE = Path(__file__).resolve().parents[1] / 'config/fsim_runtime_parameters.csv'


def read_parameter_table(path=DEFAULT_TABLE):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    required = {'parameter', 'config_key', 'value_json', 'role', 'scope',
                'lower', 'upper'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError('Invalid or empty FSIM parameter table')
    ids, keys = set(), set()
    for row in rows:
        name, key = row['parameter'], row['config_key']
        if name in ids or key in keys or len(key.split('.')) != 2:
            raise ValueError(f'Duplicate parameter/key or invalid key: {name}, {key}')
        ids.add(name)
        keys.add(key)
        row['value'] = json.loads(row['value_json'])
        if row['role'] == 'candidate':
            lo, hi = float(row['lower']), float(row['upper'])
            value = row['value']
            levels = int(row['grid_levels']) if row.get('grid_levels', '').strip() else None
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not all(math.isfinite(v) for v in (lo, hi, value))
                    or not 0 <= lo < hi or not lo <= value <= hi
                    or (levels is not None and levels < 3)):
                raise ValueError(f'Invalid candidate range/default/grid: {name}')
            if row.get('grid_values_json', '').strip():
                grid = json.loads(row['grid_values_json'])
                if (not isinstance(grid, list) or len(grid) < 3
                        or (levels is not None and len(grid) != levels)
                        or any(isinstance(v, bool) or not isinstance(v, (int, float))
                               or not math.isfinite(v) for v in grid)
                        or any(a >= b for a, b in zip(grid, grid[1:]))
                        or grid[0] != lo or grid[-1] != hi):
                    raise ValueError(f'Explicit grid must be ordered, unique and span bounds: {name}')
                row['grid_values'] = grid
    return rows


def load_config(path=DEFAULT_TABLE):
    rows = read_parameter_table(path)
    reference = {r['parameter']: r for r in read_parameter_table(DEFAULT_TABLE)}
    if {r['parameter'] for r in rows} != set(reference):
        raise ValueError('Configuration must contain the full registered parameter set')
    cfg = {'sporulation': {}}
    for row in rows:
        ref = reference[row['parameter']]
        for field in ('config_key', 'role', 'scope'):
            if row[field] != ref[field]:
                raise ValueError(f'Cannot change parameter identity: {row["parameter"]}.{field}')
        if row['role'] not in ('candidate', 'structure') and row['value'] != ref['value']:
            raise ValueError(f'Fixed setting differs from current model: {row["parameter"]}')
        group, key = row['config_key'].split('.')
        cfg.setdefault(group, {})[key] = copy.deepcopy(row['value'])
    if cfg['incubation']['spl_end'] != 1:
        raise ValueError('Incubation completion is defined as 1')
    if cfg['conventions']['bbch_min'] != 15 or cfg['conventions']['initialization'] != '01-01':
        raise ValueError('Current controller supports BBCH15 and January-1 analysis initialization')
    if not isinstance(cfg['dispersal']['use_wind_gate'], bool):
        raise ValueError('Wind gate switch must be boolean')
    return cfg


def candidate_rows(model, path=DEFAULT_TABLE):
    if model not in ('M0', 'M1'):
        raise ValueError(f'Unknown model: {model}')
    return [r for r in read_parameter_table(path)
            if r['role'] == 'candidate' and model in r['scope'].split('|')]


TABLE_CONFIG = load_config()
