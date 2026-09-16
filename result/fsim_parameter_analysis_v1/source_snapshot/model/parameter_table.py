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
                'lower', 'upper', 'grid_levels'}
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
            value, levels = row['value'], int(row['grid_levels'])
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not all(math.isfinite(v) for v in (lo, hi, value))
                    or not 0 <= lo < hi or not lo <= value <= hi or levels < 3):
                raise ValueError(f'Invalid candidate range/default/grid: {name}')
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
