#!/usr/bin/env python3
"""Audit and migrate one known LOSYO scheduler revision without changing trials."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import calibrate_fsim_losyo_joblib as run

POOL_PATH = 'src/fsim_joblib_pool.py'
OLD_SHA256 = '2d22a9b6f9275bdcc33dfcd56a034a038f824f6f06c2c77e020d7e4cb3b13355'
NEW_SHA256 = 'ab3bd7002b4c0f2e25e0649ed8443ca71e209f0d3540506314f43eb004c16c49'


def verify_upgrade(saved, current):
    """Accept exactly the reviewed scheduler replacement, never other drift."""
    if saved == current:
        return False
    adjusted = copy.deepcopy(saved)
    if set(adjusted['dimensions']) != {'2p', '3p', '4p'}:
        raise ValueError('Expected all three LOSYO dimensions')
    for dimension, identity in adjusted['dimensions'].items():
        if identity['source_sha256'].get(POOL_PATH) != OLD_SHA256:
            raise ValueError(f'Unrecognized original scheduler: {dimension}')
        if current['dimensions'][dimension]['source_sha256'].get(POOL_PATH) != NEW_SHA256:
            raise ValueError('Current scheduler is not the reviewed streaming revision')
        identity['source_sha256'][POOL_PATH] = NEW_SHA256
    if adjusted != current:
        raise ValueError('Inputs, settings, versions or other source files changed; migration refused')
    return True


def migrate(output, current, apply=False):
    if not (output / 'run_identity.json').is_file():
        raise ValueError('No existing LOSYO run_identity.json in output directory')
    with run.base_runner.output_lock(output):
        identity_path = output / 'run_identity.json'
        saved = json.loads(identity_path.read_text())
        if not verify_upgrade(saved, current):
            return 'Run identity already matches; no migration needed.'
        original = output / 'source_snapshot' / POOL_PATH
        if hashlib.sha256(original.read_bytes()).hexdigest() != OLD_SHA256:
            raise ValueError('Original scheduler snapshot is missing or does not match')
        pool_bytes = (run.ROOT / POOL_PATH).read_bytes()
        if hashlib.sha256(pool_bytes).hexdigest() != NEW_SHA256:
            raise ValueError('Current scheduler source changed during verification')
        if not apply:
            return 'Verified: only the reviewed scheduler differs. Run again with --apply to migrate.'
        # Keep the original snapshot intact. Store the replacement and identities
        # in an audit directory; replace the active identity LAST and atomically.
        audit = output / 'scheduler_migrations' / NEW_SHA256
        audit.mkdir(parents=True, exist_ok=True)
        run.base_runner.core.write_json(audit / 'identity_before.json', saved)
        run.base_runner.core.write_json(audit / 'identity_after.json', current)
        (audit / 'fsim_joblib_pool.py').write_bytes(pool_bytes)
        run.base_runner.core.write_json(audit / 'migration.json', dict(
            migrated_utc=datetime.now(timezone.utc).isoformat(),
            source_path=POOL_PATH, old_sha256=OLD_SHA256, new_sha256=NEW_SHA256,
            change='Batch dispatcher replaced by continuously fed loky executor',
            preserved='All scientific settings, inputs, versions, trials and model caches',
            original_snapshot='source_snapshot/' + POOL_PATH))
        run.base_runner.core.write_json(identity_path, current)
        return 'Scheduler migration recorded. Resume with the original calibration command and --resume.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=run.ROOT / 'result/fsim_losyo_joblib')
    parser.add_argument('--settings', type=Path, default=run.ROOT / 'config/fsim_losyo_joblib.json')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--apply', action='store_true', help='Write audited migration; default only verifies')
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    if args.smoke:
        settings = run.smoke_settings(settings)
    _, dimensions = run.prepare(settings)
    current = dict(format_version=1, order=list(run.ORDER),
                   dimensions={d: identity for d, _, _, _, identity in dimensions})
    print(migrate(args.output_dir.resolve(), current, args.apply), flush=True)


if __name__ == '__main__':
    main()
