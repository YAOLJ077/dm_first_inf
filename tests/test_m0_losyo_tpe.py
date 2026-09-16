"""TPE leakage, feasibility, replay and runtime-configuration checks (synthetic data)."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import optuna
import pandas as pd

from model.first_infection import FirstInfection
from model.parameter_table import load_config
from model.sporulation_timing import normalized_lalancette_sporulation
from src.calibrate_m0_losyo_tpe import (
    ROOT, Evaluator, choose_training, make_study, ask_params, tell, run_folds,
    run_seed, runtime_config, training_metrics, validate_settings, worker_evaluate,
)


class SyntheticEvaluator:
    def __init__(self, base):
        self.base = base
        self.calls = []

    def evaluate(self, params, keys):
        self.calls.append((copy.deepcopy(params), list(keys)))
        # Held-out performance is deliberately bad, irrespective of training fit.
        error = 999 if len(keys) == 1 else abs(params['I_req'] - 200) / 100 + params['MP_sp']
        return [dict(input_key=k, status='predicted', signed_interval_error_days=error,
                     **params) for k in keys], []


class TPEChecks(unittest.TestCase):
    def setUp(self):
        self.s = json.loads((ROOT / 'config/fsim_m0_losyo_tpe.json').read_text())
        self.base = load_config(ROOT / self.s['parameter_table'])
        self.anchor = dict(VPD_w=3.875, I_req=282.5, RH_sp=80, MP_sp=3)
        self.keys = [f'case{i}' for i in range(5)]
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def small_settings(self):
        return {**self.s, 'n_trials': 8, 'n_startup_trials': 3, 'checkpoints': [3, 8]}

    def test_infeasible_is_excluded_even_with_zero_optimizer_placeholder(self):
        records = [dict(input_key=k, status='predicted', signed_interval_error_days=10) for k in self.keys]
        good = training_metrics(records, self.keys)
        records[0]['status'] = 'no_prediction'
        bad = training_metrics(records, self.keys)
        self.assertIsNone(bad['training_MAE_days'])
        self.assertEqual(bad['optimizer_value'], 0)
        trials = [dict(seed=1, number=i, params=self.anchor, metrics=m) for i, m in enumerate([bad, good])]
        self.assertEqual(choose_training(trials, 1e-9)[0]['number'], 1)
        study = make_study(1, self.s, self.anchor)
        for m in [bad, good]:
            trial, _ = ask_params(study, self.s['search_space'])
            tell(study, trial, m)
        self.assertEqual(study.trials[0].system_attrs['constraints'], (1,))
        self.assertEqual(study.best_trial.number, 1)
        records[0]['status'] = 'runtime_error'
        with self.assertRaises(RuntimeError):
            training_metrics(records, self.keys)
        with self.assertRaises(ValueError):
            training_metrics(records, self.keys[:4] + ['heldout'])

    def test_resume_replays_tpe_proposals_after_startup_without_recalculation(self):
        s = self.small_settings()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            partial = SyntheticEvaluator(self.base)
            # Interrupt after five completed trials, already inside TPE sampling.
            original = partial.evaluate
            def interrupt(params, keys):
                if len(partial.calls) == 5:
                    raise KeyboardInterrupt()
                return original(params, keys)
            partial.evaluate = interrupt
            with self.assertRaises(KeyboardInterrupt):
                run_seed(s, 17, self.anchor, self.keys, path / 'resume', partial, False)
            resumed = SyntheticEvaluator(self.base)
            a = run_seed(s, 17, self.anchor, self.keys, path / 'resume', resumed, False)
            uninterrupted = SyntheticEvaluator(self.base)
            b = run_seed(s, 17, self.anchor, self.keys, path / 'fresh', uninterrupted, False)
            self.assertEqual(a, b)
            self.assertEqual(len(resumed.calls), 3)
            self.assertEqual(len(uninterrupted.calls), 8)

    def test_six_folds_freeze_training_selection_before_held_out_calls(self):
        s = self.small_settings()
        keys = self.keys + ['case5']
        evaluator = SyntheticEvaluator(self.base)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            original = evaluator.evaluate
            def evaluate(params, requested):
                if len(requested) == 1:
                    selection = json.loads((out / 'folds' / requested[0] / 'selection.json').read_text())
                    self.assertNotIn(requested[0], selection['training_keys'])
                    self.assertEqual(params, {n: selection[n] for n in self.anchor})
                return original(params, requested)
            evaluator.evaluate = evaluate
            summary, count, _ = run_folds(s, [dict(input_key=k) for k in keys],
                {k: self.anchor for k in keys}, out, evaluator, False)
            self.assertEqual(summary['MAE_days'], 999)
            self.assertEqual(count, 6 * 2 * 8 * 5 + 6)
            self.assertEqual(len([k for _, k in evaluator.calls if len(k) == 1]), 6)
            for key in keys:
                for path in (out / 'folds' / key).glob('seed_*/trial_*.json'):
                    trial = json.loads(path.read_text())
                    self.assertEqual(set(trial['training_keys']), set(keys) - {key})
            # A completed resume also reuses only frozen held-out predictions.
            evaluator.calls.clear()
            run_folds(s, [dict(input_key=k) for k in keys], {k: self.anchor for k in keys}, out, evaluator, False)
            self.assertEqual(evaluator.calls, [])

    def test_runtime_override_is_local_and_worker_does_not_batch_different_parameters(self):
        params = {**self.anchor, 'RH_sp': 85, 'MP_sp': 6}
        cfg = runtime_config(self.base, params)
        self.assertEqual(cfg['fixed']['min_moist_hours'], 6)
        self.assertEqual(cfg['mp_tmp']['rh_wet_night'], 85)
        self.assertFalse(cfg['dispersal']['use_wind_gate'])
        expected = copy.deepcopy(self.base['fixed'])
        expected['min_moist_hours'] = 6
        self.assertEqual(cfg['fixed'], expected)
        self.assertEqual(self.base['fixed']['min_moist_hours'], 3)
        from src import calibrate_m0_losyo_tpe as module
        with patch.object(module, '_CASES', [dict(input_key='a')], create=True), \
             patch.object(module, '_BASE', self.base, create=True), \
             patch.object(module, 'evaluate_group', return_value=[{'status': 'no_prediction'}]) as run:
            worker_evaluate((0, params))
            case, samples, actual = run.call_args.args[0]
            self.assertEqual(len(samples), 1)
            self.assertEqual(actual['fixed']['min_moist_hours'], 6)

    def test_mp_default_compatibility_and_runtime_gate(self):
        for temperature in (10, 15, 20, 25, 30):
            for hours in range(13):
                self.assertEqual(normalized_lalancette_sporulation(temperature, hours),
                                 normalized_lalancette_sporulation(temperature, hours, 3))
        # Spy on the actual stage call: it must pass the runtime MP, not a global.
        weather = pd.DataFrame(dict(datetime=pd.date_range('2024-01-01', periods=48, freq='h'),
                                    air_temp=20., relative_humidity=90.,
                                    precip_acc_period_adjusted=0., wind_speed=2.))
        weather['date'] = weather.datetime.dt.normalize()
        daily = pd.DataFrame({'datetime': pd.date_range('2024-01-01', periods=2)})
        state = SimpleNamespace(weather_hourly=weather, weather_daily=daily)
        cfg = runtime_config(self.base, {**self.anchor, 'MP_sp': 6})
        with patch('model.first_infection.sporulation_dose_for_day', return_value=0.) as dose:
            FirstInfection(state, cfg).stage_attached_and_dispersal()
            self.assertTrue(dose.called)
            self.assertTrue(all(c.kwargs['min_moist_hours'] == 6 for c in dose.call_args_list))

    def test_settings_reject_invalid_budget_and_space(self):
        validate_settings(self.s)
        for edit in [dict(n_startup_trials=513), dict(workers=3), dict(seeds=[1, 1]), dict(require_all_training_predictions=False)]:
            with self.assertRaises(ValueError):
                validate_settings({**self.s, **edit})

    def test_four_process_evaluation_on_tiny_synthetic_weather(self):
        weather = pd.DataFrame(dict(datetime=pd.date_range('2024-01-01', periods=48, freq='h').astype(str),
                                    air_temp=20., relative_humidity=50.,
                                    precip_acc_period_adjusted=0., wind_speed=2.))
        rb = dict(year=2024, BBCH_15='2024-01-01', weather_hourly=weather.to_dict('records'))
        cases = [dict(input_key=k, L='2024-01-01', U='2024-01-02', rb=rb) for k in self.keys]
        evaluator = Evaluator(cases, [{**cases[0], 'weather_policy': 'keep_first'}], self.base, 4)
        try:
            predictions, duplicates = evaluator.evaluate({**self.anchor, 'MP_sp': 6}, self.keys)
            self.assertEqual([r['input_key'] for r in predictions], self.keys)
            self.assertEqual(len(duplicates), 1)
            self.assertTrue(all(r['status'] == 'no_prediction' and r['MP_sp'] == 6 for r in predictions))
            self.assertEqual(training_metrics(predictions, self.keys)['n_no_prediction'], 5)
        finally:
            evaluator.close()


if __name__ == '__main__':
    unittest.main()
