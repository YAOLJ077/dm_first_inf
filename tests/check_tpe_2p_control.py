"""Run in an isolated interpreter so reference and live model packages cannot mix."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import calibrate_m0_losyo_tpe_2p as control
import json
import tempfile
import unittest
import pandas as pd


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.s = json.loads((control.ROOT / 'config/fsim_m0_losyo_tpe_2p.json').read_text())
        self.base = control.load_config(control.ROOT / self.s['parameter_table'])
        self.anchor = dict(VPD_w=3.875, I_req=282.5, RH_sp=80., MP_sp=3)

    def test_only_two_dimensions_including_after_startup(self):
        control.validate_settings(self.s)
        study = control.make_study(20260914, self.s, self.anchor)
        for i in range(140):
            trial, params = control.ask_params(study, self.s['search_space'])
            self.assertEqual(set(trial.params), {'VPD_w', 'I_req'})
            self.assertEqual((params['RH_sp'], params['MP_sp']), (80., 3))
            cfg = control.runtime_config(self.base, params)
            self.assertEqual(cfg['fixed']['min_moist_hours'], 3)
            self.assertEqual(cfg['mp_tmp']['rh_wet_night'], 80.)
            control.tell(study, trial, dict(n_no_prediction=0, training_MAE_days=10., optimizer_value=10.))

    def test_four_workers_use_frozen_model(self):
        import model.first_infection as model
        self.assertTrue(Path(model.__file__).is_relative_to(control.REFERENCE))
        weather = pd.DataFrame(dict(datetime=pd.date_range('2024-01-01', periods=48, freq='h').astype(str),
                                    air_temp=20., relative_humidity=50., precip_acc_period_adjusted=0., wind_speed=2.))
        rb = dict(year=2024, BBCH_15='2024-01-01', weather_hourly=weather.to_dict('records'))
        keys = [f'case{i}' for i in range(5)]
        cases = [dict(input_key=k, L='2024-01-01', U='2024-01-02', rb=rb) for k in keys]
        evaluator = control.Evaluator(cases, [], self.base, 4)
        try:
            predictions, _ = evaluator.evaluate(self.anchor, keys)
            self.assertEqual(len(predictions), 5)
            self.assertTrue(all(r['status'] == 'no_prediction' for r in predictions))
            self.assertTrue(all(r['RH_sp'] == 80 and r['MP_sp'] == 3 for r in predictions))
            metrics = control.training_metrics(predictions, keys)
            self.assertFalse(metrics['feasible'])
            self.assertIsNone(metrics['training_MAE_days'])
        finally:
            evaluator.close()

    def test_resume_replays_two_dimensional_search(self):
        s = {**self.s, 'n_trials': 8, 'n_startup_trials': 3}
        keys = [f'case{i}' for i in range(5)]
        class Fake:
            count = 0
            def evaluate(self, params, keys):
                self.count += 1
                return [dict(input_key=k, status='predicted', signed_interval_error_days=params['VPD_w']) for k in keys], []
        with tempfile.TemporaryDirectory() as tmp:
            fake = Fake()
            records = control.run_seed(s, 3, self.anchor, keys, Path(tmp), fake, False)
            self.assertEqual(fake.count, 8)
            replay = control.run_seed(s, 3, self.anchor, keys, Path(tmp), fake, False)
            self.assertEqual(fake.count, 8)
            self.assertEqual(records, replay)


if __name__ == '__main__':
    unittest.main()
