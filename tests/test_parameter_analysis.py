"""Configuration wiring, censoring metrics, and cached-versus-public entry tests."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model.parameter_table import DEFAULT_TABLE, load_config, candidate_rows
from model.crop_season import CropSeasonFirst
from src.analyze_fsim_parameters import (configure, score_window, evaluate_group,
                                        make_design, summarize, check_zero_gate,
                                        elementary_effects, profile_and_near_optimal,
                                        load_cases)

ROOT = Path(__file__).resolve().parents[1]


class ParameterAnalysisTests(unittest.TestCase):
    def write_table(self, directory, parameter, column, value):
        with DEFAULT_TABLE.open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row['parameter'] == parameter:
                row[column] = value
        path = Path(directory)/'config.csv'
        with path.open('w',newline='') as handle:
            writer=csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader(); writer.writerows(rows)
        return path

    def test_table_controls_real_configuration_and_isolates_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.write_table(directory,'VPD_w','value_json','3.0')
            cfg=load_config(path)
            self.assertEqual(cfg['zoospore_alive']['vpd'],3.)
            cfg['fixed']['incubation']['a1']=0
            self.assertEqual(load_config()['fixed']['incubation']['a1'],45.1)
            path=self.write_table(directory,'VPD_w','upper','0')
            with self.assertRaises(ValueError): load_config(path)
            path=self.write_table(directory,'theta_inc','value_json','{}')
            with self.assertRaises(ValueError): load_config(path)
            path=self.write_table(directory,'S_inc','value_json','0.9')
            with self.assertRaises(ValueError): load_config(path)

    def test_structure_specific_design(self):
        self.assertEqual([r['parameter'] for r in candidate_rows('M0')],['VPD_w','I_req'])
        self.assertEqual([r['parameter'] for r in candidate_rows('M1')],['VPD_w','I_req','u_tr'])
        d=make_design(DEFAULT_TABLE)
        self.assertEqual(len(d[d.design=='grid']),63+567)
        self.assertTrue(d.loc[d.model=='M0','u_tr'].isna().all())

    def test_expansion_preserves_baseline_grid_and_runtime_defaults(self):
        path = ROOT/'config/fsim_Ireq200_exploration.csv'
        base = make_design(DEFAULT_TABLE)
        extended = make_design(path)
        for model in ('M0', 'M1'):
            columns = ['VPD_w','I_req'] + (['u_tr'] if model == 'M1' else [])
            old = base[(base.model == model) & (base.design == 'grid')]
            new = extended[(extended.model == model) & (extended.design == 'grid')]
            self.assertEqual(len(old.merge(new, on=columns)), len(old))
            self.assertEqual(new.I_req.max(), 200)
        self.assertEqual(len(extended[extended.design == 'grid']), 1620)
        self.assertEqual(load_config(path), load_config(DEFAULT_TABLE))
        with tempfile.TemporaryDirectory() as directory:
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            row = next(r for r in rows if r['parameter'] == 'I_req')
            row['grid_values_json'] = '[45,45,200]'
            row['grid_levels'] = '3'
            invalid = Path(directory)/'invalid.csv'
            with invalid.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader(); writer.writerows(rows)
            with self.assertRaises(ValueError): load_config(invalid)

    def test_interval_contact_and_exact_date_are_distinct(self):
        l,u=pd.Timestamp('2024-05-05'),pd.Timestamp('2024-05-08')
        score=score_window(l,u,l-pd.Timedelta(days=1),l)
        self.assertEqual(score['interval_distance_days'],0)
        self.assertFalse(score['overlap'])
        score=score_window(l,u,l+pd.Timedelta(days=1),u-pd.Timedelta(days=1))
        self.assertTrue(score['overlap'])
        self.assertEqual(score['first_positive_distance_days'],1)
        score=score_window(l,u,u+pd.Timedelta(days=2),u+pd.Timedelta(days=3))
        self.assertEqual(score['signed_interval_error_days'],2)

    def test_cached_group_matches_public_model_and_zero_gate(self):
        rb=json.loads((ROOT/'data/DA_MPT_2022_grape_rb.json').read_text())
        base=load_config()
        case={'input_key':'DA_MPT_2022','L':'2022-05-05','U':'2022-05-08','rb':rb}
        pairs=[]
        for variant,wind in [('M0',np.nan),('M1',0.),('M1',8.)]:
            samples=[{'sample_id':f'{variant}_{wind}_{i}','model':variant,'design':'grid',
                      'VPD_w':v,'I_req':req,'u_tr':wind}
                     for i,(v,req) in enumerate([(1.,90.),(5.,45.)])]
            cached=evaluate_group((case,samples,base))
            for sample, record in zip(samples,cached):
                cs=CropSeasonFirst(rb,configure(base,sample));cs.simulate()
                self.assertNotEqual(record['status'],'runtime_error',record)
                if cs.primary_chain_unique.empty:
                    self.assertEqual(record['status'],'no_prediction')
                else:
                    first=cs.primary_chain_unique.sort_values(['oil_spot_start','infection_datetime']).iloc[0]
                    self.assertEqual(record['predicted_start'],first.oil_spot_start.isoformat())
                    self.assertEqual(record['predicted_end'],first.oil_spot_end.isoformat())
                    self.assertEqual(record['n_windows'],len(cs.primary_chain_unique))
            pairs.extend(cached)
        self.assertEqual(check_zero_gate(pd.DataFrame(pairs))['status'],'passed')

    def test_missing_prediction_does_not_improve_complete_case_mae(self):
        common=dict(sample_id='test',model='M0',design='grid',VPD_w=1,I_req=45,u_tr=np.nan,
                    overlap=False,window_width_days=2,first_positive_distance_days=1)
        frame=pd.DataFrame([{**common,'status':'predicted','interval_distance_days':1,'signed_interval_error_days':1},
                            {**common,'status':'no_prediction'}])
        result=summarize(frame).iloc[0]
        self.assertEqual(result.n_predictions,1)
        self.assertEqual(result.conditional_mae_days,1)
        self.assertTrue(pd.isna(result.all_site_year_mae_days))

    def test_grid_effects_recover_known_normalized_slopes(self):
        design = make_design(DEFAULT_TABLE, levels_override=3)
        design = design[design.design == 'grid'].copy()
        # An additive synthetic response with known normalized derivatives.
        design['interval_distance_days'] = (10 + 3*(design.VPD_w-1)/4
                                            - 2*(design.I_req-45)/45
                                            + design.u_tr.fillna(0)/8)
        design['predicted_midpoint_doy'] = design.interval_distance_days + 100
        design['status'] = 'predicted'
        design['input_key'] = 'synthetic'
        design['all_site_year_mae_days'] = design.interval_distance_days
        design['n_predictions'] = 6
        effects = elementary_effects(design, design, DEFAULT_TABLE)
        for row in effects.itertuples():
            expected = 0 if row.response == 'n_predictions' else {'VPD_w': 3, 'I_req': -2, 'u_tr': 1}[row.parameter]
            self.assertAlmostEqual(row.mean_effect, expected)
            self.assertAlmostEqual(row.mean_abs_effect, abs(expected))

    def test_profile_prefers_coverage_over_artificially_small_loss(self):
        frame = pd.DataFrame([
            dict(sample_id='a', model='M0', design='grid', VPD_w=1, I_req=45,
                 n_predictions=6, conditional_mae_days=10),
            dict(sample_id='b', model='M0', design='grid', VPD_w=2, I_req=45,
                 n_predictions=5, conditional_mae_days=0)])
        profiles, near, ranges = profile_and_near_optimal(frame, DEFAULT_TABLE)
        self.assertEqual(set(near.sample_id), {'a'})
        self.assertTrue((ranges.max_n_predictions == 6).all())
        self.assertFalse(profiles[(profiles.parameter == 'VPD_w') &
                                  (profiles.value == 2)].iloc[0].comparable_to_global_best)

    def test_weather_duplicate_policies_preserve_original_files(self):
        path = ROOT/'data/DA_MPT_2024_grape_rb.json'
        before = path.read_bytes()
        cases, errors, context, alternatives, repairs = load_cases(
            ROOT/'result/fsim_inputs/site_year_manifest.csv')
        self.assertEqual(errors, [])
        self.assertEqual(len(cases), 6)
        self.assertEqual(len(alternatives), 2)
        self.assertEqual(repairs.extra_rows.tolist(), [24, 24])
        for case in cases + alternatives:
            times = pd.to_datetime(pd.DataFrame(case['rb']['weather_hourly']).datetime)
            self.assertFalse(times.duplicated().any())
            self.assertTrue(times.is_monotonic_increasing)
        self.assertEqual(path.read_bytes(), before)


if __name__=='__main__': unittest.main()
