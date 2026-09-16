"""Isolated grouped-grid tests using synthetic scores; no field calibration."""
import sys,json,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import calibrate_m0_grouped_grid as runner
import pandas as pd
import numpy as np

class GroupTests(unittest.TestCase):
    def setUp(self):
        self.s=json.loads((runner.ROOT/'config/fsim_m0_grouped_grid.json').read_text())
        self.manifest=pd.read_csv(runner.ROOT/self.s['manifest'])
        self.keys=self.manifest.input_key.tolist()

    def test_partitions_are_disjoint_and_cover_six_once(self):
        for scheme in ['site','year']:
            groups=runner.make_groups(self.manifest,scheme)
            self.assertEqual(len(groups),3)
            self.assertEqual(sorted(k for v in groups.values() for k in v),sorted(self.keys))
            column='place_code' if scheme=='site' else 'Year'
            for value,test in groups.items():
                train=self.manifest[~self.manifest.input_key.isin(test)]
                self.assertTrue(all(str(v)!=value for v in train[column]))
        self.assertEqual([len(x) for x in runner.make_groups(self.manifest,'site').values()],[3,1,2])

    def test_variable_training_count_and_missing_prediction(self):
        for n in (3,4,5):
            keys=self.keys[:n]
            records=[dict(input_key=k,sample_id='a',grid_x=0,grid_y=0,VPD_w=1,I_req=40,status='predicted',signed_interval_error_days=i+1) for i,k in enumerate(keys)]
            # A held-out row must not affect training loss.
            records.append({**records[0],'input_key':self.keys[-1],'signed_interval_error_days':999})
            score=runner.training_scores(records,keys).iloc[0]
            self.assertAlmostEqual(score.training_MAE_days,(n+1)/2)
            records[0]['status']='no_prediction';records[0]['signed_interval_error_days']=np.nan
            self.assertFalse(runner.training_scores(records,keys).iloc[0].feasible)

    def test_training_selection_precedes_whole_group_test(self):
        s={**self.s,'coarse_levels':[3,3],'refinement_rounds':1,'max_candidates_per_fold':25}
        groups=runner.make_groups(self.manifest,'site')
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);test_calls=[]
            class Cache:
                def evaluate(inner,samples,keys,label):
                    if 'held-out' in label:
                        group=label.split()[0]
                        selection=json.loads((out/'folds'/group/'selection.json').read_text())
                        self.assertEqual(keys,groups[group]);self.assertEqual(len(samples),1)
                        self.assertEqual(selection['selected_sample_id'],samples[0]['sample_id'])
                        self.assertFalse(set(keys)&set(selection['training_keys']));test_calls.append(group)
                    return [{**sample,'input_key':key,'status':'predicted',
                             'signed_interval_error_days':999 if 'held-out' in label else sample['VPD_w']}
                            for key in keys for sample in samples]
            folds,summary=runner.run_grouped([{'input_key':k} for k in self.keys],Cache(),[[1,8],[40,400]],s,out,groups,False)
            self.assertEqual(summary['MAE_days'],999)
            self.assertEqual(summary['n_folds'],3)
            self.assertEqual(summary['n_site_years'],6)
            self.assertEqual(len(test_calls),3)
            for f in folds:self.assertEqual(f['VPD_w'],1)

    def test_failure_preserves_denominator_and_group_weighting(self):
        groups=runner.make_groups(self.manifest,'site')
        rows=[dict(input_key=k,held_out=g,status='predicted',calibration_status='selected',signed_interval_error_days={'DA':0,'GL':9,'NN':3}[g]) for g,keys in groups.items() for k in keys]
        summary=runner.group_summary(pd.DataFrame(rows),self.s,groups)
        self.assertEqual(summary['MAE_days'],2.5)
        self.assertEqual(summary['equal_group_MAE_days'],4)
        for r in rows:
            if r['held_out']=='DA':r.update(status='calibration_failed',calibration_status='no_feasible_training_combination',signed_interval_error_days=np.nan)
        summary=runner.group_summary(pd.DataFrame(rows),self.s,groups)
        self.assertEqual(summary['n_calibration_failed_folds'],1)
        self.assertEqual(summary['n_calibration_failed_site_years'],3)
        self.assertEqual(summary['prediction_rate_fixed_six'],.5)
        self.assertIsNone(summary['MAE_days'])

    def test_four_process_cache_synthetic_weather(self):
        weather=pd.DataFrame(dict(datetime=pd.date_range('2024-01-01',periods=48,freq='h').astype(str),air_temp=20.,relative_humidity=50.,precip_acc_period_adjusted=0.,wind_speed=2.))
        rb=dict(year=2024,BBCH_15='2024-01-01',weather_hourly=weather.to_dict('records'))
        cases=[dict(input_key=k,L='2024-01-01',U='2024-01-02',rb=rb) for k in self.keys]
        base=runner.load_config(runner.ROOT/self.s['parameter_table']);base['dispersal']['use_wind_gate']=False
        with tempfile.TemporaryDirectory() as tmp:
            cache=runner.EvaluationCache(cases,[],base,self.s,Path(tmp),False)
            try:
                sample=runner.sample_for((0,0),(112,288),[[1,8],[40,400]])
                records=cache.evaluate([sample],self.keys[:3],'synthetic training')
                self.assertEqual(len(records),3)
                self.assertTrue(all(r['status']=='no_prediction' for r in records))
                self.assertFalse(runner.training_scores(records,self.keys[:3]).iloc[0].feasible)
            finally:cache.close()

    def test_frozen_reference_and_old_five_case_scores_match(self):
        import model.first_infection as model
        self.assertTrue(Path(model.__file__).is_relative_to(runner.REFERENCE))
        from src.calibrate_m0_losyo import training_scores as original
        parent=runner.ROOT/self.s['reference_run'];key='DA_MPT_2022'
        selection=json.loads((parent/'folds'/key/'selection.json').read_text())
        data=pd.read_csv(parent/'folds'/key/'training_predictions.csv')
        actual=runner.training_scores(data.to_dict('records'),selection['training_keys'])
        pd.testing.assert_frame_equal(actual,original(data.to_dict('records'),selection['training_keys']))

if __name__=='__main__':unittest.main()
