"""Analytical Sobol benchmarks, paired-design integrity and missing predictions."""
import copy
import json
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from SALib.sample import sobol
from model.parameter_table import candidate_rows,load_config
from src.sobol_fsim import (validate_settings,make_design,build_responses,analyze_response,analyze_all)

ROOT=Path(__file__).resolve().parents[1]


class SobolTests(unittest.TestCase):
    def setUp(self):
        self.s=json.loads((ROOT/'config/fsim_sobol_analysis.json').read_text())
        self.table=ROOT/self.s['parameter_table']

    def test_continuous_ranges_and_nesting(self):
        validate_settings(self.s)
        self.assertEqual(load_config(self.table)['infection']['infection_condition'],load_config()['infection']['infection_condition'])
        self.assertEqual([r['parameter'] for r in candidate_rows('M0',self.table)],['VPD_w','I_req'])
        self.assertEqual([r['parameter'] for r in candidate_rows('M1',self.table)],['VPD_w','I_req','u_tr'])
        s=copy.deepcopy(self.s);s.update(base_n=16,checkpoints=[8,16],sampling_seeds=[123])
        large,problems=make_design(s,self.table)
        s.update(base_n=8,checkpoints=[8])
        small,_=make_design(s,self.table)
        for model,p in problems.items():
            a=small[small.model==model];b=large[large.model==model].iloc[:len(a)]
            np.testing.assert_array_equal(a[p['names']],b[p['names']])
            self.assertEqual(len(a),8*(2*p['num_vars']+2))
            for name,(lo,hi) in zip(p['names'],p['bounds']):
                self.assertTrue(a[name].between(lo,hi).all())
        self.assertTrue(large.loc[large.model=='M0','u_tr'].isna().all())
        self.assertEqual(problems['M1']['bounds'],[[1.,8.],[45.,300.],[1.,8.]])

    def test_theoretical_interaction_indices(self):
        p={'num_vars':2,'names':['x','z'],'bounds':[[0,1],[0,1]]}
        x=sobol.sample(p,2048,calc_second_order=True,seed=321)
        s=copy.deepcopy(self.s);s['bootstrap_resamples']=100
        for y,expected in [(x[:,0]+2*x[:,1],{'S1':[.2,.8],'ST':[.2,.8],'S2':[0.]}),
                           (x[:,0]*x[:,1],{'S1':[3/7,3/7],'ST':[4/7,4/7],'S2':[1/7]})]:
            diag,rows=analyze_response(p,y,s,55)
            self.assertEqual(diag['status'],'ok')
            for kind,wanted in expected.items():
                actual=[r['estimate'] for r in rows if r['index_type']==kind]
                np.testing.assert_allclose(actual,wanted,atol=.02)

    def test_missing_and_constant_responses_never_fabricate_indices(self):
        p={'num_vars':2,'names':['x','z'],'bounds':[[0,1],[0,1]]}
        for y,status in [(np.full(48,np.nan),'unavailable_missing_predictions'),
                         (np.ones(48),'undefined_constant_response')]:
            diag,rows=analyze_response(p,y,self.s,1)
            self.assertEqual(diag['status'],status)
            self.assertEqual(rows,[])
        y=np.arange(48,dtype=float);y[10]=np.nan
        self.assertEqual(analyze_response(p,y,self.s,1)[0]['n_missing'],1)

    def test_fixed_denominator_and_row_order(self):
        design=pd.DataFrame([dict(sample_id='a',model='M0',seed=1,design_index=0),
                             dict(sample_id='b',model='M0',seed=1,design_index=1)])
        keys=[f'case{i}' for i in range(6)]
        rows=[]
        for sid in ('a','b'):
            for i,key in enumerate(keys):
                rows.append(dict(sample_id=sid,input_key=key,status='predicted',signed_interval_error_days=[3,-4,0,0,0,0][i]))
        rows[-1].update(status='no_prediction',signed_interval_error_days=np.nan)
        frame=pd.DataFrame(rows).sample(frac=1,random_state=42)
        output=build_responses(frame,design,keys)
        self.assertEqual(output.sample_id.tolist(),['a','b'])
        self.assertAlmostEqual(output.iloc[0]['ALL_SIX|RMSE_days'],np.sqrt(25/6))
        self.assertAlmostEqual(output.iloc[0]['ALL_SIX|MAE_days'],7/6)
        self.assertTrue(pd.isna(output.iloc[1]['ALL_SIX|RMSE_days']))
        self.assertAlmostEqual(output.iloc[1]['ALL_SIX|prediction_rate'],5/6)
        with self.assertRaises(ValueError): build_responses(frame.iloc[:-1],design,keys)
        broken=frame.copy();broken.loc[broken.index[0],'status']='runtime_error'
        with self.assertRaises(ValueError): build_responses(broken,design,keys)

    def test_one_site_missing_blocks_only_its_error_and_aggregate(self):
        s=copy.deepcopy(self.s);s.update(base_n=8,checkpoints=[8],sampling_seeds=[123],models=['M0'],bootstrap_resamples=20)
        d,problems=make_design(s,self.table)
        y=d[['sample_id','model','seed','design_index']].copy()
        y['ALL_SIX|RMSE_days']=np.arange(len(y),dtype=float)
        y.loc[10,'ALL_SIX|RMSE_days']=np.nan
        y['siteA|absolute_error_days']=d.VPD_w+2*d.I_req
        y['siteB|absolute_error_days']=np.nan
        diag,indices=analyze_all(y,problems,s)
        self.assertEqual(diag[diag.scope=='ALL_SIX'].iloc[0].status,'unavailable_missing_predictions')
        self.assertFalse(indices[indices.scope=='siteA'].empty)
        self.assertTrue(indices[indices.scope=='siteB'].empty)

    def test_accuracy_tolerance_boundary_missing_and_legacy_preservation(self):
        design=pd.DataFrame([dict(sample_id='a',model='M0',seed=1,design_index=0)])
        keys=[f'case{i}' for i in range(6)]
        errors=[-10,10,10.01,-10.01,0,np.nan]
        rows=pd.DataFrame([dict(sample_id='a',input_key=key,
                               status='no_prediction' if i==5 else 'predicted',
                               signed_interval_error_days=errors[i]) for i,key in enumerate(keys)])
        old=build_responses(rows,design,keys)
        new=build_responses(rows.sample(frac=1,random_state=2),design,keys,[10])
        pd.testing.assert_frame_equal(new[old.columns],old)
        name='accurate_prediction_rate_10d'
        self.assertEqual(new[f'ALL_SIX|{name}'].iloc[0],3/6)
        self.assertEqual([new[f'{k}|{name}'].iloc[0] for k in keys],[1,1,0,0,1,0])
        self.assertTrue(pd.isna(new['ALL_SIX|RMSE_days'].iloc[0]))
        rows.loc[5,'status']='runtime_error'
        with self.assertRaises(ValueError): build_responses(rows,design,keys,[10])
        for invalid in ([0],[-1],[np.nan],[10,10],[True]):
            s=copy.deepcopy(self.s);s['accuracy_tolerances_days']=invalid
            with self.assertRaises(ValueError): validate_settings(s)


if __name__=='__main__': unittest.main()
