"""Training-only selection, feasibility and disconnected grid-refinement checks."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from model.parameter_table import candidate_rows,load_config
from src.calibrate_m0_losyo import (validate_settings,training_scores,best_and_ties,
    make_lattice,sample_for,refine_cells,run_losyo,summarize_oof)
ROOT=Path(__file__).resolve().parents[1]


class ToyCache:
    def __init__(self,out,keys,changed=None):self.out=out;self.keys=keys;self.changed=changed;self.calls=[]
    def evaluate(self,samples,keys,label):
        self.calls.append((list(keys),label))
        if label.endswith('held-out'):
            path=self.out/'folds'/keys[0]/'selection.json'
            assert path.exists(), 'Selection must be frozen before held-out evaluation'
        result=[]
        for key in keys:
            for s in samples:
                # Site 0 prefers the opposite side; changing only its target must
                # not affect selection in the fold where it is held out.
                target=7 if key==self.changed else 2
                e=abs(s['VPD_w']-target)+abs(s['I_req']-100)/100
                result.append({**s,'input_key':key,'status':'predicted','signed_interval_error_days':e})
        return result


class LOSYOTests(unittest.TestCase):
    def setUp(self):
        self.s=json.loads((ROOT/'config/fsim_m0_losyo.json').read_text())
        self.keys=[f'case{i}' for i in range(6)]
        self.bounds=[[1,8],[45,300]]

    def test_feasibility_ties_and_heldout_filter(self):
        rows=[]
        for x in [1,2]:
            for key in self.keys:
                rows.append(dict(sample_id=str(x),grid_x=x,grid_y=0,VPD_w=x,I_req=100,
                                 input_key=key,status='predicted',signed_interval_error_days=-2 if key!='case5' else 100*x))
        scores=training_scores(rows,self.keys[:5]);best,ties=best_and_ties(scores,1e-9)
        self.assertEqual(best,2);self.assertEqual(ties.VPD_w.tolist(),[1,2])
        rows[0]['status']='no_prediction';rows[0]['signed_interval_error_days']=np.nan
        scores=training_scores(rows,self.keys[:5])
        self.assertFalse(scores.iloc[0].feasible);self.assertTrue(np.isnan(scores.iloc[0].training_MAE_days))
        self.assertEqual(best_and_ties(scores,1e-9)[1].VPD_w.tolist(),[2])
        rows[0]['status']='runtime_error'
        with self.assertRaises(ValueError):training_scores(rows,self.keys[:5])

    def test_disconnected_regions_and_no_feasible_refinement(self):
        s={**self.s,'coarse_levels':[3,3],'refinement_rounds':1}
        extent,points,cells=make_lattice(self.bounds,s)
        scores=pd.DataFrame([{'grid_x':x,'grid_y':y,'VPD_w':x,'I_req':y,'feasible':True,'training_MAE_days':0 if (x,y) in [(0,0),(4,4)] else 10}
                             for x,y in points])
        new,leaves,n=refine_cells(cells,scores,s)
        self.assertIn((1,1),new);self.assertIn((3,3),new);self.assertEqual(n,2)
        scores['feasible']=False;scores['training_MAE_days']=np.nan
        new,_,n=refine_cells(cells,scores,s)
        self.assertEqual(n,4);self.assertEqual(len(new),25)
        self.assertEqual(sample_for((4,4),extent,self.bounds)['I_req'],300)

    def test_six_folds_freeze_selection_and_no_heldout_target_leakage(self):
        s={**self.s,'coarse_levels':[3,3],'refinement_rounds':1,'max_candidates_per_fold':25}
        selections=[]
        for changed in [None,'case0']:
            with tempfile.TemporaryDirectory() as d:
                out=Path(d);cache=ToyCache(out,self.keys,changed)
                folds,summary=run_losyo([{'input_key':k} for k in self.keys],cache,self.bounds,s,out,False)
                self.assertEqual(summary['n_folds'],6);self.assertEqual(summary['n_calibration_failures'],0)
                self.assertEqual(len(folds),6)
                for keys,label in cache.calls:
                    held=label.split()[0]
                    if 'training' in label:
                        self.assertEqual(len(keys),5);self.assertNotIn(held,keys)
                    else:self.assertEqual(keys,[held])
                selections.append(json.loads((out/'folds/case0/selection.json').read_text()))
        self.assertEqual(selections[0],selections[1])

    def test_oof_missing_fixed_denominator(self):
        rows=[dict(calibration_status='selected',status='predicted',signed_interval_error_days=3.) for _ in range(6)]
        rows[-1].update(status='no_prediction',signed_interval_error_days=np.nan)
        summary=summarize_oof(pd.DataFrame(rows),self.s)
        self.assertIsNone(summary['MAE_days']);self.assertEqual(summary['conditional_MAE_days'],3)
        self.assertEqual(summary['accurate_prediction_rate_7d_fixed_six'],5/6)
        self.assertEqual(summary['n_no_predictions'],1)
        validate_settings(self.s)

    def test_configured_grid_range_and_refinement_spacing(self):
        table=ROOT/self.s['parameter_table']
        load_config(table)
        bounds=[[float(r['lower']),float(r['upper'])] for r in candidate_rows('M0',table)]
        self.assertEqual(bounds,[[1,8],[40,400]])
        extent,points,_=make_lattice(bounds,self.s)
        self.assertEqual(len(points),555)
        self.assertEqual((extent[0]+1)*(extent[1]+1),8265)
        self.assertGreaterEqual(self.s['max_candidates_per_fold'],8265)
        for stage,expected in enumerate([[.5,10],[.25,5],[.125,2.5]]):
            spacing=[(hi-lo)/((n-1)*self.s['refinement_factor']**stage)
                     for (lo,hi),n in zip(bounds,self.s['coarse_levels'])]
            np.testing.assert_allclose(spacing,expected)

    def test_no_feasible_fold_never_requests_heldout(self):
        s={**self.s,'coarse_levels':[3,3],'refinement_rounds':1,'max_candidates_per_fold':25}
        class MissingCache(ToyCache):
            def evaluate(self,samples,keys,label):
                assert 'held-out' not in label
                rows=super().evaluate(samples,keys,label)
                for r in rows:
                    r.update(status='no_prediction',signed_interval_error_days=np.nan)
                return rows
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);cache=MissingCache(out,self.keys)
            folds,summary=run_losyo([{'input_key':k} for k in self.keys],cache,self.bounds,s,out,False)
            self.assertEqual(summary['n_calibration_failures'],6)
            self.assertEqual(summary['n_no_predictions'],0)
            self.assertIsNone(summary['MAE_days'])
            self.assertTrue(all(f['n_candidates']==25 for f in folds))

    def test_budget_does_not_silently_take_partial_refinement_region(self):
        s={**self.s,'coarse_levels':[3,3],'refinement_rounds':1,'max_candidates_per_fold':9}
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);cache=ToyCache(out,self.keys)
            folds,_=run_losyo([{'input_key':k} for k in self.keys],cache,self.bounds,s,out,False)
            self.assertTrue(all(f['stopping_reason']=='candidate_budget' for f in folds))
            self.assertTrue(all(f['n_candidates']==9 for f in folds))


if __name__=='__main__':unittest.main()
