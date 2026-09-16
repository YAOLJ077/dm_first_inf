"""RH candidates are selected exclusively from training loss, with a fixed tie rule."""
import copy
import json
import unittest
import tempfile
from pathlib import Path
import pandas as pd
from model.parameter_table import load_config
from src.analyze_fsim_parameters import configure
from src.calibrate_m0_rh_trial import choose_rh_from_training,validate_trial,combine_runs
from src.calibrate_m0_losyo import summarize_oof

ROOT=Path(__file__).resolve().parents[1]


class RHTrialTests(unittest.TestCase):
    def setUp(self):
        self.s=json.loads((ROOT/'config/fsim_m0_losyo_refine3.json').read_text())
        self.trial=json.loads((ROOT/'config/fsim_m0_losyo_rh_trial.json').read_text())

    def surface(self,loss,feasible=True):
        return pd.DataFrame([dict(sample_id='a',VPD_w=2,I_req=200,feasible=feasible,training_MAE_days=loss),
                             dict(sample_id='b',VPD_w=3,I_req=150,feasible=feasible,training_MAE_days=loss)])

    def test_training_loss_and_infeasibility_control_selection(self):
        choice,_,_=choose_rh_from_training({80:self.surface(10),85:self.surface(9),90:self.surface(1,False)},self.s)
        self.assertEqual(choice.RH_sp,85);self.assertEqual(choice.sample_id,'a')
        choice,_,_=choose_rh_from_training({80:self.surface(10),85:self.surface(10)},self.s)
        self.assertEqual(choice.RH_sp,80)
        choice,_,_=choose_rh_from_training({75:self.surface(8),80:self.surface(10),85:self.surface(8)},self.s)
        self.assertEqual(choice.RH_sp,75)
        choice,_,_=choose_rh_from_training({80:self.surface(float('nan'),False)},self.s)
        self.assertIsNone(choice)

    def test_runtime_override_survives_sample_configuration(self):
        base=load_config(ROOT/'config/fsim_m0_losyo_parameters.csv')
        altered=copy.deepcopy(base);altered['mp_tmp']['rh_wet_night']=85
        actual=configure(altered,dict(model='M0',VPD_w=2,I_req=200,u_tr=None))
        self.assertEqual(actual['mp_tmp']['rh_wet_night'],85)
        self.assertEqual(base['mp_tmp']['rh_wet_night'],80)
        self.assertFalse(actual['dispersal']['use_wind_gate'])
        self.assertEqual(actual['fixed'],base['fixed'])

    def test_trial_validation(self):
        validate_trial(self.trial)
        for values in [[75,85],[80,80],[80,101],[80,float('nan')]]:
            trial={**self.trial,'RH_sp_values':values}
            with self.assertRaises(ValueError):validate_trial(trial)

    def test_combination_does_not_choose_using_better_heldout_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);runs={80:root/'baseline',85:root/'trial'}
            keys=[f'case{i}' for i in range(6)]
            for rh,path in runs.items():
                path.mkdir()
                pd.DataFrame({'input_key':keys}).to_csv(path/'site_year_manifest.csv',index=False)
                records=[]
                for key in keys:
                    fold=path/'folds'/key;fold.mkdir(parents=True)
                    self.surface(10 if rh==80 else 9).to_csv(fold/'loss_surface.csv',index=False)
                    # RH=85 is preferable in training but deliberately much worse
                    # on every held-out target. It must still be the chosen value.
                    record=dict(input_key=key,sample_id='a',VPD_w=2,I_req=200,
                                status='predicted',calibration_status='selected',
                                signed_interval_error_days=0 if rh==80 else 999)
                    records.append(record)
                    pd.DataFrame([record]).to_csv(fold/'held_out_prediction.csv',index=False)
                frame=pd.DataFrame(records);frame.to_csv(path/'oof_predictions.csv',index=False)
                (path/'oof_summary.json').write_text(json.dumps(summarize_oof(frame,self.s)))
                pd.DataFrame({'held_out':keys,'best_training_MAE_days':[10 if rh==80 else 9]*6}).to_csv(path/'fold_summary.csv',index=False)
            out=root/'combined';out.mkdir()
            summary=combine_runs(runs,self.s,self.trial,out)
            self.assertEqual(summary['MAE_days'],999)
            self.assertTrue(pd.read_csv(out/'fold_summary.csv').RH_sp.eq(85).all())


if __name__=='__main__':unittest.main()
