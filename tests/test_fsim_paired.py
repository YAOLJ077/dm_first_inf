"""Exact matching and fixed-denominator structural contrasts."""
import unittest
import numpy as np
import pandas as pd
from src.compare_fsim_paired import common_design
from src.report_fsim_paired import build_pairs,delta_responses,summarize_pairs


class PairedTests(unittest.TestCase):
    def fixture(self):
        design=pd.DataFrame([dict(sample_id=f's{i}',model='M1',seed=1,design_index=i,
                                  VPD_w=2.,I_req=100.,u_tr=u) for i,u in enumerate([2.,6.])])
        common,mapping=common_design(design);keys=[f'case{i}' for i in range(6)]
        base=[];gated=[]
        for i,key in enumerate(keys):
            base.append(dict(sample_id=common.common_id.iloc[0],input_key=key,VPD_w=2.,I_req=100.,
                             status='predicted' if i<4 else 'no_prediction',
                             signed_interval_error_days=[7,8,11,0,np.nan,np.nan][i],
                             predicted_start='2022-05-01' if i<4 else None,
                             predicted_end='2022-05-03' if i<4 else None))
            for j,u in enumerate([2.,6.]):
                available=i in [0,1,2,4]
                gated.append(dict(sample_id=f's{j}',input_key=key,VPD_w=2.,I_req=100.,u_tr=u,
                                  status='predicted' if available else 'no_prediction',
                                  signed_interval_error_days=[8,7,10,np.nan,7,np.nan][i],
                                  predicted_start='2022-05-02' if available else None,
                                  predicted_end='2022-05-04' if available else None))
        return common,mapping,keys,pd.DataFrame(base),pd.DataFrame(gated)

    def test_deduplication_keeps_original_saltelli_rows(self):
        common,mapping,keys,m0,m1=self.fixture()
        self.assertEqual(len(common),1);self.assertEqual(len(mapping),2)
        pairs=build_pairs(mapping,m0,m1.sample(frac=1,random_state=4))
        self.assertEqual(len(pairs),12)
        self.assertEqual(pairs.groupby('sample_id').size().tolist(),[6,6])
        wrong=m0.copy();wrong.loc[0,'I_req']=101
        with self.assertRaises(ValueError):build_pairs(mapping,wrong,m1)
        with self.assertRaises(ValueError):build_pairs(mapping,m0,m1.iloc[:-1])
        near=mapping[['sample_id','model','seed','design_index','VPD_w','I_req','u_tr']].copy()
        near.loc[1,'VPD_w']=np.nextafter(2.,3.)
        self.assertEqual(len(common_design(near)[0]),2)

    def test_accuracy_transitions_do_not_impute_missing_error(self):
        _,mapping,keys,m0,m1=self.fixture()
        pairs=build_pairs(mapping,m0,m1)
        responses=delta_responses(pairs,mapping,keys)
        np.testing.assert_allclose(responses['ALL_SIX|availability_delta'],0)
        np.testing.assert_allclose(responses['ALL_SIX|accuracy_delta_7d'],0)
        np.testing.assert_allclose(responses['ALL_SIX|accuracy_delta_10d'],1/6)
        self.assertTrue(pairs.loc[pairs.availability_transition.ne('both'),'absolute_error_delta_days'].isna().all())
        self.assertTrue(pairs.loc[pairs.availability_transition.eq('both'),'onset_shift_days'].eq(1).all())
        summary,_=summarize_pairs(pairs,keys,2)
        row=summary.query('sample_set=="base_AB" and scope=="ALL_SIX"').iloc[0]
        self.assertAlmostEqual(row.availability_lost_rate,1/6)
        self.assertAlmostEqual(row.availability_gained_rate,1/6)
        self.assertAlmostEqual(row.accuracy_delta_10d,row.gained_accuracy_10d_rate-row.lost_accuracy_10d_rate)
        bad=m1.copy();bad.loc[0,'status']='runtime_error'
        with self.assertRaises(ValueError):build_pairs(mapping,m0,bad)


if __name__=='__main__':unittest.main()
