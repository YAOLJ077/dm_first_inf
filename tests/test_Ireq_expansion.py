"""Range-comparison safeguards using synthetic diagnostic scores."""
import unittest
import numpy as np
import pandas as pd
from src.compare_fsim_Ireq_expansion import (
    select_best, compare_at_fixed_other_parameters, boundary_checks,
)


class ExpansionComparisonTests(unittest.TestCase):
    def metrics(self):
        rows=[]
        for model in ('M0','M1'):
            for i, (req, n, loss) in enumerate([(45,6,10),(90,6,9),(120,6,7),(150,5,1),(180,6,8),(200,5,0)]):
                rows.append(dict(sample_id=f'{model}_{i}',model=model,design='grid',
                                 VPD_w=1.5,I_req=req,u_tr=np.nan if model=='M0' else 2.,
                                 n_errors=0,n_predictions=n,conditional_mae_days=loss))
        return pd.DataFrame(rows)

    def test_lower_coverage_cannot_win_by_having_smaller_error(self):
        metrics=self.metrics()
        best=select_best(metrics[metrics.model=='M0'])
        self.assertEqual(best.I_req,120)
        self.assertEqual(best.n_predictions,6)
        boundaries=boundary_checks({'metrics':metrics})
        row=boundaries[(boundaries.model=='M0') & (boundaries.upper_cap==200)].iloc[0]
        self.assertEqual(row.best_I_req,120)
        self.assertFalse(row.best_at_cap)

    def test_fixed_other_parameter_comparison_is_nested(self):
        extended=self.metrics()
        old=extended[extended.I_req<=90]
        comparison=compare_at_fixed_other_parameters({'metrics':old},{'metrics':extended})
        self.assertTrue((comparison.baseline_I_req==90).all())
        self.assertTrue((comparison.extended_I_req==120).all())
        self.assertTrue((comparison.mae_reduction_days==2).all())
        self.assertTrue(comparison.both_complete.all())


if __name__=='__main__': unittest.main()
