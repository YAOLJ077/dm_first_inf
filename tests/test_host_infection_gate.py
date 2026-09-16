import copy
import unittest
from types import SimpleNamespace

import pandas as pd

from model.config import config
from model.first_infection import FirstInfection


class HostInfectionGateTests(unittest.TestCase):
    def run_chain(self, gate, dry_hour=None):
        times = pd.date_range("2024-04-01", periods=500, freq="h")
        weather = pd.DataFrame({"datetime": times, "date": times.normalize(),
                                "air_temp": 20., "relative_humidity": 100.,
                                "precip_acc_period_adjusted": 0.})
        if dry_hour is not None:
            weather.loc[dry_hour, "relative_humidity"] = 0.
        state = SimpleNamespace(
            weather_hourly=weather, weather_daily=pd.DataFrame(),
            bbch_15_datetime=gate,
            detached_survival=pd.DataFrame({
                "deposition_datetime": [times[0]],
                "sporangia_alive_end": [times[24]],
            }),
        )
        cfg = copy.deepcopy(config)
        cfg["infection"]["infection_condition"] = 60.
        FirstInfection(state, cfg).stage_primary_chain()
        return state.primary_chain

    def test_pre_gate_wetness_does_not_contribute(self):
        gate = pd.Timestamp("2024-04-01 06:00")
        result = self.run_chain(gate)
        self.assertEqual(result.iloc[0].infection_datetime, gate + pd.Timedelta(hours=2))
        self.assertLess(result.iloc[0].zoospore_release_datetime, gate)

    def test_zoospores_dead_before_gate_cannot_infect(self):
        self.assertTrue(self.run_chain(pd.Timestamp("2024-04-01 06:00"), 4).empty)

    def test_missing_gate_is_explicit_ungated_case(self):
        result = self.run_chain(pd.NaT)
        self.assertEqual(result.iloc[0].infection_datetime, pd.Timestamp("2024-04-01 02:00"))
