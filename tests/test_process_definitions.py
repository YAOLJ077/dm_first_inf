import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from model.config import config
from model.first_infection import FirstInfection
from model.first_infection_spore_infthr import FirstInfectionSpore
from model.process_rules import INCUBATION_COMPLETION


class ProcessDefinitionTests(unittest.TestCase):
    def state(self, humidity, rain):
        times = pd.date_range("2024-04-01", periods=500, freq="h")
        weather = pd.DataFrame({
            "datetime": times, "date": times.normalize(),
            "air_temp": 20.0, "relative_humidity": humidity,
            "precip_acc_period_adjusted": rain,
        })
        return SimpleNamespace(
            weather_hourly=weather, weather_daily=pd.DataFrame(),
            bbch_15_datetime=pd.NaT,
            detached_survival=pd.DataFrame({
                "deposition_datetime": [times[0]],
                "sporangia_alive_end": [times[24]],
            }),
        )

    def test_rain_is_not_an_alternative_to_low_vpd_in_either_model(self):
        for cls in (FirstInfection, FirstInfectionSpore):
            with self.subTest(model=cls.__name__):
                state = self.state(humidity=0.0, rain=100.0)
                cfg = copy.deepcopy(config)
                # Even an obsolete supplied rain threshold cannot reopen wetness.
                cfg["zoospore_alive"]["rain"] = 0.0
                cls(state, cfg).stage_primary_chain()
                self.assertTrue(state.primary_chain.empty)

    def test_humid_rainless_conditions_use_the_same_shared_definitions(self):
        outputs = []
        for cls in (FirstInfection, FirstInfectionSpore):
            state = self.state(humidity=100.0, rain=0.0)
            model = cls(state, copy.deepcopy(config))
            with patch.object(FirstInfectionSpore, "_spore_rolling_value", return_value=0.0):
                model.stage_primary_chain()
            self.assertFalse(state.primary_chain.empty)
            outputs.append(state.primary_chain)
        pd.testing.assert_frame_equal(*outputs)

    def test_incubation_completion_is_fixed_and_old_overrides_fail(self):
        self.assertEqual(INCUBATION_COMPLETION, 1.0)
        self.assertEqual(config["incubation"]["spl_end"], 1.0)
        for cls in (FirstInfection, FirstInfectionSpore):
            cfg = copy.deepcopy(config)
            cfg["incubation"]["spl_end"] = 1.5
            with self.subTest(model=cls.__name__), self.assertRaisesRegex(ValueError, "fixed at 1"):
                cls(self.state(100.0, 0.0), cfg)


if __name__ == "__main__":
    unittest.main()
