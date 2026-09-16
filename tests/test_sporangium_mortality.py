import math
import unittest
from types import SimpleNamespace

import pandas as pd

from model.first_infection import FirstInfection
from model.first_infection_spore_infthr import FirstInfectionSpore


class SporangiumMortalityTests(unittest.TestCase):
    def test_physical_vpd_and_brischetto_mortality_index_are_distinct(self):
        datetimes = pd.date_range("2024-01-01", periods=1, freq="h")
        weather = pd.DataFrame(
            {
                "datetime": datetimes,
                "date": datetimes.normalize(),
                "air_temp": 20.0,
                "relative_humidity": 80.0,
                "precip_acc_period_adjusted": 0.0,
            }
        )
        state = SimpleNamespace(
            weather_hourly=weather,
            weather_daily=pd.DataFrame({"datetime": datetimes.normalize()}),
        )

        model = FirstInfection(state, {})

        expected_vpd_hpa = 6.11 * math.exp(17.47 * 20.0 / (239.0 + 20.0)) * 0.2
        self.assertAlmostEqual(model.VPD[0], expected_vpd_hpa)
        self.assertAlmostEqual(model.sporangium_mortality_index[0], 20.0 * 0.2)

    def test_attached_rate_matches_brischetto_equation(self):
        coefficients = {"a0": 9.27, "a1": 1.12, "a2": 0.04}
        expected = 1.0 / (24.0 * (9.27 - 1.12 * 10.0 + 0.04 * 10.0**2))

        self.assertAlmostEqual(
            FirstInfection._hourly_mortality_rate(10.0, coefficients), expected
        )

    def test_detached_rate_matches_brischetto_equation(self):
        coefficients = {"b0": 5.67, "b1": 0.47, "b2": 0.02}
        expected = 1.0 / (24.0 * (5.67 - 0.47 * 10.0 + 0.02 * 10.0**2))

        self.assertAlmostEqual(
            FirstInfection._hourly_mortality_rate(10.0, coefficients), expected
        )

    def test_state_update_uses_cumulative_mortality(self):
        coefficients = {"b0": 5.67, "b1": 0.47, "b2": 0.02}
        rate = FirstInfection._hourly_mortality_rate(10.0, coefficients)

        cumulative, amount = FirstInfection._advance_mortality(
            0.8, 0.25, 10.0, coefficients
        )
        self.assertAlmostEqual(cumulative, 0.25 + rate)
        self.assertAlmostEqual(amount, 0.8 * (1.0 - cumulative))

    def test_cumulative_mortality_reaches_zero_without_dose_cutoff(self):
        self.assertEqual(FirstInfection._remaining_amount(0.8, 1.0), 0.0)
        self.assertEqual(FirstInfection._remaining_amount(0.8, 1.2), 0.0)

    def test_fsim_and_fsim_s_use_identical_rates(self):
        for coefficients in (
            {"a0": 9.27, "a1": 1.12, "a2": 0.04},
            {"b0": 5.67, "b1": 0.47, "b2": 0.02},
        ):
            for vpd in (0.0, 1.0, 5.0, 10.0, 20.0):
                self.assertAlmostEqual(
                    FirstInfection._hourly_mortality_rate(vpd, coefficients),
                    FirstInfectionSpore._hourly_mortality_rate(vpd, coefficients),
                )
                self.assertEqual(
                    FirstInfection._remaining_amount(1.0, 0.4),
                    FirstInfectionSpore._remaining_amount(1.0, 0.4),
                )

    def test_merged_deposition_has_no_separate_delay_or_mortality_reset(self):
        datetimes = pd.date_range("2024-01-01", periods=150, freq="h")
        weather = pd.DataFrame(
            {
                "datetime": datetimes,
                "date": datetimes.normalize(),
                "air_temp": 20.0,
                "relative_humidity": 80.0,
                "precip_acc_period_adjusted": 0.0,
            }
        )
        weather.loc[3, "precip_acc_period_adjusted"] = 2.0
        state = SimpleNamespace(
            weather_hourly=weather,
            weather_daily=pd.DataFrame({"datetime": [datetimes[0].normalize()]}),
            dispersal_cohorts=pd.DataFrame(
                {
                    "Num": [0],
                    "dispersal_datetime": [datetimes[0]],
                    "dispersal_type": ["wind"],
                    "airborne_amount": [1.0],
                }
            ),
        )
        model = FirstInfection(state, {})
        hourly_rate = model._hourly_mortality_rate(
            model.sporangium_mortality_index[0], model._DETACHED_MORTALITY
        )

        model.stage_deposition()
        deposited = state.deposition.iloc[0]
        self.assertAlmostEqual(
            deposited["detached_mortality_cumulative_at_deposition"],
            0.0,
        )
        self.assertAlmostEqual(
            deposited["deposited_amount"], 1.0
        )

        self.assertEqual(deposited["deposition_datetime"], datetimes[0])
        # Check that the survival stage consumes supplied cumulative mortality
        # rather than restarting it, even when an aged state is supplied.
        state.deposition["detached_mortality_cumulative_at_deposition"] = 1 - hourly_rate / 2
        model.stage_detached_survival()
        survived = state.detached_survival.iloc[0]
        self.assertEqual(survived["detached_mortality_cumulative_at_end"], 1.0)
        self.assertEqual(survived["sporangia_alive_end"], datetimes[0])


if __name__ == "__main__":
    unittest.main()
