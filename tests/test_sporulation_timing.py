import math
import unittest
from types import SimpleNamespace

import pandas as pd
import numpy as np

from model.first_infection import FirstInfection
from model.first_infection_spore_infthr import FirstInfectionSpore
from model.sporulation_timing import (
    lalancette_sporangia,
    normalized_lalancette_sporulation,
    sporulation_dose_for_day,
    summarize_sporulation_nights,
)


class SporulationTimingTests(unittest.TestCase):
    def test_rossi_release_wetness_equation_parentheses(self):
        for model in (FirstInfection, FirstInfectionSpore):
            # At 20 C the published requirement is about 0.96 h, not 2.54 h.
            self.assertTrue(model._release_condition(1, 20))
            for temperature in (1., 5., 10., 15., 20., 25., 30.):
                threshold = math.exp(-1.022 + 19.634 / temperature)
                self.assertFalse(model._release_condition(threshold * 0.999, temperature))
                self.assertTrue(model._release_condition(threshold * 1.001, temperature))

    def test_release_temperature_boundaries_without_numeric_warnings(self):
        temperatures = [-5., 0., 0.01, np.nextafter(0., 1.), np.nan, np.inf, 20.]
        with np.errstate(all="raise"):
            for model in (FirstInfection, FirstInfectionSpore):
                np.testing.assert_array_equal(
                    model._release_condition(1., temperatures),
                    [False, False, False, False, False, False, True],
                )
                self.assertFalse(model._release_condition(0., 20.))

    def test_lalancette_reference_is_normalized_to_one(self):
        temperature = 20.0
        moist_hours = 24.0
        asymptote = (
            -873_164.0
            + 114_369.0 * temperature
            - 2_828.0 * temperature**2
            + 50_000.0
        )
        intercept = -8.96 + 1.16 * temperature - 0.029 * temperature**2
        duration_term = moist_hours * (
            1.51 - 0.20 * temperature + 0.0049 * temperature**2
        )
        expected = asymptote * (1.0 - math.exp(intercept + duration_term)) ** 2

        self.assertAlmostEqual(
            lalancette_sporangia(temperature, moist_hours), expected
        )
        self.assertAlmostEqual(
            normalized_lalancette_sporulation(temperature, moist_hours), 1.0
        )

    def test_lalancette_response_increases_with_moist_period_at_20c(self):
        dose_8h = normalized_lalancette_sporulation(20.0, 8.0)
        dose_12h = normalized_lalancette_sporulation(20.0, 12.0)

        self.assertGreater(dose_8h, 0.0)
        self.assertLess(dose_8h, dose_12h)
        self.assertLess(dose_12h, 1.0)

    def test_brischetto_sporulation_switch_is_fixed(self):
        self.assertEqual(normalized_lalancette_sporulation(20.0, 2.99), 0.0)
        self.assertEqual(normalized_lalancette_sporulation(10.0, 12.0), 0.0)
        self.assertEqual(normalized_lalancette_sporulation(30.0, 12.0), 0.0)

    def test_daily_dose_has_no_repeated_event_decay(self):
        day = SimpleNamespace(TMP=20.0, MP=12.0)

        self.assertEqual(sporulation_dose_for_day(day), sporulation_dose_for_day(day))

    def test_night_is_grouped_by_its_ending_date(self):
        weather = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    [
                        "2024-01-01 20:00",
                        "2024-01-01 23:00",
                        "2024-01-02 00:00",
                        "2024-01-02 07:00",
                        "2024-01-02 20:00",
                    ]
                ),
                "relative_humidity": [90, 90, 90, 90, 90],
                "precip_acc_period_adjusted": [0, 0, 0, 0, 0],
                "air_temp": [20, 20, 20, 20, 22],
            }
        )

        result = summarize_sporulation_nights(weather)

        self.assertEqual(result.loc[0, "production_date"], pd.Timestamp("2024-01-02"))
        self.assertEqual(result.loc[0, "MP"], 4)
        self.assertEqual(result.loc[1, "production_date"], pd.Timestamp("2024-01-03"))
        self.assertEqual(result.loc[1, "MP"], 1)

    def test_rain_or_leaf_wetness_can_define_a_moist_hour(self):
        weather = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2024-01-01 20:00", "2024-01-01 21:00", "2024-01-01 22:00"]
                ),
                "relative_humidity": [60, 60, 80],
                "precip_acc_period_adjusted": [0.1, 0, 0],
                "leaf_wetness_minutes": [0, 31, 0],
                "air_temp": [20, 20, 20],
            }
        )

        result = summarize_sporulation_nights(weather)

        self.assertEqual(result.loc[0, "MP"], 3)

    def test_completed_night_dose_is_not_available_before_0800(self):
        datetimes = pd.date_range("2024-01-01 00:00", "2024-01-02 11:00", freq="h")
        weather = pd.DataFrame(
            {
                "datetime": datetimes,
                "air_temp": 20.0,
                "relative_humidity": 50.0,
                "precip_acc_period_adjusted": 0.0,
                "wind_speed": 2.0,
            }
        )
        night = (weather["datetime"] >= pd.Timestamp("2024-01-01 20:00")) & (
            weather["datetime"] < pd.Timestamp("2024-01-02 08:00")
        )
        weather.loc[night, "relative_humidity"] = 90.0
        weather["date"] = weather["datetime"].dt.normalize()
        daily = weather.groupby("date", as_index=False).agg(datetime=("date", "first"))
        state = SimpleNamespace(weather_hourly=weather, weather_daily=daily)
        config = {
            "mp_tmp": {"rh_wet_night": 80.0},
            "sporulation": {},
            "dispersal": {"use_wind_gate": True, "wind_min": 1.0},
        }

        FirstInfection(state, config).stage_attached_and_dispersal()

        self.assertFalse(state.dispersal_cohorts.empty)
        self.assertEqual(
            state.dispersal_cohorts.iloc[0]["dispersal_datetime"],
            pd.Timestamp("2024-01-02 08:00"),
        )

    def test_production_events_remain_independent_until_release(self):
        datetimes = pd.date_range("2024-01-01 00:00", "2024-01-03 12:00", freq="h")
        weather = pd.DataFrame(
            {
                "datetime": datetimes,
                "air_temp": 20.0,
                "relative_humidity": 50.0,
                "precip_acc_period_adjusted": 0.0,
                "wind_speed": 0.0,
            }
        )
        for production_date in (pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")):
            night = (weather["datetime"] >= production_date - pd.Timedelta(hours=4)) & (
                weather["datetime"] < production_date + pd.Timedelta(hours=8)
            )
            weather.loc[night, "relative_humidity"] = 90.0
        weather.loc[weather["datetime"] >= pd.Timestamp("2024-01-03 08:00"), "wind_speed"] = 2.0
        weather["date"] = weather["datetime"].dt.normalize()
        daily = weather.groupby("date", as_index=False).agg(datetime=("date", "first"))
        state = SimpleNamespace(weather_hourly=weather, weather_daily=daily)
        config = {
            "mp_tmp": {"rh_wet_night": 80.0},
            "sporulation": {},
            "dispersal": {"use_wind_gate": True, "wind_min": 1.0},
        }

        FirstInfection(state, config).stage_attached_and_dispersal()

        released = state.dispersal_cohorts
        release_time = pd.Timestamp("2024-01-03 08:00")
        same_release = released[released["dispersal_datetime"].eq(release_time)]
        self.assertEqual(len(same_release), 2)
        self.assertEqual(same_release["production_datetime"].nunique(), 2)
        older = same_release.sort_values("production_datetime").iloc[0]
        newer = same_release.sort_values("production_datetime").iloc[1]
        self.assertGreater(
            older["attached_mortality_cumulative"],
            newer["attached_mortality_cumulative"],
        )
        self.assertLess(older["airborne_amount"], newer["airborne_amount"])


if __name__ == "__main__":
    unittest.main()
