"""Bookkeeping tests with synthetic doses and arbitrary gate values."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from model.first_infection import FirstInfection
from model.crop_season import CropSeasonFirst
from src.audit_fsim_dispersal_scenarios import SCENARIOS, run_scenario


class MergedTransferTests(unittest.TestCase):
    def make_model(self, use_wind_gate=False, wind_min=None, rain=0.0):
        times = pd.date_range("2024-01-01", periods=24, freq="h")
        weather = pd.DataFrame({
            "datetime": times,
            "date": times.normalize(),
            "air_temp": 20.0,
            "relative_humidity": 80.0,
            "precip_acc_period_adjusted": rain,
            "wind_speed": 0.0,
        })
        # A gate event before dose availability, and one after availability.
        weather.loc[[2, 14], "wind_speed"] = 2.0
        state = SimpleNamespace(
            weather_hourly=weather,
            weather_daily=pd.DataFrame({"datetime": [times[0]]}),
        )
        # No rain-transfer or deposition configuration is provided.
        cfg = {"mp_tmp": {"rh_wet_night": 80.0},
               "dispersal": {"use_wind_gate": use_wind_gate, "wind_min": wind_min}}
        model = FirstInfection(state, cfg)
        return model, state

    def transfer(self, model, dose=1.0):
        with patch("model.first_infection.sporulation_dose_for_day", return_value=dose):
            model.stage_attached_and_dispersal()
        model.stage_deposition()

    def test_only_two_candidates_and_obsolete_mode_rejected(self):
        self.assertEqual(
            {s["use_wind_gate"] for s in SCENARIOS.values()},
            {False, True},
        )
        self.assertEqual(len(SCENARIOS), 2)
        with self.assertRaises(ValueError):
            self.make_model(use_wind_gate="legacy")

    def test_no_transfer_before_availability_and_no_deposition_delay(self):
        for mode, threshold, expected_hour in ((False, None, 8), (True, 2.0, 14)):
            with self.subTest(mode=mode):
                model, state = self.make_model(mode, threshold)
                self.transfer(model)
                self.assertEqual(len(state.deposition), 1)
                cohort = state.deposition.iloc[0]
                self.assertEqual(cohort.dispersal_datetime.hour, expected_hour)
                self.assertEqual(cohort.deposition_datetime, cohort.dispersal_datetime)
                self.assertAlmostEqual(cohort.deposited_amount, cohort.airborne_amount)
                rate = model._hourly_mortality_rate(
                    model.sporangium_mortality_index[8], model._ATTACHED_MORTALITY
                )
                # The transfer hour is NOT also charged to attached mortality.
                self.assertAlmostEqual(
                    cohort.attached_mortality_cumulative, (expected_hour - 8) * rate
                )
                self.assertAlmostEqual(
                    cohort.deposited_amount,
                    cohort.initial_sporangia_amount * (1 - cohort.attached_mortality_cumulative),
                )
                self.assertEqual(cohort.detached_mortality_cumulative_at_deposition, 0)
                model.stage_detached_survival()
                self.assertGreater(
                    state.detached_survival.iloc[0].detached_mortality_cumulative_at_end, 0
                )

    def test_rain_cannot_trigger_or_delay_transfer_of_identical_available_doses(self):
        for mode, threshold in ((False, None), (True, 2.0)):
            outputs = []
            for rain in (0.0, 100.0):
                model, state = self.make_model(mode, threshold, rain)
                self.transfer(model)
                outputs.append(state.deposition)
            pd.testing.assert_frame_equal(*outputs)

    def test_absent_or_dead_inventory_cannot_transfer(self):
        model, state = self.make_model()
        self.transfer(model, dose=0.0)
        self.assertTrue(state.deposition.empty)
        model, state = self.make_model(True, 2.0)
        with patch.object(FirstInfection, "_hourly_mortality_rate", return_value=1.0):
            self.transfer(model)
        self.assertTrue(state.deposition.empty)

    def test_always_open_gate_matches_ungated_candidate(self):
        plain, plain_state = self.make_model()
        gated, gated_state = self.make_model(True, 0.0)
        self.transfer(plain)
        self.transfer(gated)
        pd.testing.assert_frame_equal(
            plain_state.deposition.drop(columns="dispersal_type"),
            gated_state.deposition.drop(columns="dispersal_type"),
        )

    def test_invalid_wind_and_threshold_are_not_silently_accepted(self):
        for threshold in (None, -1, np.nan, np.inf):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                model, _ = self.make_model(True, threshold)
                self.transfer(model)
        _, state = self.make_model()
        state.weather_hourly.loc[0, "wind_speed"] = np.nan
        with self.assertRaises(ValueError):
            model = FirstInfection(state, {"dispersal": {"wind_min": 2.0}})
            model.stage_attached_and_dispersal()

    def test_public_entry_and_candidate_share_caller_config(self):
        model, state = self.make_model(True, 2.0)
        rb = {"weather_hourly": state.weather_hourly.drop(columns="date").assign(
            datetime=lambda df: df.datetime.astype(str)
        ).to_dict("records")}
        cfg = {**model.cfg, "zoospore_alive": {"vpd": 0.0},
               "infection": {"infection_condition": 60.0}, "incubation": {"spl_end": 1.0}}
        direct = CropSeasonFirst(rb, cfg)
        with patch("model.first_infection.sporulation_dose_for_day", return_value=1.0):
            direct.simulate(use_wind_gate=False)
            candidate = run_scenario(rb, SCENARIOS["M0_merged_transfer"], config=cfg)
        self.assertTrue(cfg["dispersal"]["use_wind_gate"])
        self.assertFalse(direct.use_wind_gate)
        pd.testing.assert_frame_equal(direct.deposition, candidate.deposition)
        with patch("model.first_infection.sporulation_dose_for_day", return_value=1.0):
            direct.simulate()  # Return to the unchanged configured M1.
            candidate = run_scenario(rb, SCENARIOS["M1_merged_transfer_wind_gate"], config=cfg)
        self.assertTrue(direct.use_wind_gate)
        self.assertEqual(direct.deposition.iloc[0].deposition_datetime.hour, 14)
        pd.testing.assert_frame_equal(direct.deposition, candidate.deposition)


if __name__ == "__main__":
    unittest.main()
