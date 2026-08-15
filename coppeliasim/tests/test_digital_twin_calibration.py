import copy
from pathlib import Path
import unittest

from coppelia_env.digital_twin import load_profile
from coppelia_env.digital_twin_calibration import (
    REQUIRED_CATEGORIES,
    calibration_plan,
    fit_calibration_dataset,
    measured_profile_from_fit,
    relative_error,
    synthetic_calibration_dataset,
    validate_calibration_dataset,
)


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "coppeliasim/profiles/trackmaker_turtlebot4_v2_1_prior.json"


class DigitalTwinCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prior = load_profile(PROFILE_PATH)
        cls.truth = {}
        for role in ("defender", "attacker"):
            robot = cls.prior["robots"][role]
            cls.truth[role] = {
                "left_gain": 0.91,
                "right_gain": 1.04,
                "left_deadband_mps": 0.004,
                "right_deadband_mps": 0.006,
                "time_constant_s": 0.11,
                "acceleration_mps2": 0.88,
                "braking_mps2": 2.25,
                "fixed_delay_s": 0.030,
                "wheel_radius_m": robot["wheel_radius_m"] * 1.03,
                "wheel_separation_m": robot["wheel_separation_m"] * 0.98,
            }
        cls.dataset = synthetic_calibration_dataset(cls.prior, truth_by_role=cls.truth, seed=7719)
        cls.fit = fit_calibration_dataset(cls.dataset, cls.prior, split_seed=191)

    def test_plan_covers_all_required_excitation_and_outage(self):
        plan = calibration_plan(self.prior["robots"]["defender"])
        self.assertTrue(REQUIRED_CATEGORIES.issubset({phase.category for phase in plan}))
        outage = next(phase for phase in plan if phase.name == "watchdog_outage")
        self.assertFalse(outage.publish_commands)

    def test_synthetic_fit_recovers_known_parameters(self):
        physics_step = self.prior["simulation"]["physics_step_s"]
        for role in ("defender", "attacker"):
            actual = self.fit["roles"][role]["parameters"]
            expected = self.truth[role]
            for key in ("left_gain", "right_gain", "wheel_radius_m", "wheel_separation_m"):
                self.assertLessEqual(relative_error(actual[key], expected[key]), 0.02, (role, key, actual[key]))
            self.assertLessEqual(abs(actual["fixed_delay_s"] - expected["fixed_delay_s"]), physics_step + 1e-12)
            for key in ("time_constant_s", "acceleration_mps2"):
                self.assertLessEqual(relative_error(actual[key], expected[key]), 0.10, (role, key, actual[key]))
            self.assertLessEqual(
                relative_error(actual["braking_mps2"], expected["braking_mps2"]),
                0.15,
                (role, "braking_mps2", actual["braking_mps2"]),
            )
            self.assertEqual(self.fit["roles"][role]["split"]["ratios"], [0.60, 0.20, 0.20])

    def test_measured_profile_is_independent_and_valid(self):
        unchanged = copy.deepcopy(self.prior)
        measured = measured_profile_from_fit(
            self.prior,
            self.fit,
            profile_id="trackmaker_turtlebot4_v2_1_measured_synthetic_test",
            calibration_seed=7719,
        )
        self.assertEqual(self.prior, unchanged)
        self.assertEqual(measured["provenance"], "measured")
        self.assertEqual(measured["calibration_state"], "calibrated")
        self.assertNotEqual(measured["checksum"], self.prior["checksum"])
        self.assertEqual(measured["source"]["prior_profile_checksum"], self.prior["checksum"])
        self.assertEqual(measured["source"]["urdf_sha256"], self.prior["source"]["urdf_sha256"])

    def test_dataset_rejects_failed_graceful_shutdown(self):
        failed = copy.deepcopy(self.dataset)
        failed["graceful_shutdown"]["actual_wheel_stopped"] = False
        with self.assertRaisesRegex(ValueError, "stopped actual wheel speeds"):
            validate_calibration_dataset(failed)

    def test_dataset_rejects_checksum_tampering(self):
        tampered = copy.deepcopy(self.dataset)
        tampered["roles"]["defender"]["samples"][10]["joint_actual_radps"][0] += 0.1
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_calibration_dataset(tampered)


if __name__ == "__main__":
    unittest.main()
