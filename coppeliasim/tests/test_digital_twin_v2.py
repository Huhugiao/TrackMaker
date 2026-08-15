import copy
import json
import math
from pathlib import Path
import unittest

import jsonschema

from coppelia_env.digital_twin import (
    ActuatorConfig,
    DeterministicActuator,
    aggregate_urdf_inertials,
    build_prior_profile,
    evaluate_input_timing,
    load_profile,
    profile_checksum,
    runtime_mismatches,
    validate_profile,
)


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = ROOT / "coppeliasim/profiles/trackmaker_turtlebot4_v2_1_prior.json"
SCHEMA_PATH = ROOT / "coppeliasim/profiles/trackmaker_digital_twin_profile.schema.json"
URDF_PATH = ROOT / "coppeliasim/urdf/turtlebot4_standard.coppeliasim.urdf"
SCENE_DIR = ROOT / "coppeliasim/scenes"


class DigitalTwinProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = load_profile(PROFILE_PATH)

    def test_profile_satisfies_json_schema_and_checksum(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator(schema).validate(self.profile)
        self.assertEqual(self.profile["checksum"], profile_checksum(self.profile))
        self.assertEqual(self.profile["provenance"], "prior")
        self.assertEqual(self.profile["calibration_state"], "uncalibrated")

    def test_checked_profile_is_exactly_reproducible_from_urdf_and_seed(self):
        self.assertEqual(build_prior_profile(URDF_PATH, seed=20260815), self.profile)

    def test_profile_rejects_tampering_and_invalid_provenance(self):
        changed = copy.deepcopy(self.profile)
        changed["simulation"]["physics_step_s"] = 0.01
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_profile(changed)
        changed["checksum"] = profile_checksum(changed)
        with self.assertRaisesRegex(ValueError, "5 ms"):
            validate_profile(changed)

        changed = copy.deepcopy(self.profile)
        changed["provenance"] = "synthetic"
        changed["checksum"] = profile_checksum(changed)
        with self.assertRaisesRegex(ValueError, "prior.*measured"):
            validate_profile(changed)

        changed = copy.deepcopy(self.profile)
        changed["robots"]["attacker"]["actuator"]["response_space"] = "wheel_sides"
        changed["checksum"] = profile_checksum(changed)
        with self.assertRaisesRegex(ValueError, "response_space.*body_twist"):
            validate_profile(changed)

    def test_urdf_aggregate_preserves_total_and_independent_wheels(self):
        full = aggregate_urdf_inertials(URDF_PATH)
        chassis = aggregate_urdf_inertials(
            URDF_PATH,
            excluded_links={"left_wheel", "right_wheel", "front_caster_link"},
        )
        robot = self.profile["robots"]["defender"]
        self.assertAlmostEqual(full.mass_kg, robot["mass_kg"], places=10)
        self.assertAlmostEqual(chassis.mass_kg, robot["chassis"]["mass_kg"], places=10)
        self.assertNotIn("left_wheel", robot["chassis"]["aggregated_links"])
        self.assertNotIn("right_wheel", robot["chassis"]["aggregated_links"])
        expected_total = (
            robot["chassis"]["mass_kg"]
            + robot["caster"]["mass_kg"]
            + 2.0 * robot["wheel"]["mass_kg_each"]
        )
        self.assertAlmostEqual(robot["mass_kg"], expected_total, places=10)

    def test_geometry_has_two_wheels_and_caster_support_without_chassis_floor_contact(self):
        geometry = self.profile["robots"]["defender"]["geometry"]
        floor_top = 0.02
        base_z = geometry["nominal_base_z_m"]
        for key in ("left_wheel", "right_wheel"):
            wheel = geometry[key]
            self.assertAlmostEqual(base_z + wheel["center_m"][2] - wheel["radius_m"], floor_top, places=9)
        caster = geometry["caster"]
        self.assertAlmostEqual(
            base_z + caster["center_m"][2] - caster["radius_m"],
            floor_top - caster["preload_m"],
            places=9,
        )
        self.assertEqual(caster["preload_m"], 0.0)
        self.assertEqual(caster["collision_model"], "equivalent_front_slider")
        self.assertEqual(caster["mount_model"], "passive_spherical_joint")
        self.assertEqual(caster["contact_material"], "caster")
        base = geometry["base"]
        self.assertGreater(base_z + base["center_m"][2] - 0.5 * base["height_m"], floor_top)
        self.assertGreater(geometry["base_clearance_m"], 0.0)
        self.assertFalse(geometry["visual_mesh_respondable"])
        self.assertFalse(geometry["wheel_drop_enabled"])

    def test_runtime_introspection_reports_exact_mismatch(self):
        runtime = {
            "profile_checksum": self.profile["checksum"],
            "engine_index": 0,
            "engine_version": 0,
            "outer_step_s": 0.05,
            "physics_step_s": 0.005,
            "solver_iterations": 100,
            "gravity_mps2": [0.0, 0.0, -9.81],
            "robots": {},
        }
        for role, robot in self.profile["robots"].items():
            runtime["robots"][role] = {
                "chassis_mass_kg": robot["chassis"]["mass_kg"],
                "caster_mass_kg": robot["caster"]["mass_kg"],
                "caster_mount_model": robot["geometry"]["caster"]["mount_model"],
                "wheel_radius_m": robot["wheel_radius_m"],
                "wheel_separation_m": robot["wheel_separation_m"],
                "target_force_nm": robot["actuator"]["target_force_nm"],
            }
        self.assertEqual(runtime_mismatches(self.profile, runtime), [])
        runtime["solver_iterations"] = 20
        self.assertEqual(len(runtime_mismatches(self.profile, runtime)), 1)
        self.assertIn("solver_iterations", runtime_mismatches(self.profile, runtime)[0])

    def test_checked_scenes_embed_profile_driven_nonteleport_contract(self):
        cases = (
            ("trackmaker_turtlebot4_v2_1_scene", "dense", 29),
            ("trackmaker_turtlebot4_v2_1_calibration_scene", "none", 4),
        )
        for stem, density, obstacle_count in cases:
            with self.subTest(scene=stem):
                manifest = json.loads((SCENE_DIR / f"{stem}.json").read_text(encoding="utf-8"))
                meta = manifest["meta"]
                self.assertEqual(manifest["profile"], self.profile)
                self.assertEqual(meta["profile"]["checksum"], self.profile["checksum"])
                self.assertEqual(meta["density"], density)
                self.assertEqual(len(manifest["obstacles"]), obstacle_count)
                self.assertEqual(meta["motion_model"], "profiled_turtlebot4_wheel_velocity")
                self.assertFalse(meta["runtime_teleport_enabled"])
                self.assertFalse(meta["collision_proxy_enabled"])
                self.assertFalse(meta["visual_mesh_respondable"])
                self.assertFalse(meta["controller_env_obstacle_mask"])
                self.assertFalse(meta["action_shield"])
                for role in ("defender", "attacker"):
                    actuator = self.profile["robots"][role]["actuator"]
                    self.assertEqual(meta[f"{role}_speed_m_per_s"], actuator["max_linear_mps"])
                    self.assertEqual(meta[f"{role}_max_turn_rad_per_s"], actuator["max_angular_radps"])
                self.assertGreater((SCENE_DIR / f"{stem}.ttt").stat().st_size, 1_000_000)


class DeterministicActuatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = load_profile(PROFILE_PATH)["robots"]["defender"]
        cls.config = ActuatorConfig.from_robot_profile(cls.robot)

    def test_finite_clamp_delay_and_watchdog_order(self):
        actuator = DeterministicActuator(self.config, seed=123)
        event = actuator.enqueue(0.0, 99.0, -99.0)
        self.assertEqual(event["clamped_linear_mps"], self.config.max_linear_mps)
        self.assertEqual(event["clamped_angular_radps"], -self.config.max_angular_radps)
        self.assertGreaterEqual(event["scheduled_time_s"], 0.015)
        self.assertLessEqual(event["scheduled_time_s"], 0.025)
        self.assertTrue(actuator.step(0.01, 0.005)["watchdog_active"])
        executed = actuator.step(0.025, 0.005)
        self.assertEqual(len(executed["executed_events"]), 1)
        self.assertFalse(executed["watchdog_active"])
        self.assertTrue(actuator.step(0.53, 0.005)["watchdog_active"])

        invalid = DeterministicActuator(self.config, seed=123).enqueue(0.0, math.nan, 0.1)
        self.assertFalse(invalid["finite"])
        self.assertEqual(invalid["clamped_linear_mps"], 0.0)
        self.assertEqual(invalid["clamped_angular_radps"], 0.0)

    def test_packet_loss_and_preset_outage_are_independent_drop_reasons(self):
        loss_config = ActuatorConfig(**{**self.config.__dict__, "packet_loss_probability": 1.0})
        loss = DeterministicActuator(loss_config, seed=7).enqueue(0.0, 0.1, 0.0)
        self.assertTrue(loss["dropped"])
        self.assertEqual(loss["drop_reason"], "packet_loss")

        outage_config = ActuatorConfig(
            **{
                **self.config.__dict__,
                "packet_loss_probability": 0.0,
                "dropout_windows_s": ((0.1, 0.2),),
            }
        )
        outage = DeterministicActuator(outage_config, seed=7).enqueue(0.15, 0.1, 0.0)
        self.assertTrue(outage["dropped"])
        self.assertEqual(outage["drop_reason"], "preset_outage")

    def test_same_profile_and_seed_reproduce_queue_and_wheel_targets(self):
        def rollout(seed):
            actuator = DeterministicActuator(self.config, seed=seed)
            events = [actuator.enqueue(index * 0.05, 0.1, (-1) ** index * 0.2) for index in range(8)]
            samples = [actuator.step(index * 0.005, 0.005) for index in range(100)]
            return events, samples

        self.assertEqual(rollout(919), rollout(919))
        self.assertNotEqual(rollout(919)[0], rollout(920)[0])

    def test_turn_rate_does_not_cancel_body_linear_slew(self):
        config = ActuatorConfig(
            **{
                **self.config.__dict__,
                "fixed_delay_s": 0.0,
                "uniform_jitter_s": 0.0,
                "packet_loss_probability": 0.0,
                "left_gain": 1.0,
                "right_gain": 1.0,
                "left_deadband_mps": 0.0,
                "right_deadband_mps": 0.0,
            }
        )
        straight = DeterministicActuator(config, seed=31)
        turning = DeterministicActuator(config, seed=31)
        straight.enqueue(0.0, 0.18, 0.0)
        turning.enqueue(0.0, 0.18, config.max_angular_radps)
        straight_state = turning_state = None
        for index in range(1, 101):
            now = index * 0.005
            straight_state = straight.step(now, 0.005)
            turning_state = turning.step(now, 0.005)
        self.assertIsNotNone(straight_state)
        self.assertIsNotNone(turning_state)
        self.assertAlmostEqual(
            straight_state["filtered_linear_mps"], turning_state["filtered_linear_mps"], places=12
        )
        self.assertGreater(turning_state["filtered_linear_mps"], 0.0)

    def test_five_input_skew_and_staleness_fail_closed(self):
        self.assertTrue(
            evaluate_input_timing([1.0, 1.01, 1.02, 1.03, 1.04], now=1.05, stale_after_s=0.5).fresh
        )
        self.assertEqual(
            evaluate_input_timing([1.0, 1.01, 1.02, 1.03, 1.051], now=1.06, stale_after_s=0.5).reason,
            "skew",
        )
        self.assertEqual(
            evaluate_input_timing([1.0] * 5, now=1.6, stale_after_s=0.5).reason,
            "stale",
        )


if __name__ == "__main__":
    unittest.main()
