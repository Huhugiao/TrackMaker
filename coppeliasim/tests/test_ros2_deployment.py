import math
from dataclasses import replace
from pathlib import Path
import unittest

import numpy as np

from attacker.observations import build_decentralized_actor_observation
from coppelia_env.ros2_deployment import (
    DefenderObservationMemory,
    DeploymentConfig,
    EpisodeMonitor,
    LinearSlewLimiter,
    Pose2D,
    build_policy_observations,
    deployment_input_timing,
    inputs_are_fresh,
    normalize_laser_scan,
    normalized_action_to_command,
    quaternion_to_yaw,
    sample_fixed_spawn,
    world_pose_to_training,
    yaw_to_quaternion,
)


class Ros2DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.config = DeploymentConfig()
        self.poses = {
            "defender": Pose2D(-1.0, 0.0, 0.0),
            "attacker": Pose2D(1.0, 0.0, math.pi),
            "target": Pose2D(2.0, 0.0, 0.0),
        }

    def test_world_pose_uses_training_image_convention(self):
        state = world_pose_to_training(Pose2D(1.0, 2.0, math.pi / 2.0), self.config)
        self.assertAlmostEqual(state["x"], 418.0)
        self.assertAlmostEqual(state["y"], 118.0)
        self.assertAlmostEqual(state["theta"], 270.0)

    def test_builds_original_policy_shapes(self):
        radar = np.zeros(64, dtype=np.float32)
        actor, privileged, critic = build_policy_observations(self.poses, radar, radar, self.config)
        self.assertEqual(actor.shape, (71,))
        self.assertEqual(privileged.shape, (72,))
        self.assertEqual(critic.shape, (143,))
        self.assertEqual(build_decentralized_actor_observation(privileged).shape, (70,))

    def test_defender_visibility_memory_matches_gym_layout(self):
        radar = np.zeros(64, dtype=np.float32)
        memory = DefenderObservationMemory()
        visible, _, _ = build_policy_observations(
            self.poses,
            radar,
            radar,
            self.config,
            defender_memory=memory,
        )
        moved = dict(self.poses)
        moved["attacker"] = Pose2D(1.5, 0.0, math.pi)
        hidden, _, _ = build_policy_observations(
            moved,
            radar,
            radar,
            self.config,
            defender_memory=memory,
            obstacles=[{"type": "rect", "x": 300.0, "y": 300.0, "w": 40.0, "h": 40.0}],
        )
        self.assertEqual(float(visible[3]), 1.0)
        self.assertEqual(float(hidden[3]), 0.0)
        self.assertAlmostEqual(float(hidden[4]), -0.975, places=6)
        self.assertAlmostEqual(float(hidden[0]), float(visible[0]), places=6)

    def test_scan_is_sampled_clockwise_from_robot_forward(self):
        count = 360
        angle_min = -math.pi
        increment = 2.0 * math.pi / count
        ranges = np.full(count, 3.0, dtype=np.float64)
        clockwise_90_index = int(round((-math.pi / 2.0 - angle_min) / increment))
        ranges[clockwise_90_index] = 0.3
        radar = normalize_laser_scan(
            ranges,
            angle_min=angle_min,
            angle_increment=increment,
            range_min=0.05,
            range_max=12.0,
            config=self.config,
        )
        self.assertAlmostEqual(float(radar[16]), -0.8, places=3)
        self.assertAlmostEqual(float(radar[0]), 1.0, places=3)

    def test_action_scaling_and_attacker_slew(self):
        command = normalized_action_to_command(
            [1.0, 1.0],
            max_linear_mps=self.config.defender_max_linear_mps,
            max_angular_radps=self.config.defender_max_angular_radps,
        )
        self.assertAlmostEqual(command.linear_mps, 0.234)
        self.assertAlmostEqual(command.angular_radps, -math.radians(54.0))
        limiter = LinearSlewLimiter(self.config.attacker_max_accel_mps2)
        self.assertAlmostEqual(limiter.apply(0.18, 1.0 / 9.0), 0.054, places=6)
        with self.assertRaises(ValueError):
            normalized_action_to_command(
                [float("nan"), 0.0],
                max_linear_mps=0.2,
                max_angular_radps=1.0,
            )

    def test_terminal_precedence_and_virtual_capture(self):
        monitor = EpisodeMonitor(self.config)
        close = dict(self.poses)
        close["attacker"] = Pose2D(-0.6, 0.0, 0.0)
        collision = monitor.update(close, defender_collision=True)
        self.assertEqual(collision.reason, "defender_collision")
        self.assertEqual(collision.outcome, "draw")

        monitor.reset()
        capture = monitor.update(close)
        self.assertEqual(capture.reason, "defender_caught_attacker")
        self.assertEqual(capture.outcome, "defender_success")

        monitor.reset()
        breach_poses = dict(self.poses)
        breach_poses["attacker"] = Pose2D(1.8, 0.0, 0.0)
        breach = monitor.update(breach_poses)
        self.assertEqual(breach.reason, "attacker_caught_target")

        monitor.reset()
        attacker_collision = monitor.update(self.poses, attacker_collision=True)
        self.assertIsNone(attacker_collision)

        timeout_monitor = EpisodeMonitor(replace(self.config, max_steps=1))
        timeout = timeout_monitor.update(self.poses)
        self.assertEqual(timeout.reason, "timeout_defender_wins")

    def test_stale_inputs_and_quaternion_validation(self):
        self.assertTrue(inputs_are_fresh([9.8, 9.9], now=10.0, timeout_s=0.5))
        self.assertFalse(inputs_are_fresh([9.0, 9.9], now=10.0, timeout_s=0.5))
        quaternion = yaw_to_quaternion(-1.2)
        self.assertAlmostEqual(quaternion_to_yaw(*quaternion), -1.2)
        with self.assertRaises(ValueError):
            quaternion_to_yaw(0.0, 0.0, 0.0, 0.0)

    def test_five_input_timing_rejects_missing_stale_and_skewed_samples(self):
        fresh = deployment_input_timing([9.96, 9.97, 9.98, 9.99, 10.0], now=10.0, config=self.config)
        self.assertTrue(fresh.fresh)
        self.assertEqual(fresh.reason, "fresh")
        self.assertEqual(
            deployment_input_timing([9.94, 9.97, 9.98, 9.99, 10.0], now=10.0, config=self.config).reason,
            "skew",
        )
        self.assertEqual(
            deployment_input_timing([9.0] * 5, now=10.0, config=self.config).reason,
            "stale",
        )
        self.assertEqual(deployment_input_timing([10.0] * 4, now=10.0, config=self.config).reason, "missing")

    def test_paired_spawn_is_deterministic_and_physically_clear(self):
        manifest = Path(__file__).resolve().parents[1] / "scenes/trackmaker_turtlebot4_scene.json"
        first = sample_fixed_spawn(manifest, 20260326)
        repeated = sample_fixed_spawn(manifest, 20260326)
        different = sample_fixed_spawn(manifest, 20260327)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)
        self.assertGreater(math.dist((first["defender"].x, first["defender"].y), (first["attacker"].x, first["attacker"].y)), 1.5)


if __name__ == "__main__":
    unittest.main()
