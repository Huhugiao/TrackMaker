import math
import sys
import unittest

import numpy as np
import torch

from coppelia_env.skill_adapter import (
    CoppeliaSkillObservationProvider,
    NormalizedCmdVelAdapter,
    _install_numpy_core_pickle_alias,
    build_critic_observation,
    compute_gae,
    serialize_recurrent_hidden,
)


class DummyEnv:
    width = 100
    height = 80
    pixel_size = 4.0
    max_steps = 300
    defender_max_v = 0.52
    defender_max_w = 2.0
    attacker_max_v = 0.40
    attacker_max_w = 3.0


class SkillAdapterTests(unittest.TestCase):
    def setUp(self):
        self.env = DummyEnv()
        self.transition = {
            "obs": {},
            "state": {
                "defender": {"x": 10.0, "y": 20.0, "theta": 0.0},
                "attacker": {"x": 30.0, "y": 20.0, "theta": 90.0},
                "target": {"x": 50.0, "y": 60.0, "theta": 0.0},
                "step_count": 7,
            },
        }

    def test_provider_shapes(self):
        provider = CoppeliaSkillObservationProvider(self.env)
        actor, privileged, critic = provider.observations(self.transition)
        self.assertEqual(actor.shape, (71,))
        self.assertEqual(privileged.shape, (72,))
        self.assertEqual(critic.shape, (143,))
        self.assertTrue(np.all(np.isfinite(actor)))
        self.assertTrue(np.all(np.isfinite(privileged)))
        self.assertTrue(np.all(np.isfinite(critic)))

    def test_actor_relative_features(self):
        provider = CoppeliaSkillObservationProvider(
            self.env,
            radar_fn=lambda _agent: np.zeros(64, dtype=np.float32),
        )
        actor = provider.actor_observation(self.transition)
        map_diag = math.hypot(self.env.width, self.env.height)
        expected_dist = (20.0 / map_diag) * 2.0 - 1.0
        self.assertAlmostEqual(float(actor[0]), expected_dist, places=6)
        self.assertAlmostEqual(float(actor[1]), 0.0, places=6)
        self.assertEqual(float(actor[3]), 1.0)
        expected_target_dist = (math.hypot(40.0, 40.0) / map_diag) * 2.0 - 1.0
        self.assertAlmostEqual(float(actor[5]), expected_target_dist, places=6)

    def test_privileged_observation_layout(self):
        provider = CoppeliaSkillObservationProvider(
            self.env,
            radar_fn=lambda _agent: np.ones(64, dtype=np.float32) * -0.25,
        )
        privileged = provider.privileged_observation(self.transition)
        self.assertAlmostEqual(float(privileged[0]), (30.0 / self.env.width) * 2.0 - 1.0, places=6)
        self.assertAlmostEqual(float(privileged[2]), (90.0 / 180.0) - 1.0, places=6)
        self.assertTrue(np.allclose(privileged[8:], -0.25))

    def test_build_critic_observation(self):
        actor = np.ones(71, dtype=np.float32)
        privileged = np.ones(72, dtype=np.float32) * 2.0
        critic = build_critic_observation(actor, privileged)
        self.assertEqual(critic.shape, (143,))
        self.assertTrue(np.allclose(critic[:71], 1.0))
        self.assertTrue(np.allclose(critic[71:], 2.0))

    def test_cmd_vel_adapter(self):
        adapter = NormalizedCmdVelAdapter(self.env)
        cmd = adapter.defender_command(np.array([0.5, 0.0], dtype=np.float32))
        self.assertAlmostEqual(cmd.angular_radps, 1.0, places=6)
        self.assertAlmostEqual(cmd.linear_mps, 0.26, places=6)
        reverse_cmd = NormalizedCmdVelAdapter(self.env, allow_reverse=True).defender_command([0.0, -1.0])
        self.assertAlmostEqual(reverse_cmd.linear_mps, -0.52, places=6)

    def test_compute_gae(self):
        adv, returns = compute_gae(
            rewards=np.array([1.0, 1.0], dtype=np.float32),
            dones=np.array([0.0, 1.0], dtype=np.float32),
            values=np.array([0.5, 0.25], dtype=np.float32),
            last_value=0.0,
            gamma=0.9,
            gae_lambda=1.0,
        )
        expected_returns = np.array([1.9, 1.0], dtype=np.float32)
        self.assertTrue(np.allclose(returns, expected_returns, atol=1e-6))
        self.assertTrue(np.allclose(adv, expected_returns - np.array([0.5, 0.25], dtype=np.float32), atol=1e-6))

    def test_installs_numpy_core_pickle_alias(self):
        old_core = sys.modules.pop("numpy._core", None)
        old_multiarray = sys.modules.pop("numpy._core.multiarray", None)
        old_numeric = sys.modules.pop("numpy._core.numeric", None)
        try:
            _install_numpy_core_pickle_alias()
            self.assertIs(sys.modules["numpy._core"], np.core)
            self.assertIn("numpy._core.multiarray", sys.modules)
            self.assertIn("numpy._core.numeric", sys.modules)
        finally:
            if old_core is not None:
                sys.modules["numpy._core"] = old_core
            if old_multiarray is not None:
                sys.modules["numpy._core.multiarray"] = old_multiarray
            if old_numeric is not None:
                sys.modules["numpy._core.numeric"] = old_numeric

    def test_serialize_recurrent_hidden_uses_network_spec(self):
        class Network:
            def recurrent_hidden_spec(self, role):
                self.role = role
                return 2, 3

        class Model:
            network = Network()
            _actor_hidden = torch.arange(6, dtype=torch.float32).reshape(2, 1, 3)

        hidden = serialize_recurrent_hidden(Model(), "actor")
        self.assertEqual(hidden.shape, (2, 3))
        self.assertTrue(np.allclose(hidden, np.arange(6, dtype=np.float32).reshape(2, 3)))

    def test_serialize_recurrent_hidden_returns_zero_when_missing(self):
        class Network:
            def recurrent_hidden_spec(self, role):
                return 1, 4

        class Model:
            network = Network()
            _critic_hidden = None

        hidden = serialize_recurrent_hidden(Model(), "critic")
        self.assertEqual(hidden.shape, (1, 4))
        self.assertTrue(np.allclose(hidden, 0.0))


if __name__ == "__main__":
    unittest.main()
