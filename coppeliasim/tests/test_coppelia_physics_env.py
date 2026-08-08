import unittest
import math
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from coppelia_env.track_env import CoppeliaTrackEnv, SimState, VelocityCommand
from coppelia_env.turtlebot4 import TurtleBot4Spec


class FakeClient:
    def __init__(self):
        self.steps = 0

    def step(self, wait=True):
        self.steps += 1


class FakeSocket:
    def __init__(self):
        self.options = []

    def setsockopt(self, option, value):
        self.options.append((option, value))


class FakeTimeoutClient(FakeClient):
    def __init__(self):
        super().__init__()
        self.socket = FakeSocket()


class FakeSim:
    def __init__(self):
        self.positions = {1: [0.0, 0.0, 0.07], 2: [1.0, 0.0, 0.07], 3: [2.0, 0.0, 0.04]}
        self.orientations = {1: [0.0, 0.0, 0.0], 2: [0.0, 0.0, 0.0], 3: [0.0, 0.0, 0.0]}
        self.velocity_calls = []
        self.position_calls = []
        self.orientation_calls = []

    def getObjectPosition(self, handle, relative_to):
        return list(self.positions[handle])

    def getObjectOrientation(self, handle, relative_to):
        return list(self.orientations[handle])

    def setObjectPosition(self, handle, relative_to, pos):
        self.position_calls.append((handle, list(pos)))
        self.positions[handle] = list(pos)

    def setObjectOrientation(self, handle, relative_to, ori):
        self.orientation_calls.append((handle, list(ori)))
        self.orientations[handle] = list(ori)

    def setObjectVelocity(self, handle, linear, angular):
        self.velocity_calls.append((handle, list(linear), list(angular)))

    def checkCollision(self, a, b):
        return 0


class FakeWheelSim(FakeSim):
    def __init__(self):
        super().__init__()
        self.joint_target_velocity_calls = []
        self.matrix_calls = []
        self.proximity_calls = []
        self.proximity_distances = [0.5, 1.25]

    def setJointTargetVelocity(self, handle, velocity):
        self.joint_target_velocity_calls.append((handle, float(velocity)))

    def getObjectMatrix(self, handle, relative_to):
        return [1.0, 0.0, 0.0, self.positions[handle][0], 0.0, 1.0, 0.0, self.positions[handle][1], 0.0, 0.0, 1.0, self.positions[handle][2]]

    def setObjectMatrix(self, handle, relative_to, matrix):
        self.matrix_calls.append((handle, relative_to, list(matrix)))

    def handleProximitySensor(self, handle):
        self.proximity_calls.append(handle)
        distance = self.proximity_distances[min(len(self.proximity_calls) - 1, len(self.proximity_distances) - 1)]
        return 1, distance, [0.0, 0.0, distance], 99, [0.0, 0.0, -1.0]


class FakeCollisionWheelSim(FakeWheelSim):
    def __init__(self):
        super().__init__()
        self.positions[10] = list(self.positions[1])
        self.positions[15] = list(self.positions[1])
        self.collision_calls = []

    def checkCollision(self, a, b):
        self.collision_calls.append((a, b))
        return 1 if a in {10, 15} and b == 4 else 0


class FakeResolveSim:
    def __init__(self, aliases):
        self.aliases = dict(aliases)

    def getObject(self, path):
        if path not in self.aliases:
            raise RuntimeError(f"missing alias {path}")
        return self.aliases[path]


class ForceOnlyFakeSim:
    def __init__(self):
        self.positions = {1: [0.0, 0.0, 0.07], 2: [1.0, 0.0, 0.07], 3: [2.0, 0.0, 0.04]}
        self.orientations = {1: [0.0, 0.0, 0.0], 2: [0.0, 0.0, 0.0], 3: [0.0, 0.0, 0.0]}
        self.position_calls = []
        self.orientation_calls = []
        self.force_calls = []

    def getObjectPosition(self, handle, relative_to):
        return list(self.positions[handle])

    def getObjectOrientation(self, handle, relative_to):
        return list(self.orientations[handle])

    def setObjectPosition(self, handle, relative_to, pos):
        self.position_calls.append((handle, list(pos)))
        self.positions[handle] = list(pos)

    def setObjectOrientation(self, handle, relative_to, ori):
        self.orientation_calls.append((handle, list(ori)))
        self.orientations[handle] = list(ori)

    def getObjectVelocity(self, handle):
        if handle == 1:
            return [0.1, 0.0, 0.0], [0.0, 0.0, -0.2]
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def getShapeMass(self, handle):
        return 4.0

    def addForceAndTorque(self, handle, force, torque):
        self.force_calls.append((handle, list(force), list(torque)))

    def checkCollision(self, a, b):
        return 0


class CoppeliaPhysicsEnvTests(unittest.TestCase):
    def make_env(self, sim=None):
        env = CoppeliaTrackEnv.__new__(CoppeliaTrackEnv)
        env.client = FakeClient()
        env.sim = sim or FakeSim()
        env.handles = {
            "defender": 1,
            "attacker": 2,
            "target": 3,
            "obstacle_001": 4,
            "defender_base": 1,
            "attacker_base": 2,
            "defender_left_wheel_joint": 11,
            "defender_right_wheel_joint": 12,
            "attacker_left_wheel_joint": 21,
            "attacker_right_wheel_joint": 22,
        }
        env.robot_specs = {
            "defender": TurtleBot4Spec(
                base_link="base_link",
                left_wheel_joint="left_wheel_joint",
                right_wheel_joint="right_wheel_joint",
                lidar_link="rplidar_link",
                rgb_camera_frame="oakd_rgb_camera_frame",
                wheel_radius_m=0.05,
                wheel_separation_m=0.30,
            ),
            "attacker": TurtleBot4Spec(
                base_link="base_link",
                left_wheel_joint="left_wheel_joint",
                right_wheel_joint="right_wheel_joint",
                lidar_link="rplidar_link",
                rgb_camera_frame="oakd_rgb_camera_frame",
                wheel_radius_m=0.05,
                wheel_separation_m=0.30,
            ),
        }
        env.scale = 0.01
        env.sim_step_dt = 0.05
        env.motion_model = "dynamic_velocity"
        env.physics_mode = ""
        env.dynamic_linear_gain = 1.0
        env.dynamic_angular_gain = 1.0
        env.dynamic_z_gain = 1.0
        env.dynamic_attitude_gain = 1.0
        env.dynamic_max_force = 100.0
        env.dynamic_max_torque = 100.0
        env.dynamic_nominal_z = 0.07
        env.stepping = True
        env.width = 640
        env.height = 640
        env.pixel_size = 4.0
        env.robot_radius_px = 8.0
        env.target_radius_px = 16.0
        env.capture_radius_px = 20.0
        env.capture_sector_angle_deg = 30.0
        env.capture_required_steps = 1
        env.max_steps = 100
        env.defender_max_v = 0.52
        env.attacker_max_v = 0.40
        env.defender_max_w = 2.0
        env.attacker_max_w = 3.0
        env.reward_mode = "chase"
        env._def_capture_count = 0
        env._att_capture_count = 0
        env._state = SimState(
            defender={"x": 318.0, "y": 318.0, "theta": 0.0},
            attacker={"x": 418.0, "y": 318.0, "theta": 0.0},
            target={"x": 518.0, "y": 318.0, "theta": 0.0},
            step_count=0,
        )
        env._last_state = env._copy_state(env._state)
        env._set_initial_distances(env._state)
        env._done = False
        env._check_collision = lambda _key: False
        env._compute_reward = lambda **_kwargs: (0.0, False, False, {"reason": "", "win": False})
        env._transition = lambda reward, terminated, truncated, info: {
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "done": bool(terminated or truncated),
            "info": info,
        }
        return env

    def test_init_prefers_motion_model_over_robot_model_metadata(self):
        manifest = {
            "meta": {
                "robot_model": "official_turtlebot4_urdf",
                "motion_model": "kinematic_velocity_base",
            },
            "map": {"width_px": 640, "height_px": 640, "scale_m_per_px": 0.01},
            "objects": {},
            "obstacles": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "scene.json"
            manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")

            env = CoppeliaTrackEnv(manifest=manifest_path, connect=False)

        self.assertEqual(env.motion_model, "kinematic_velocity_base")

    def test_dynamic_step_uses_object_velocity_not_pose_teleport(self):
        env = self.make_env()
        env.step(VelocityCommand(0.3, 0.5), VelocityCommand(0.2, -0.25))

        self.assertEqual(env.sim.position_calls, [])
        self.assertEqual(env.sim.orientation_calls, [])
        self.assertEqual(len(env.sim.velocity_calls), 2)
        self.assertEqual(env.sim.velocity_calls[0][0], 1)
        self.assertAlmostEqual(env.sim.velocity_calls[0][1][0], 0.3, places=6)
        self.assertAlmostEqual(env.sim.velocity_calls[0][2][2], -0.5, places=6)
        self.assertEqual(env.client.steps, 1)

    def test_wheel_velocity_step_targets_turtlebot4_wheel_joints(self):
        env = self.make_env(sim=FakeWheelSim())
        env.motion_model = "wheel_velocity"

        env.step(VelocityCommand(0.4, 1.0), VelocityCommand(0.2, -0.5))

        self.assertEqual(env.sim.position_calls, [])
        self.assertEqual(env.sim.orientation_calls, [])
        self.assertEqual(env.sim.velocity_calls, [])
        self.assertEqual(len(env.sim.joint_target_velocity_calls), 4)
        self.assertEqual(env.sim.joint_target_velocity_calls[0][0], 11)
        self.assertAlmostEqual(env.sim.joint_target_velocity_calls[0][1], (0.4 - 0.15) / 0.05, places=6)
        self.assertEqual(env.sim.joint_target_velocity_calls[1][0], 12)
        self.assertAlmostEqual(env.sim.joint_target_velocity_calls[1][1], (0.4 + 0.15) / 0.05, places=6)
        self.assertEqual(env.sim.joint_target_velocity_calls[2][0], 21)
        self.assertAlmostEqual(env.sim.joint_target_velocity_calls[2][1], (0.2 - 0.5 * 0.30 * -0.5) / 0.05, places=6)
        self.assertEqual(env.sim.joint_target_velocity_calls[3][0], 22)
        self.assertAlmostEqual(env.sim.joint_target_velocity_calls[3][1], (0.2 + 0.5 * 0.30 * -0.5) / 0.05, places=6)
        self.assertEqual(env.client.steps, 1)

    def test_wheel_velocity_radar_reads_coppelia_proximity_sensor(self):
        env = self.make_env(sim=FakeWheelSim())
        env.motion_model = "wheel_velocity"
        env.handles["defender_lidar"] = 13
        env.handles["defender_base"] = 1
        env.radar_rays = 2
        env.sensor_range_m = 3.0

        radar = CoppeliaTrackEnv._radar(env, env._state.defender)

        self.assertEqual(env.sim.proximity_calls, [13, 13])
        self.assertEqual(len(env.sim.matrix_calls), 2)
        self.assertAlmostEqual(float(radar[0]), 50.0, places=5)
        self.assertAlmostEqual(float(radar[1]), 125.0, places=5)

    def test_wheel_velocity_collision_checks_turtlebot4_base_not_root_dummy(self):
        env = self.make_env(sim=FakeCollisionWheelSim())
        env.motion_model = "wheel_velocity"
        env.handles["defender_base"] = 10

        with patch("coppelia_env.track_env.env_lib.is_point_blocked", return_value=False):
            collides = CoppeliaTrackEnv._check_collision(env, "defender")

        self.assertTrue(collides)
        self.assertIn((10, 4), env.sim.collision_calls)
        self.assertNotIn((1, 4), env.sim.collision_calls)

    def test_wheel_velocity_collision_prefers_collision_proxy_when_available(self):
        env = self.make_env(sim=FakeCollisionWheelSim())
        env.motion_model = "wheel_velocity"
        env.handles["defender_base"] = 10
        env.handles["defender_collision_proxy"] = 15

        with patch("coppelia_env.track_env.env_lib.is_point_blocked", return_value=False):
            collides = CoppeliaTrackEnv._check_collision(env, "defender")

        self.assertTrue(collides)
        self.assertIn((15, 4), env.sim.collision_calls)
        self.assertNotIn((10, 4), env.sim.collision_calls)

    def test_resolve_handles_includes_turtlebot4_robot_joints_and_sensors(self):
        env = CoppeliaTrackEnv.__new__(CoppeliaTrackEnv)
        env.sim = FakeResolveSim(
            {
                "/TrackMaker_defender": 1,
                "/TrackMaker_attacker": 2,
                "/TrackMaker_target": 3,
                "/TrackMaker_defender/base_link": 10,
                "/TrackMaker_defender/collision_proxy": 15,
                "/TrackMaker_defender/left_wheel_joint": 11,
                "/TrackMaker_defender/right_wheel_joint": 12,
                "/TrackMaker_defender/rplidar_sensor": 13,
                "/TrackMaker_defender/oakd_rgb_sensor": 14,
                "/TrackMaker_attacker/base_link": 20,
                "/TrackMaker_attacker/collision_proxy": 25,
                "/TrackMaker_attacker/left_wheel_joint": 21,
                "/TrackMaker_attacker/right_wheel_joint": 22,
                "/TrackMaker_attacker/rplidar_sensor": 23,
                "/TrackMaker_attacker/oakd_rgb_sensor": 24,
            }
        )
        env.manifest = {
            "objects": {
                "defender": {"alias": "TrackMaker_defender"},
                "attacker": {"alias": "TrackMaker_attacker"},
                "target": {"alias": "TrackMaker_target"},
            },
            "robots": {
                "defender": {
                    "root": "TrackMaker_defender",
                    "base_link": "TrackMaker_defender/base_link",
                    "collision_proxy": "TrackMaker_defender/collision_proxy",
                    "left_wheel_joint": "TrackMaker_defender/left_wheel_joint",
                    "right_wheel_joint": "TrackMaker_defender/right_wheel_joint",
                    "lidar_sensor": "TrackMaker_defender/rplidar_sensor",
                    "vision_sensor": "TrackMaker_defender/oakd_rgb_sensor",
                },
                "attacker": {
                    "root": "TrackMaker_attacker",
                    "base_link": "TrackMaker_attacker/base_link",
                    "collision_proxy": "TrackMaker_attacker/collision_proxy",
                    "left_wheel_joint": "TrackMaker_attacker/left_wheel_joint",
                    "right_wheel_joint": "TrackMaker_attacker/right_wheel_joint",
                    "lidar_sensor": "TrackMaker_attacker/rplidar_sensor",
                    "vision_sensor": "TrackMaker_attacker/oakd_rgb_sensor",
                },
            },
        }

        handles = CoppeliaTrackEnv._resolve_handles(env)

        self.assertEqual(handles["defender"], 1)
        self.assertEqual(handles["defender_base"], 10)
        self.assertEqual(handles["defender_collision_proxy"], 15)
        self.assertEqual(handles["defender_left_wheel_joint"], 11)
        self.assertEqual(handles["defender_right_wheel_joint"], 12)
        self.assertEqual(handles["defender_lidar"], 13)
        self.assertEqual(handles["defender_vision"], 14)
        self.assertEqual(handles["attacker_collision_proxy"], 25)
        self.assertEqual(handles["attacker_left_wheel_joint"], 21)

    def test_dynamic_step_falls_back_to_force_control_when_velocity_setter_is_missing(self):
        env = self.make_env(sim=ForceOnlyFakeSim())
        env.step(VelocityCommand(0.3, 0.5), VelocityCommand(0.2, -0.25))

        self.assertEqual(env.sim.position_calls, [])
        self.assertEqual(env.sim.orientation_calls, [])
        self.assertEqual(len(env.sim.force_calls), 2)
        self.assertEqual(env.sim.force_calls[0][0], 1)
        self.assertAlmostEqual(env.sim.force_calls[0][1][0], 0.8, places=6)
        self.assertAlmostEqual(env.sim.force_calls[0][1][2], 0.0, places=6)
        self.assertAlmostEqual(env.sim.force_calls[0][2][0], 0.0, places=6)
        self.assertAlmostEqual(env.sim.force_calls[0][2][1], 0.0, places=6)
        self.assertLess(env.sim.force_calls[0][2][2], 0.0)
        self.assertEqual(env.client.steps, 1)

    def test_physics_mode_dynamic_is_treated_as_dynamic(self):
        env = self.make_env()
        env.motion_model = "kinematic_velocity_base"
        env.physics_mode = "dynamic"

        env.step(VelocityCommand(0.3, 0.5), VelocityCommand(0.2, -0.25))

        self.assertEqual(env.sim.position_calls, [])
        self.assertEqual(env.sim.orientation_calls, [])
        self.assertEqual(len(env.sim.velocity_calls), 2)

    def test_timeout_truncates_without_counting_as_capture_win(self):
        env = self.make_env()
        env.max_steps = 1

        result = env.step(VelocityCommand(0.0, 0.0), VelocityCommand(0.0, 0.0))

        self.assertTrue(result["truncated"])
        self.assertTrue(result["done"])
        self.assertEqual(result["info"]["reason"], "step_limit")
        self.assertFalse(result["info"]["win"])

    def test_compute_reward_passes_initial_distances_for_chase_shaping(self):
        env = self.make_env()
        env._radar = lambda _agent: [1.0] * 64
        captured = {}

        def fake_reward_calculate_chase(**kwargs):
            captured.update(kwargs)
            return 0.0, False, False, {"reason": "", "win": False}

        import envs.rewards as reward_fn

        original = reward_fn.reward_calculate_chase
        reward_fn.reward_calculate_chase = fake_reward_calculate_chase
        try:
            CoppeliaTrackEnv._compute_reward(env, defender_collision=False, attacker_collision=False)
        finally:
            reward_fn.reward_calculate_chase = original

        self.assertIn("initial_dist_def_att", captured)
        self.assertIn("initial_dist_def_tgt", captured)
        self.assertGreater(captured["initial_dist_def_att"], 0.0)
        self.assertGreater(captured["initial_dist_def_tgt"], 0.0)

    def test_real_chase_reward_accepts_initial_distance_kwargs_and_shapes_progress(self):
        env = self.make_env()
        env._radar = lambda _agent: [1.0] * 64
        env._last_state = SimState(
            defender={"x": 300.0, "y": 318.0, "theta": 0.0},
            attacker={"x": 418.0, "y": 318.0, "theta": 0.0},
            target={"x": 518.0, "y": 318.0, "theta": 0.0},
            step_count=0,
        )

        reward, terminated, truncated, info = CoppeliaTrackEnv._compute_reward(
            env,
            defender_collision=False,
            attacker_collision=False,
        )

        self.assertGreater(reward, -0.08)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertFalse(info.get("win", False))

    def test_default_tad_reward_accepts_initial_distance_kwargs(self):
        env = self.make_env()
        env.reward_mode = "hrl"
        env._radar = lambda _agent: [1.0] * 64

        reward, terminated, truncated, info = CoppeliaTrackEnv._compute_reward(
            env,
            defender_collision=False,
            attacker_collision=False,
        )

        self.assertIsInstance(reward, float)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertIn("capture_progress_defender", info)

    def test_configures_remote_api_socket_timeout(self):
        env = self.make_env()
        env.client = FakeTimeoutClient()
        env.remote_timeout_ms = 1234

        fake_zmq = SimpleNamespace(RCVTIMEO=1, SNDTIMEO=2, LINGER=3)
        with patch.dict(sys.modules, {"zmq": fake_zmq}):
            env._configure_remote_timeout()

        values = [value for _option, value in env.client.socket.options]
        self.assertIn(1234, values)
        self.assertIn(0, values)

    def test_default_attacker_command_uses_safe_alternative_when_direct_path_is_blocked(self):
        env = self.make_env()
        env.motion_model = "wheel_velocity"
        env._state = SimState(
            defender={"x": 0.0, "y": 100.0, "theta": 0.0},
            attacker={"x": 100.0, "y": 100.0, "theta": 0.0},
            target={"x": 200.0, "y": 100.0, "theta": 0.0},
            step_count=0,
        )
        env.attacker_max_v = 0.40
        env.attacker_max_w = 4.0
        env.sim_step_dt = 0.05
        env.scale = 0.01
        calls = []

        def fake_blocked(x, y, padding=0.0):
            calls.append((x, y, padding))
            return x > 101.0 and abs(y - 102.0) < 0.05

        with (
            patch("coppelia_env.track_env.env_lib._OBS_COMPILED", True),
            patch("coppelia_env.track_env.env_lib.is_point_blocked", side_effect=fake_blocked),
        ):
            cmd = CoppeliaTrackEnv._default_attacker_command(env)

        self.assertGreater(len(calls), 1)
        self.assertLess(cmd.linear_mps, env.attacker_max_v)
        self.assertNotAlmostEqual(cmd.angular_radps, 0.0)

    def test_default_attacker_command_stops_when_all_candidates_are_blocked(self):
        env = self.make_env()
        env.motion_model = "wheel_velocity"
        env._state = SimState(
            defender={"x": 0.0, "y": 100.0, "theta": 0.0},
            attacker={"x": 100.0, "y": 100.0, "theta": 0.0},
            target={"x": 200.0, "y": 100.0, "theta": 0.0},
            step_count=0,
        )

        with (
            patch("coppelia_env.track_env.env_lib._OBS_COMPILED", True),
            patch("coppelia_env.track_env.env_lib.is_point_blocked", return_value=True),
        ):
            cmd = CoppeliaTrackEnv._default_attacker_command(env)

        self.assertEqual(cmd.linear_mps, 0.0)

    def test_defender_command_is_safed_before_wheel_velocity_application(self):
        env = self.make_env(sim=FakeWheelSim())
        env.motion_model = "wheel_velocity"
        env._state = SimState(
            defender={"x": 100.0, "y": 100.0, "theta": 0.0},
            attacker={"x": 200.0, "y": 100.0, "theta": 0.0},
            target={"x": 300.0, "y": 100.0, "theta": 0.0},
            step_count=0,
        )

        with (
            patch("coppelia_env.track_env.env_lib._OBS_COMPILED", True),
            patch("coppelia_env.track_env.env_lib.is_point_blocked", return_value=True),
        ):
            CoppeliaTrackEnv._apply_agent_cmd(env, "defender", VelocityCommand(0.4, 0.0))

        self.assertEqual(len(env.sim.joint_target_velocity_calls), 2)
        self.assertEqual(env.sim.joint_target_velocity_calls[0][1], 0.0)
        self.assertEqual(env.sim.joint_target_velocity_calls[1][1], 0.0)

if __name__ == "__main__":
    unittest.main()
