"""Non-Gym CoppeliaSim TrackMaker environment.

The environment exposes robot-like velocity commands and reads state back from
CoppeliaSim. It intentionally does not inherit from Gym; training code should
adapt to the small reset/step protocol here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from configs import map_config
from configs.map_config import EnvParameters
from coppelia_env.turtlebot4 import TurtleBot4Spec, cmd_vel_to_wheel_speeds
from envs import env_lib
from envs import rewards as reward_fn
from coppeliasim.tools.coppeliasim_export_trackmaker_scene import _sample_scene_entities


def _remote_api_client_class():
    """Load the optional simulator client only when a connection is requested."""

    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    except ModuleNotFoundError as exc:
        if exc.name != "coppeliasim_zmqremoteapi_client":
            raise
        raise ModuleNotFoundError(
            "CoppeliaTrackEnv.connect() requires the optional "
            "'coppeliasim_zmqremoteapi_client' package"
        ) from exc
    return RemoteAPIClient


@dataclass
class VelocityCommand:
    """Robot velocity command in SI units."""

    linear_mps: float = 0.0
    angular_radps: float = 0.0


@dataclass
class SimState:
    defender: dict[str, float]
    attacker: dict[str, float]
    target: dict[str, float]
    step_count: int


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _normalize_angle_deg(angle: float) -> float:
    angle = float(angle) % 360.0
    if angle > 180.0:
        angle -= 360.0
    return angle


class CoppeliaTrackEnv:
    """TrackMaker task backed by CoppeliaSim object state.

    This is a validation bridge, not a Gym wrapper. It applies differential-drive
    commands to CoppeliaSim objects and reads poses and collisions back from the
    simulator. The command contract uses SI linear and angular velocities.
    """

    def __init__(
        self,
        manifest: str | Path = "coppeliasim/scenes/trackmaker_turtlebot4_scene.json",
        host: str = "127.0.0.1",
        port: int = 23000,
        sim_step_dt: float | None = None,
        reward_mode: str = "hrl",
        start_simulation: bool = True,
        stepping: bool = True,
        connect: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.host = str(host)
        self.port = int(port)
        self.client = None
        self.sim = None
        self.handles: dict[str, int] = {}
        self.robot_specs: dict[str, TurtleBot4Spec] = {}

        meta = dict(self.manifest.get("meta", {}) or {})
        map_data = dict(self.manifest.get("map", {}) or {})
        self.scale = float(map_data.get("scale_m_per_px", meta.get("scale_m_per_px", 0.01)))
        self.sim_step_dt = float(sim_step_dt if sim_step_dt is not None else meta.get("sim_step_dt", 0.05))
        self.reward_mode = str(reward_mode)
        self.start_simulation = bool(start_simulation)
        self.stepping = bool(stepping)
        self.remote_timeout_ms = int(meta.get("remote_timeout_ms", 30000))
        self.motion_model = str(meta.get("motion_model", meta.get("robot_model", "kinematic_velocity_base"))).strip().lower()
        self.physics_mode = str(meta.get("physics_mode", "")).strip().lower()
        self.dynamic_linear_gain = float(meta.get("dynamic_linear_gain", 8.0))
        self.dynamic_angular_gain = float(meta.get("dynamic_angular_gain", 0.8))
        self.dynamic_z_gain = float(meta.get("dynamic_z_gain", 40.0))
        self.dynamic_attitude_gain = float(meta.get("dynamic_attitude_gain", 8.0))
        self.dynamic_max_force = float(meta.get("dynamic_max_force", 12.0))
        self.dynamic_max_torque = float(meta.get("dynamic_max_torque", 2.0))
        self.dynamic_nominal_z = float(meta.get("dynamic_nominal_z", 0.082))
        self.dynamic_reset_settle_steps = int(meta.get("dynamic_reset_settle_steps", 3))
        self._configure_robot_specs()

        self.width = int(map_data.get("width_px", map_config.width))
        self.height = int(map_data.get("height_px", map_config.height))
        self.pixel_size = float(meta.get("pixel_size_px", getattr(map_config, "pixel_size", 4.0)))
        self.robot_radius_px = float(meta.get("robot_radius_px", getattr(map_config, "agent_radius", 8.0)))
        self.target_radius_px = float(meta.get("target_radius_px", getattr(map_config, "target_radius", 16.0)))
        self.capture_radius_px = float(meta.get("capture_radius_px", getattr(map_config, "capture_radius", 20.0)))
        self.capture_sector_angle_deg = float(
            meta.get("capture_sector_angle_deg", getattr(map_config, "capture_sector_angle_deg", 30.0))
        )
        self.capture_required_steps = int(meta.get("capture_required_steps", getattr(map_config, "capture_required_steps", 1)))
        self.max_steps = int(meta.get("episode_len", EnvParameters.EPISODE_LEN))
        self.radar_rays = int(meta.get("radar_rays", getattr(EnvParameters, "RADAR_RAYS", 64)))
        self.sensor_range_m = float(
            meta.get(
                "sensor_range_m",
                min(getattr(EnvParameters, "FOV_RANGE", 300), math.hypot(self.width, self.height)) * self.scale,
            )
        )

        self.defender_max_v = float(meta.get("defender_speed_m_per_s", 0.52))
        self.attacker_max_v = float(meta.get("attacker_speed_m_per_s", 0.40))
        self.defender_max_w = float(meta.get("defender_max_turn_rad_per_s", math.radians(6.0) / self.sim_step_dt))
        self.attacker_max_w = float(meta.get("attacker_max_turn_rad_per_s", math.radians(12.0) / self.sim_step_dt))

        self._def_capture_count = 0
        self._att_capture_count = 0
        self._last_state: SimState | None = None
        self._state: SimState | None = None
        self.initial_dist_def_att: float | None = None
        self.initial_dist_def_tgt: float | None = None
        self._done = False

        self._configure_map()
        if connect:
            self.connect()

    def connect(self) -> None:
        if self.sim is not None:
            return
        remote_api_client = _remote_api_client_class()
        self.client = remote_api_client(host=self.host, port=self.port)
        self._configure_remote_timeout()
        self.sim = self.client.require("sim")
        if self.stepping:
            self.sim.setStepping(True)
        try:
            self.sim.setFloatParam(self.sim.floatparam_simulation_time_step, self.sim_step_dt)
        except Exception:
            pass
        self.handles = self._resolve_handles()
        if self.start_simulation and self.sim.getSimulationState() == self.sim.simulation_stopped:
            self.sim.startSimulation()
            if self.stepping:
                self.client.step(wait=True)

    def close(self) -> None:
        if self.sim is None:
            return
        try:
            try:
                if self.sim.getSimulationState() != self.sim.simulation_stopped:
                    self.sim.stopSimulation()
                    self._wait_for_simulation_stopped()
            except Exception:
                pass
        finally:
            self.sim = None
            self.client = None

    def _configure_remote_timeout(self) -> None:
        socket = getattr(self.client, "socket", None)
        if socket is None:
            return
        try:
            import zmq

            socket.setsockopt(zmq.RCVTIMEO, int(self.remote_timeout_ms))
            socket.setsockopt(zmq.SNDTIMEO, int(self.remote_timeout_ms))
            socket.setsockopt(zmq.LINGER, 0)
        except Exception:
            pass

    def _wait_for_simulation_stopped(self, max_steps: int = 200) -> None:
        if self.sim is None:
            return
        for _ in range(max(1, int(max_steps))):
            try:
                if self.sim.getSimulationState() == self.sim.simulation_stopped:
                    return
            except Exception:
                return
            if self.stepping and self.client is not None:
                try:
                    self.client.step(wait=True)
                    continue
                except Exception:
                    return
            time.sleep(max(0.001, min(0.05, self.sim_step_dt)))

    def reset(self, seed: int | None = None, entities: dict[str, Any] | None = None) -> dict[str, Any]:
        self._require_connected()
        if seed is None:
            seed = int(dict(self.manifest.get("meta", {}) or {}).get("spawn_seed", 0))
        if entities is None:
            entities = self._sample_entities(int(seed))

        self._def_capture_count = 0
        self._att_capture_count = 0
        self._done = False
        self._last_state = None

        defender = self._entity_to_state(entities["defender"])
        attacker = self._entity_to_state(entities["attacker"])
        target = self._entity_to_state(entities["target"])
        self._set_agent_state("defender", defender)
        self._set_agent_state("attacker", attacker)
        self._set_target_state(target)
        self._zero_agent_velocity("defender")
        self._zero_agent_velocity("attacker")
        if self.stepping:
            for _ in range(max(1, self.dynamic_reset_settle_steps if self._uses_dynamic_velocity() else 1)):
                self.client.step(wait=True)
        self._state = SimState(defender=defender, attacker=attacker, target=target, step_count=0)
        self._last_state = self._copy_state(self._state)
        self._set_initial_distances(self._state)
        return self._transition(0.0, False, False, {"reason": "", "win": False})

    def step(
        self,
        defender_cmd: VelocityCommand | dict[str, float] | tuple[float, float] | list[float],
        attacker_cmd: VelocityCommand | dict[str, float] | tuple[float, float] | list[float] | None = None,
    ) -> dict[str, Any]:
        self._require_connected()
        if self._state is None:
            self.reset()
        if self._done:
            return self._transition(0.0, True, False, {"reason": "already_done", "win": False})

        self._last_state = self._copy_state(self._state)
        d_cmd = self._coerce_cmd(defender_cmd, self.defender_max_v, self.defender_max_w)
        a_cmd = self._coerce_cmd(
            attacker_cmd if attacker_cmd is not None else self._default_attacker_command(),
            self.attacker_max_v,
            self.attacker_max_w,
        )

        self._apply_agent_cmd("defender", d_cmd)
        self._apply_agent_cmd("attacker", a_cmd)
        if self.stepping:
            self.client.step(wait=True)

        state = self._read_state()
        state.step_count = int(self._state.step_count + 1)
        self._state = state

        defender_collision = self._check_collision("defender")
        attacker_collision = self._check_collision("attacker")
        defender_captured = self._defender_captures_attacker()
        attacker_captured = self._attacker_captures_target()
        self._def_capture_count = min(self._def_capture_count + 1, self.capture_required_steps) if defender_captured else 0
        self._att_capture_count = min(self._att_capture_count + 1, self.capture_required_steps) if attacker_captured else 0

        reward, terminated, truncated, info = self._compute_reward(
            defender_collision=defender_collision,
            attacker_collision=attacker_collision,
        )
        if state.step_count >= self.max_steps and not terminated:
            truncated = True
            info = dict(info)
            info["reason"] = "step_limit"
            info["win"] = False
        self._done = bool(terminated or truncated)
        return self._transition(float(reward), bool(terminated), bool(truncated), info)

    def get_state(self) -> dict[str, Any]:
        if self._state is None:
            return {}
        return asdict(self._state)

    def _transition(self, reward: float, terminated: bool, truncated: bool, info: dict[str, Any]) -> dict[str, Any]:
        obs = self._observation()
        info = dict(info or {})
        return {
            "obs": obs,
            "state": self.get_state(),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "done": bool(terminated or truncated),
            "info": info,
        }

    def _configure_map(self) -> None:
        map_data = dict(self.manifest.get("map", {}) or {})
        map_config.width = self.width
        map_config.height = self.height
        map_config.map_diagonal = math.hypot(float(self.width), float(self.height))
        map_config.pixel_size = self.pixel_size
        map_config.agent_radius = self.robot_radius_px
        map_config.target_radius = self.target_radius_px
        map_config.capture_radius = self.capture_radius_px
        map_config.capture_sector_angle_deg = self.capture_sector_angle_deg
        map_config.obstacles[:] = [dict(obs) for obs in self.manifest.get("obstacles", [])]
        env_lib.build_occupancy(
            width=self.width,
            height=self.height,
            cell=getattr(map_config, "occ_cell", self.pixel_size),
            obstacles=map_config.obstacles,
        )

    def _resolve_handles(self) -> dict[str, int]:
        objects = dict(self.manifest.get("objects", {}) or {})
        handles = {}
        for key, spec in objects.items():
            if isinstance(spec, dict) and spec.get("alias"):
                try:
                    handles[key] = self.sim.getObject("/" + str(spec["alias"]))
                except Exception:
                    pass
        robots = dict(self.manifest.get("robots", {}) or {})
        for role, robot in robots.items():
            if not isinstance(robot, dict):
                continue
            alias_keys = {
                role: "root",
                f"{role}_base": "base_link",
                f"{role}_collision_proxy": "collision_proxy",
                f"{role}_left_wheel_joint": "left_wheel_joint",
                f"{role}_right_wheel_joint": "right_wheel_joint",
                f"{role}_lidar": "lidar_sensor",
                f"{role}_vision": "vision_sensor",
            }
            for handle_key, robot_key in alias_keys.items():
                alias = robot.get(robot_key)
                if not alias:
                    continue
                try:
                    handles[handle_key] = self.sim.getObject("/" + str(alias))
                except Exception:
                    pass
        required = ("defender", "attacker", "target")
        missing = [key for key in required if key not in handles]
        if missing:
            raise RuntimeError(f"CoppeliaTrackEnv missing scene objects: {missing}")
        return handles

    def _configure_robot_specs(self) -> None:
        self.robot_specs = {}
        robots = dict(self.manifest.get("robots", {}) or {})
        meta_turtlebot = dict(dict(self.manifest.get("meta", {}) or {}).get("turtlebot4", {}) or {})
        for role in ("defender", "attacker"):
            robot = dict(robots.get(role, {}) or {})
            if not robot:
                continue
            try:
                self.robot_specs[role] = TurtleBot4Spec(
                    base_link=str(meta_turtlebot.get("base_link", "base_link")),
                    left_wheel_joint=str(meta_turtlebot.get("left_wheel_joint", "left_wheel_joint")),
                    right_wheel_joint=str(meta_turtlebot.get("right_wheel_joint", "right_wheel_joint")),
                    lidar_link=str(meta_turtlebot.get("lidar_link", "rplidar_link")),
                    rgb_camera_frame=str(meta_turtlebot.get("rgb_camera_frame", "oakd_rgb_camera_frame")),
                    wheel_radius_m=float(robot.get("wheel_radius_m", meta_turtlebot.get("wheel_radius_m"))),
                    wheel_separation_m=float(robot.get("wheel_separation_m", meta_turtlebot.get("wheel_separation_m"))),
                )
            except (TypeError, ValueError):
                continue

    def _sample_entities(self, seed: int) -> dict[str, dict[str, float]]:
        obstacles = [dict(obs) for obs in self.manifest.get("obstacles", [])]
        raw = _sample_scene_entities(obstacles, int(seed), self.robot_radius_px, self.target_radius_px)
        return {
            key: {"center_px": [float(v["cx"]), float(v["cy"])], "theta_deg": float(v.get("theta", 0.0))}
            for key, v in raw.items()
        }

    def _entity_to_state(self, entity: dict[str, Any]) -> dict[str, float]:
        if "center_px" in entity:
            cx, cy = entity["center_px"]
        else:
            cx, cy = entity["cx"], entity["cy"]
        return {
            "x": float(cx) - self.pixel_size * 0.5,
            "y": float(cy) - self.pixel_size * 0.5,
            "theta": float(entity.get("theta_deg", entity.get("theta", 0.0))) % 360.0,
        }

    def _state_to_world(self, state: dict[str, float]) -> tuple[float, float, float]:
        cx = float(state["x"]) + self.pixel_size * 0.5
        cy = float(state["y"]) + self.pixel_size * 0.5
        x = (cx - self.width * 0.5) * self.scale
        y = (self.height * 0.5 - cy) * self.scale
        yaw = -math.radians(float(state.get("theta", 0.0)))
        return x, y, yaw

    def _world_to_state(self, handle: int) -> dict[str, float]:
        pos = self.sim.getObjectPosition(handle, -1)
        ori = self.sim.getObjectOrientation(handle, -1)
        cx = float(pos[0]) / self.scale + self.width * 0.5
        cy = self.height * 0.5 - float(pos[1]) / self.scale
        theta = (-math.degrees(float(ori[2]))) % 360.0
        return {"x": cx - self.pixel_size * 0.5, "y": cy - self.pixel_size * 0.5, "theta": theta}

    def _set_agent_state(self, key: str, state: dict[str, float]) -> None:
        x, y, yaw = self._state_to_world(state)
        handle = self.handles[key]
        self.sim.setObjectPosition(handle, -1, [x, y, self._object_z(handle)])
        self.sim.setObjectOrientation(handle, -1, [0.0, 0.0, yaw])

    def _set_target_state(self, state: dict[str, float]) -> None:
        x, y, yaw = self._state_to_world(state)
        handle = self.handles["target"]
        self.sim.setObjectPosition(handle, -1, [x, y, self._object_z(handle)])
        self.sim.setObjectOrientation(handle, -1, [0.0, 0.0, yaw])

    def _object_z(self, handle: int) -> float:
        try:
            return float(self.sim.getObjectPosition(handle, -1)[2])
        except Exception:
            return 0.05

    def _read_state(self) -> SimState:
        return SimState(
            defender=self._world_to_state(self._agent_pose_handle("defender")),
            attacker=self._world_to_state(self._agent_pose_handle("attacker")),
            target=self._world_to_state(self.handles["target"]),
            step_count=int(self._state.step_count if self._state is not None else 0),
        )

    def _apply_agent_cmd(self, key: str, cmd: VelocityCommand) -> None:
        cmd = self._safe_agent_command(key, cmd)
        if self._uses_wheel_velocity():
            self._apply_agent_wheel_velocity(key, cmd)
            return
        if self._uses_dynamic_velocity():
            self._apply_agent_velocity(key, cmd)
            return
        current = self._world_to_state(self.handles[key])
        theta = float(current["theta"])
        theta_next = (theta + math.degrees(float(cmd.angular_radps) * self.sim_step_dt)) % 360.0
        ds_px = float(cmd.linear_mps) * self.sim_step_dt / self.scale
        x_next = float(current["x"]) + ds_px * math.cos(math.radians(theta_next))
        y_next = float(current["y"]) + ds_px * math.sin(math.radians(theta_next))
        proposed = {"x": x_next, "y": y_next, "theta": theta_next}
        self._set_agent_state(key, proposed)

    def _uses_wheel_velocity(self) -> bool:
        return self.motion_model in {
            "wheel_velocity",
            "turtlebot4_wheel_velocity",
            "official_turtlebot4_wheel_velocity",
        }

    def _agent_pose_handle(self, key: str) -> int:
        if self._uses_wheel_velocity():
            return self.handles.get(f"{key}_base", self.handles[key])
        return self.handles[key]

    def _agent_collision_handle(self, key: str) -> int:
        if self._uses_wheel_velocity():
            return self.handles.get(
                f"{key}_collision_proxy",
                self.handles.get(f"{key}_base", self.handles[key]),
            )
        return self.handles[key]

    def _apply_agent_wheel_velocity(self, key: str, cmd: VelocityCommand) -> None:
        spec = self.robot_specs.get(key)
        if spec is None:
            raise RuntimeError(f"missing TurtleBot4 wheel spec for {key}")
        left_handle = self.handles.get(f"{key}_left_wheel_joint")
        right_handle = self.handles.get(f"{key}_right_wheel_joint")
        if left_handle is None or right_handle is None:
            raise RuntimeError(f"missing TurtleBot4 wheel joint handles for {key}")
        max_v = self.defender_max_v if key == "defender" else self.attacker_max_v
        max_w = self.defender_max_w if key == "defender" else self.attacker_max_w
        left, right = cmd_vel_to_wheel_speeds(
            cmd.linear_mps,
            cmd.angular_radps,
            spec,
            max_linear_mps=max_v,
            max_angular_radps=max_w,
        )
        self.sim.setJointTargetVelocity(left_handle, left)
        self.sim.setJointTargetVelocity(right_handle, right)

    def _uses_dynamic_velocity(self) -> bool:
        if self.physics_mode == "dynamic":
            return True
        return self.motion_model in {
            "dynamic_velocity",
            "native_dynamic_velocity_base",
            "dynamic_velocity_base",
            "coppelia_dynamic_velocity_base",
        }

    def _apply_agent_velocity(self, key: str, cmd: VelocityCommand) -> None:
        handle = self.handles[key]
        current = self._world_to_state(handle)
        yaw = -math.radians(float(current.get("theta", 0.0)))
        linear = [
            float(cmd.linear_mps) * math.cos(yaw),
            float(cmd.linear_mps) * math.sin(yaw),
            0.0,
        ]
        angular = [0.0, 0.0, -float(cmd.angular_radps)]
        try:
            self.sim.setObjectVelocity(handle, linear, angular)
            return
        except AttributeError:
            pass
        except Exception:
            pass
        self._apply_agent_force_control(handle, linear, angular)

    def _apply_agent_force_control(self, handle: int, target_linear: list[float], target_angular: list[float]) -> None:
        if not hasattr(self.sim, "addForceAndTorque"):
            raise RuntimeError("Coppelia dynamic velocity mode requires setObjectVelocity or addForceAndTorque")
        try:
            current_linear, current_angular = self.sim.getObjectVelocity(handle)
        except Exception:
            current_linear, current_angular = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        try:
            mass = float(self.sim.getShapeMass(handle))
        except Exception:
            mass = 4.0
        pos = self.sim.getObjectPosition(handle, -1)
        ori = self.sim.getObjectOrientation(handle, -1)
        force = [
            _clip(
                (float(target_linear[i]) - float(current_linear[i])) * mass * self.dynamic_linear_gain,
                -self.dynamic_max_force,
                self.dynamic_max_force,
            )
            for i in range(2)
        ]
        z_force = _clip(
            (self.dynamic_nominal_z - float(pos[2])) * mass * self.dynamic_z_gain - float(current_linear[2]) * mass,
            -self.dynamic_max_force,
            self.dynamic_max_force,
        )
        force.append(z_force)
        torque = [
            _clip(-float(ori[0]) * mass * self.dynamic_attitude_gain - float(current_angular[0]) * mass,
                  -self.dynamic_max_torque, self.dynamic_max_torque),
            _clip(-float(ori[1]) * mass * self.dynamic_attitude_gain - float(current_angular[1]) * mass,
                  -self.dynamic_max_torque, self.dynamic_max_torque),
            _clip(
                (float(target_angular[2]) - float(current_angular[2])) * mass * self.dynamic_angular_gain,
                -self.dynamic_max_torque,
                self.dynamic_max_torque,
            ),
        ]
        self.sim.addForceAndTorque(handle, force, torque)

    def _zero_agent_velocity(self, key: str) -> None:
        if self._uses_wheel_velocity():
            left_handle = self.handles.get(f"{key}_left_wheel_joint")
            right_handle = self.handles.get(f"{key}_right_wheel_joint")
            if left_handle is not None:
                self.sim.setJointTargetVelocity(left_handle, 0.0)
            if right_handle is not None:
                self.sim.setJointTargetVelocity(right_handle, 0.0)
            return
        if not self._uses_dynamic_velocity() or key not in self.handles:
            return
        try:
            self.sim.setObjectVelocity(self.handles[key], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            return
        except Exception:
            pass
        try:
            self.sim.resetDynamicObject(self.handles[key])
        except Exception:
            pass

    def _coerce_cmd(self, value: Any, max_v: float, max_w: float) -> VelocityCommand:
        if isinstance(value, VelocityCommand):
            v, w = value.linear_mps, value.angular_radps
        elif isinstance(value, dict):
            v = value.get("linear_mps", value.get("v", value.get("linear", 0.0)))
            w = value.get("angular_radps", value.get("w", value.get("angular", 0.0)))
        else:
            arr = np.asarray(value if value is not None else [0.0, 0.0], dtype=np.float32).reshape(-1)
            v = float(arr[0]) if arr.size >= 1 else 0.0
            w = float(arr[1]) if arr.size >= 2 else 0.0
        return VelocityCommand(_clip(float(v), -max_v, max_v), _clip(float(w), -max_w, max_w))

    def _default_attacker_command(self) -> VelocityCommand:
        state = self._state
        if state is None:
            return VelocityCommand()
        ax = state.attacker["x"] + self.pixel_size * 0.5
        ay = state.attacker["y"] + self.pixel_size * 0.5
        tx = state.target["x"] + self.pixel_size * 0.5
        ty = state.target["y"] + self.pixel_size * 0.5
        desired = math.degrees(math.atan2(ty - ay, tx - ax))
        err = _normalize_angle_deg(desired - state.attacker["theta"])
        w = _clip(math.radians(err) / max(self.sim_step_dt, 1e-6), -self.attacker_max_w, self.attacker_max_w)
        turn_scale = max(0.15, 1.0 - min(abs(err), 90.0) / 120.0)
        direct = VelocityCommand(self.attacker_max_v * turn_scale, w)
        return self._safe_agent_command("attacker", direct)

    def _safe_agent_command(self, key: str, desired: VelocityCommand) -> VelocityCommand:
        if self._agent_command_is_safe(key, desired):
            return desired
        max_w = self.attacker_max_w if key == "attacker" else self.defender_max_w
        turn_offsets = [0.35, -0.35, 0.7, -0.7, 1.0, -1.0]
        speed_scales = [0.75, 0.5, 0.25]
        for scale in speed_scales:
            for offset in turn_offsets:
                candidate = VelocityCommand(
                    linear_mps=float(desired.linear_mps) * float(scale),
                    angular_radps=_clip(
                        float(desired.angular_radps) + float(offset) * max_w,
                        -max_w,
                        max_w,
                    ),
                )
                if self._agent_command_is_safe(key, candidate):
                    return candidate
        return VelocityCommand(0.0, 0.0)

    def _agent_command_is_safe(self, key: str, cmd: VelocityCommand, lookahead_steps: int = 4) -> bool:
        if self._state is None:
            return True
        if not bool(getattr(env_lib, "_OBS_COMPILED", False)):
            return True
        state = dict(self._state.attacker if key == "attacker" else self._state.defender)
        for _ in range(max(1, int(lookahead_steps))):
            theta = (float(state.get("theta", 0.0)) + math.degrees(float(cmd.angular_radps) * self.sim_step_dt)) % 360.0
            ds_px = max(0.0, float(cmd.linear_mps)) * self.sim_step_dt / max(self.scale, 1e-9)
            x = float(state["x"]) + ds_px * math.cos(math.radians(theta))
            y = float(state["y"]) + ds_px * math.sin(math.radians(theta))
            cx = x + self.pixel_size * 0.5
            cy = y + self.pixel_size * 0.5
            if env_lib.is_point_blocked(cx, cy, padding=self.robot_radius_px):
                return False
            state = {"x": x, "y": y, "theta": theta}
        return True

    def _check_collision(self, key: str) -> bool:
        handle = self._agent_collision_handle(key)
        for obs_handle in self._obstacle_handles():
            try:
                result = self.sim.checkCollision(handle, obs_handle)
                if isinstance(result, (list, tuple)):
                    collides = int(result[0]) > 0
                else:
                    collides = int(result) > 0
                if collides:
                    return True
            except Exception:
                continue
        state = self._state.defender if key == "defender" else self._state.attacker
        cx = float(state["x"]) + self.pixel_size * 0.5
        cy = float(state["y"]) + self.pixel_size * 0.5
        return bool(env_lib.is_point_blocked(cx, cy, padding=self.robot_radius_px))

    def _obstacle_handles(self) -> list[int]:
        out = []
        for key, handle in self.handles.items():
            if key.startswith("obstacle_") or key == "floor":
                if key != "floor":
                    out.append(handle)
        return out

    def _defender_captures_attacker(self) -> bool:
        state = self._state
        dx = (state.attacker["x"] - state.defender["x"])
        dy = (state.attacker["y"] - state.defender["y"])
        dist = math.hypot(dx, dy)
        if dist > self.capture_radius_px:
            return False
        rel = _normalize_angle_deg(math.degrees(math.atan2(dy, dx)) - state.defender["theta"])
        if abs(rel) > self.capture_sector_angle_deg * 0.5:
            return False
        return not self._line_blocked(state.defender, state.attacker)

    def _attacker_captures_target(self) -> bool:
        state = self._state
        ax = state.attacker["x"] + self.pixel_size * 0.5
        ay = state.attacker["y"] + self.pixel_size * 0.5
        tx = state.target["x"] + self.pixel_size * 0.5
        ty = state.target["y"] + self.pixel_size * 0.5
        return math.hypot(tx - ax, ty - ay) <= (self.robot_radius_px + self.target_radius_px)

    def _line_blocked(self, a: dict[str, float], b: dict[str, float]) -> bool:
        x1 = a["x"] + self.pixel_size * 0.5
        y1 = a["y"] + self.pixel_size * 0.5
        x2 = b["x"] + self.pixel_size * 0.5
        y2 = b["y"] + self.pixel_size * 0.5
        length = math.hypot(x2 - x1, y2 - y1)
        if length <= 1e-6:
            return False
        angle = math.atan2(y2 - y1, x2 - x1)
        dist = env_lib.ray_distance_grid((x1, y1), angle, length, padding=0.0)
        return bool(dist < length - 1e-3)

    def _observation(self) -> dict[str, Any]:
        if self._state is None:
            return {}
        state = self._state
        return {
            "defender": self._agent_obs(state.defender, state.attacker, state.target),
            "attacker": self._agent_obs(state.attacker, state.defender, state.target),
            "privileged_state": asdict(state),
        }

    def _agent_obs(self, ego: dict[str, float], other: dict[str, float], target: dict[str, float]) -> dict[str, Any]:
        ecx = ego["x"] + self.pixel_size * 0.5
        ecy = ego["y"] + self.pixel_size * 0.5
        ocx = other["x"] + self.pixel_size * 0.5
        ocy = other["y"] + self.pixel_size * 0.5
        tcx = target["x"] + self.pixel_size * 0.5
        tcy = target["y"] + self.pixel_size * 0.5
        return {
            "pose_px": [float(ecx), float(ecy), float(ego["theta"])],
            "other_rel_px": [float(ocx - ecx), float(ocy - ecy)],
            "target_rel_px": [float(tcx - ecx), float(tcy - ecy)],
            "radar": self._radar(ego).tolist(),
        }

    def _radar(self, agent: dict[str, float]) -> np.ndarray:
        coppelia_radar = self._coppelia_sensor_radar(agent)
        if coppelia_radar is not None:
            return coppelia_radar
        center = np.array([agent["x"] + self.pixel_size * 0.5, agent["y"] + self.pixel_size * 0.5], dtype=float)
        heading = math.radians(agent.get("theta", 0.0))
        rays = int(getattr(self, "radar_rays", getattr(EnvParameters, "RADAR_RAYS", 64)))
        max_range = float(min(getattr(EnvParameters, "FOV_RANGE", 300), math.hypot(self.width, self.height)))
        angles = [heading + 2.0 * math.pi * i / rays for i in range(rays)]
        dists = env_lib.ray_distances_multi(center, angles, max_range, padding=0.0)
        return np.asarray(dists, dtype=np.float32)

    def _coppelia_sensor_radar(self, agent: dict[str, float]) -> np.ndarray | None:
        if not self._uses_wheel_velocity() or self.sim is None:
            return None
        role = "defender"
        if self._state is not None and agent is self._state.attacker:
            role = "attacker"
        sensor = self.handles.get(f"{role}_lidar")
        base = self.handles.get(f"{role}_base", self.handles.get(role))
        if sensor is None or base is None:
            return None
        rays = int(getattr(self, "radar_rays", getattr(EnvParameters, "RADAR_RAYS", 64)))
        max_range_m = float(getattr(self, "sensor_range_m", 3.0))
        if rays <= 0 or max_range_m <= 0.0:
            return None
        try:
            base_matrix = self.sim.getObjectMatrix(base, -1)
        except Exception:
            return None
        origin = [float(base_matrix[3]), float(base_matrix[7]), float(base_matrix[11]) + 0.18]
        heading = -math.radians(float(agent.get("theta", 0.0)))
        distances = []
        for idx in range(rays):
            angle = heading + 2.0 * math.pi * idx / rays
            direction = [math.cos(angle), math.sin(angle), 0.0]
            matrix = self._ray_sensor_matrix(origin, direction)
            try:
                self.sim.setObjectMatrix(sensor, -1, matrix)
                result = self.sim.handleProximitySensor(sensor)
            except Exception:
                return None
            detected = int(result[0]) > 0 if isinstance(result, (list, tuple)) and result else False
            distance_m = float(result[1]) if detected and len(result) > 1 else max_range_m
            distance_m = _clip(distance_m, 0.0, max_range_m)
            distances.append(distance_m / max(self.scale, 1e-9))
        return np.asarray(distances, dtype=np.float32)

    def _ray_sensor_matrix(self, origin: list[float], direction: list[float]) -> list[float]:
        dx, dy, dz = (float(direction[0]), float(direction[1]), float(direction[2]))
        norm = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1e-9)
        z_axis = [dx / norm, dy / norm, dz / norm]
        up = [0.0, 0.0, 1.0]
        x_axis = [
            up[1] * z_axis[2] - up[2] * z_axis[1],
            up[2] * z_axis[0] - up[0] * z_axis[2],
            up[0] * z_axis[1] - up[1] * z_axis[0],
        ]
        x_norm = math.sqrt(sum(v * v for v in x_axis))
        if x_norm <= 1e-9:
            x_axis = [1.0, 0.0, 0.0]
        else:
            x_axis = [v / x_norm for v in x_axis]
        y_axis = [
            z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
            z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
            z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
        ]
        return [
            x_axis[0], y_axis[0], z_axis[0], float(origin[0]),
            x_axis[1], y_axis[1], z_axis[1], float(origin[1]),
            x_axis[2], y_axis[2], z_axis[2], float(origin[2]),
        ]

    def _compute_reward(self, defender_collision: bool, attacker_collision: bool) -> tuple[float, bool, bool, dict[str, Any]]:
        state = self._state
        prev = self._last_state or state
        kwargs = dict(
            defender=state.defender,
            attacker=state.attacker,
            target=state.target,
            prev_defender=prev.defender,
            prev_attacker=prev.attacker,
            defender_collision=bool(defender_collision),
            attacker_collision=bool(attacker_collision),
            defender_captured=bool(self._def_capture_count >= self.capture_required_steps),
            attacker_captured=bool(self._att_capture_count >= self.capture_required_steps),
            capture_progress_defender=int(self._def_capture_count),
            capture_progress_attacker=int(self._att_capture_count),
            capture_required_steps=int(self.capture_required_steps),
            radar=self._radar(state.defender),
            initial_dist_def_att=self.initial_dist_def_att,
            initial_dist_def_tgt=self.initial_dist_def_tgt,
        )
        if self.reward_mode == "baseline":
            return reward_fn.reward_calculate_baseline(**kwargs)
        if self.reward_mode == "chase":
            return reward_fn.reward_calculate_chase(**kwargs)
        return reward_fn.reward_calculate_tad(**kwargs)

    def _set_initial_distances(self, state: SimState) -> None:
        dcx = float(state.defender["x"]) + self.pixel_size * 0.5
        dcy = float(state.defender["y"]) + self.pixel_size * 0.5
        acx = float(state.attacker["x"]) + self.pixel_size * 0.5
        acy = float(state.attacker["y"]) + self.pixel_size * 0.5
        tcx = float(state.target["x"]) + self.pixel_size * 0.5
        tcy = float(state.target["y"]) + self.pixel_size * 0.5
        self.initial_dist_def_att = float(math.hypot(dcx - acx, dcy - acy))
        reach_radius = float(self.robot_radius_px + self.target_radius_px)
        self.initial_dist_def_tgt = float(max(0.0, math.hypot(dcx - tcx, dcy - tcy) - reach_radius))

    def _copy_state(self, state: SimState) -> SimState:
        return SimState(
            defender=dict(state.defender),
            attacker=dict(state.attacker),
            target=dict(state.target),
            step_count=int(state.step_count),
        )

    def _require_connected(self) -> None:
        if self.sim is None or self.client is None:
            raise RuntimeError("CoppeliaTrackEnv is not connected. Call connect() first.")
