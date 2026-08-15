#!/usr/bin/env python3
"""Run frozen Defender HRL and goal-rush Attacker policies over ROS 2 topics."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from attacker.learned_policy import LearnedAttackerPolicy  # noqa: E402
from coppelia_env.ros2_deployment import (  # noqa: E402
    CheckpointSpec,
    DefenderObservationMemory,
    DeploymentConfig,
    EpisodeMonitor,
    LinearSlewLimiter,
    Pose2D,
    TerminalResult,
    VelocityCommand,
    build_policy_observations,
    deployment_input_timing,
    normalize_laser_scan,
    normalized_action_to_command,
    quaternion_to_yaw,
    validate_checkpoint,
)
from coppelia_env.skill_adapter import load_skill_model  # noqa: E402
from configs.skill_config import NetParameters  # noqa: E402

try:
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from geometry_msgs.msg import PoseStamped, Twist
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import JointState, LaserScan
    from std_msgs.msg import Bool, String
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in the deployment environment.
    raise SystemExit("Run this tool in the ros2humble environment with rclpy available") from exc


DEFAULT_TOP = PROJECT_ROOT / "models/hrl_ch2_m1_astar_cached_top_20260606_170036/best_model.pth"
DEFAULT_PROTECT = PROJECT_ROOT / "models/defender_protect_mlp_ctde_repro_20260526/final_model.pth"
DEFAULT_CHASE = PROJECT_ROOT / "models/defender_chase_nmn_dual_gru_raw_dense_05-05-19-12/final_model.pth"
DEFAULT_ATTACKER = PROJECT_ROOT / "models/attacker_nmn_mlp_diversity_continuation_20260715_120331/best_goal_rush.pth"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/coppeliasim/ros2_demo/episode.json"

CHECKSUMS = {
    "top": "b1176762ff55571d7ab3f695c73a3e313fce8b1dc0dad11bc559d90c3046a39e",
    "protect": "902dc161b78f4a2c4bc5e45020c0630a1c925a025998a3bc7db535631fe50b53",
    "chase": "eb05578fae6498fd5379ff15efef9c99a0e25b1233dcf577929f7e80150e6a92",
    "attacker": "4081f92fbf80e689ee64b76d53ed1bda53511f4a7286e7e536019953df7dcd01",
}

SKILL_NAMES = ("protect", "chase")


def decode_skill_index(action: np.ndarray | list[float] | tuple[float, ...]) -> str:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    raw = int(round(float(arr[0]))) if arr.size else 0
    return SKILL_NAMES[max(0, min(raw, len(SKILL_NAMES) - 1))]


def load_hrl_top_model(checkpoint: Path, network_type: str, device: torch.device):
    old = (
        int(getattr(NetParameters, "HRL_NUM_SKILLS", 2)),
        tuple(getattr(NetParameters, "HRL_DURATION_BINS", (1,))),
        int(getattr(NetParameters, "HRL_NUM_DURATION_BINS", 1)),
        int(getattr(NetParameters, "HRL_TOP_DISCRETE_ACTION_DIM", 2)),
    )
    NetParameters.HRL_NUM_SKILLS = 2
    NetParameters.HRL_DURATION_BINS = (1,)
    NetParameters.HRL_NUM_DURATION_BINS = 1
    NetParameters.HRL_TOP_DISCRETE_ACTION_DIM = 2
    try:
        return load_skill_model(checkpoint, network_type, device=device, global_model=False)
    finally:
        (
            NetParameters.HRL_NUM_SKILLS,
            NetParameters.HRL_DURATION_BINS,
            NetParameters.HRL_NUM_DURATION_BINS,
            NetParameters.HRL_TOP_DISCRETE_ACTION_DIM,
        ) = old


def _json_action(action: Any) -> list[float]:
    return [float(v) for v in np.asarray(action, dtype=np.float32).reshape(-1)]


def _twist(command: VelocityCommand) -> Twist:
    msg = Twist()
    msg.linear.x = float(command.linear_mps)
    msg.angular.z = float(command.angular_radps)
    return msg


def _pose_dict(pose: Pose2D) -> dict[str, float]:
    return {"x": float(pose.x), "y": float(pose.y), "yaw": float(pose.yaw)}


class TrackMakerPolicyNode(Node):
    def __init__(
        self,
        *,
        config: DeploymentConfig,
        top_model: Any,
        bottom_models: dict[str, Any],
        attacker_policy: LearnedAttackerPolicy,
        checkpoint_metadata: dict[str, Any],
        output_json: Path,
        episode_seed: int,
        obstacles: list[dict[str, Any]],
        use_sim_time: bool,
    ) -> None:
        super().__init__(
            "trackmaker_policy",
            parameter_overrides=[Parameter("use_sim_time", value=bool(use_sim_time))],
            automatically_declare_parameters_from_overrides=True,
        )
        self.config = config
        self.use_sim_time = bool(use_sim_time)
        self.top_model = top_model
        self.bottom_models = bottom_models
        self.attacker_policy = attacker_policy
        self.checkpoint_metadata = checkpoint_metadata
        self.output_json = Path(output_json).resolve()
        self.episode_seed = int(episode_seed)
        self.obstacles = list(obstacles)
        self.monitor = EpisodeMonitor(config)
        self.defender_memory = DefenderObservationMemory()
        self.attacker_slew = LinearSlewLimiter(config.attacker_max_accel_mps2)

        self.poses: dict[str, Pose2D] = {}
        self.radars: dict[str, np.ndarray] = {}
        self.input_stamps: dict[str, float] = {}
        self.last_input_timing = deployment_input_timing([], now=0.0, config=config)
        self.wheel_telemetry: dict[str, dict[str, dict[str, Any]]] = {
            role: {} for role in ("defender", "attacker")
        }
        self.actuator_states: dict[str, dict[str, Any]] = {}
        self.profile_metadata: dict[str, Any] = {}
        self.collisions = {"defender": False, "attacker": False}
        self.commands = {"defender": VelocityCommand(), "attacker": VelocityCommand()}
        self.last_decision_at = 0.0
        self.episode_started_at = time.monotonic()
        self.episode_started_control_s = self._control_now()
        self.episode_started_sim_s: float | None = None
        self.terminal: TerminalResult | None = None
        self.terminal_at: float | None = None
        self.terminal_control_at: float | None = None
        self.shutdown_requested = False
        self.records: list[dict[str, Any]] = []
        self.skill_counts = {"protect": 0, "chase": 0}
        self.last_skill = "waiting"
        self.last_status = "waiting_for_inputs"
        self._logged_ready = False

        self.defender_pub = self.create_publisher(Twist, "/defender/cmd_vel", 10)
        self.attacker_pub = self.create_publisher(Twist, "/attacker/cmd_vel", 10)
        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.outcome_pub = self.create_publisher(String, "/demo/outcome", latched)
        self.skill_pub = self.create_publisher(String, "/demo/selected_skill", 10)
        self.diagnostics_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.create_subscription(String, "/demo/profile_metadata", self._profile_callback, latched)

        for role in ("defender", "attacker", "target"):
            self.create_subscription(
                PoseStamped,
                f"/tracking/{role}/pose",
                lambda msg, role=role: self._pose_callback(role, msg),
                10,
            )
        for role in ("defender", "attacker"):
            self.create_subscription(
                LaserScan,
                f"/{role}/scan",
                lambda msg, role=role: self._scan_callback(role, msg),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Bool,
                f"/{role}/collision",
                lambda msg, role=role: self._collision_callback(role, msg),
                10,
            )
            self.create_subscription(
                JointState,
                f"/{role}/joint_targets",
                lambda msg, role=role: self._joint_callback(role, "target", msg),
                10,
            )
            self.create_subscription(
                JointState,
                f"/{role}/joint_states",
                lambda msg, role=role: self._joint_callback(role, "actual", msg),
                10,
            )
            self.create_subscription(
                String,
                f"/{role}/actuator_state",
                lambda msg, role=role: self._actuator_callback(role, msg),
                10,
            )

        self.create_timer(1.0 / float(config.decision_hz), self._decision_tick)
        self.create_timer(1.0 / float(config.command_hz), self._command_tick)
        self.create_timer(1.0, self._diagnostic_tick)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.top_model.reset_recurrent_state()
        for model in self.bottom_models.values():
            model.reset_recurrent_state()
        self.attacker_policy.reset()
        self.attacker_slew.reset()
        self.defender_memory.reset()
        self.monitor.reset()
        self.commands = {"defender": VelocityCommand(), "attacker": VelocityCommand()}
        self.records.clear()
        self.skill_counts = {"protect": 0, "chase": 0}
        self.terminal = None
        self.terminal_at = None
        self.terminal_control_at = None
        self.last_decision_at = 0.0
        self.episode_started_at = time.monotonic()
        self.episode_started_control_s = self._control_now()
        self.episode_started_sim_s = None
        self._logged_ready = False
        self.get_logger().info("episode state and all recurrent policies reset")

    def _pose_callback(self, role: str, msg: PoseStamped) -> None:
        if msg.header.frame_id not in ("", "map"):
            self.last_status = f"invalid_frame:{role}:{msg.header.frame_id}"
            return
        position = msg.pose.position
        orientation = msg.pose.orientation
        try:
            pose = Pose2D(
                float(position.x),
                float(position.y),
                quaternion_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w),
            )
        except ValueError as exc:
            self.last_status = f"invalid_pose:{role}:{exc}"
            return
        self.poses[role] = pose
        self.input_stamps[f"pose:{role}"] = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def _scan_callback(self, role: str, msg: LaserScan) -> None:
        try:
            radar = normalize_laser_scan(
                msg.ranges,
                angle_min=msg.angle_min,
                angle_increment=msg.angle_increment,
                range_min=msg.range_min,
                range_max=msg.range_max,
                config=self.config,
            )
        except ValueError as exc:
            self.last_status = f"invalid_scan:{role}:{exc}"
            return
        self.radars[role] = radar
        self.input_stamps[f"scan:{role}"] = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def _collision_callback(self, role: str, msg: Bool) -> None:
        self.collisions[role] = bool(msg.data)

    def _joint_callback(self, role: str, kind: str, msg: JointState) -> None:
        values = [float(value) for value in msg.velocity]
        if len(values) != 2 or not all(np.isfinite(value) for value in values):
            self.last_status = f"invalid_joint_state:{role}:{kind}"
            return
        self.wheel_telemetry[role][kind] = {
            "stamp_s": float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9,
            "names": list(msg.name),
            "velocity_radps": values,
        }

    def _actuator_callback(self, role: str, msg: String) -> None:
        try:
            value = json.loads(msg.data)
            if not isinstance(value, dict) or value.get("role") != role:
                raise ValueError("role mismatch")
        except (json.JSONDecodeError, ValueError) as exc:
            self.last_status = f"invalid_actuator_state:{role}:{exc}"
            return
        self.actuator_states[role] = value

    def _profile_callback(self, msg: String) -> None:
        try:
            value = json.loads(msg.data)
            if not isinstance(value, dict) or not value.get("profile_id") or not value.get("checksum"):
                raise ValueError("missing profile identity")
        except (json.JSONDecodeError, ValueError) as exc:
            self.last_status = f"invalid_profile_metadata:{exc}"
            return
        self.profile_metadata = value

    def _required_stamps(self) -> list[float]:
        keys = [
            "pose:defender",
            "pose:attacker",
            "pose:target",
            "scan:defender",
            "scan:attacker",
        ]
        return [self.input_stamps[key] for key in keys if key in self.input_stamps]

    def _inputs_ready(self, now: float) -> bool:
        self.last_input_timing = deployment_input_timing(self._required_stamps(), now=now, config=self.config)
        return len(self.poses) == 3 and len(self.radars) == 2 and self.last_input_timing.fresh

    def _control_now(self) -> float:
        if self.use_sim_time:
            return self.get_clock().now().nanoseconds * 1e-9
        return time.monotonic()

    def _decision_tick(self) -> None:
        control_now = self._control_now()
        simulation_now = self.get_clock().now().nanoseconds * 1e-9
        if self.terminal is not None:
            if self.terminal_control_at is not None and control_now - self.terminal_control_at >= 0.75:
                self.shutdown_requested = True
            return
        if not self._inputs_ready(simulation_now):
            self.commands = {"defender": VelocityCommand(), "attacker": VelocityCommand()}
            self.last_status = f"input_{self.last_input_timing.reason}"
            startup_elapsed = control_now - self.episode_started_control_s
            if startup_elapsed >= self.config.startup_timeout_s:
                self._finish(TerminalResult("input_timeout", "draw", 0))
            return
        if not self._logged_ready:
            self.get_logger().info("all ROS 2 pose and LaserScan inputs are fresh; policy control active")
            self._logged_ready = True
        if self.episode_started_sim_s is None:
            self.episode_started_sim_s = simulation_now

        try:
            actor_obs, privileged_obs, critic_obs = build_policy_observations(
                self.poses,
                self.radars["defender"],
                self.radars["attacker"],
                self.config,
                defender_memory=self.defender_memory,
                obstacles=self.obstacles,
            )
            top_action, _pre_tanh, top_value, top_log_prob = self.top_model.evaluate(
                actor_obs,
                critic_obs,
                greedy=True,
            )
            skill_name = decode_skill_index(top_action)
            bottom_action, _bottom_pre_tanh, bottom_value, bottom_log_prob = self.bottom_models[skill_name].evaluate(
                actor_obs,
                critic_obs,
                greedy=True,
            )
            attacker_action, attacker_info = self.attacker_policy.get_action_with_info(privileged_obs)
            defender_command = normalized_action_to_command(
                bottom_action,
                max_linear_mps=self.config.defender_max_linear_mps,
                max_angular_radps=self.config.defender_max_angular_radps,
            )
            attacker_command = normalized_action_to_command(
                attacker_action,
                max_linear_mps=self.config.attacker_max_linear_mps,
                max_angular_radps=self.config.attacker_max_angular_radps,
            )
            attacker_command = VelocityCommand(
                self.attacker_slew.apply(attacker_command.linear_mps, 1.0 / self.config.decision_hz),
                attacker_command.angular_radps,
            )
        except Exception as exc:  # noqa: BLE001 - unsafe output must fail closed.
            self.commands = {"defender": VelocityCommand(), "attacker": VelocityCommand()}
            self.last_status = f"inference_error:{type(exc).__name__}"
            self.get_logger().error(f"policy inference stopped safely: {exc}")
            self._finish(TerminalResult("inference_error", "draw", self.monitor.step))
            return

        terminal = self.monitor.update(
            self.poses,
            defender_collision=self.collisions["defender"],
            attacker_collision=self.collisions["attacker"],
        )
        self.last_decision_at = control_now
        self.last_skill = skill_name
        self.skill_counts[skill_name] += 1
        self.skill_pub.publish(String(data=skill_name))
        record = {
            "step": int(self.monitor.step),
            "elapsed_s": float(time.monotonic() - self.episode_started_at),
            "simulation_elapsed_s": float(simulation_now - self.episode_started_sim_s),
            "poses": {role: _pose_dict(pose) for role, pose in self.poses.items()},
            "selected_skill": skill_name,
            "top_action": _json_action(top_action),
            "top_value": float(top_value),
            "top_log_prob": float(top_log_prob),
            "defender_action": _json_action(bottom_action),
            "defender_value": float(bottom_value),
            "defender_log_prob": float(bottom_log_prob),
            "attacker_action": _json_action(attacker_action),
            "attacker_info": dict(attacker_info),
            "defender_cmd": asdict(defender_command),
            "attacker_cmd": asdict(attacker_command),
            "defender_collision": bool(self.collisions["defender"]),
            "attacker_collision": bool(self.collisions["attacker"]),
            "input_timing": asdict(self.last_input_timing),
            "wheel_telemetry": copy.deepcopy(self.wheel_telemetry),
            "actuator_state": copy.deepcopy(self.actuator_states),
        }
        self.records.append(record)

        if terminal is not None:
            self.commands = {"defender": VelocityCommand(), "attacker": VelocityCommand()}
            self._finish(terminal)
        else:
            self.commands = {"defender": defender_command, "attacker": attacker_command}
            self.last_status = "active"

    def _command_tick(self) -> None:
        now = self._control_now()
        simulation_now = self.get_clock().now().nanoseconds * 1e-9
        decision_fresh = self.last_decision_at > 0.0 and now - self.last_decision_at <= self.config.input_timeout_s
        if self.terminal is not None or not decision_fresh or not self._inputs_ready(simulation_now):
            commands = {"defender": VelocityCommand(), "attacker": VelocityCommand()}
        else:
            commands = self.commands
        self.defender_pub.publish(_twist(commands["defender"]))
        self.attacker_pub.publish(_twist(commands["attacker"]))

    def _diagnostic_tick(self) -> None:
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus()
        status.name = "trackmaker_policy"
        status.hardware_id = "coppeliasim_or_turtlebot4"
        status.level = DiagnosticStatus.OK if self.last_status in {"active", "waiting_for_inputs"} else DiagnosticStatus.WARN
        status.message = self.last_status
        status.values = [
            KeyValue(key="selected_skill", value=self.last_skill),
            KeyValue(key="controller_env_obstacle_mask", value="off"),
            KeyValue(key="action_shield", value="off"),
            KeyValue(
                key="create3_reflexes",
                value="disabled_simulation" if self.use_sim_time else "enabled_external",
            ),
            KeyValue(key="decision_hz", value=str(self.config.decision_hz)),
            KeyValue(key="command_hz", value=str(self.config.command_hz)),
            KeyValue(key="input_timing_reason", value=self.last_input_timing.reason),
            KeyValue(key="input_skew_s", value=str(self.last_input_timing.skew_s)),
            KeyValue(key="input_max_age_s", value=str(self.last_input_timing.max_age_s)),
            KeyValue(key="max_input_skew_s", value=str(self.config.max_input_skew_s)),
            KeyValue(key="watchdog_clock", value="/clock" if self.use_sim_time else "steady"),
            KeyValue(key="profile_id", value=str(self.profile_metadata.get("profile_id", "unavailable"))),
            KeyValue(key="profile_provenance", value=str(self.profile_metadata.get("provenance", "unavailable"))),
        ]
        msg.status = [status]
        self.diagnostics_pub.publish(msg)

    def _finish(self, terminal: TerminalResult) -> None:
        self.terminal = terminal
        self.terminal_at = time.monotonic()
        self.terminal_control_at = self._control_now()
        self.last_status = terminal.reason
        summary = {
            "reason": terminal.reason,
            "outcome": terminal.outcome,
            "steps": terminal.step,
            "elapsed_s": float(self.terminal_at - self.episode_started_at),
            "simulation_elapsed_s": float(
                self.get_clock().now().nanoseconds * 1e-9 - (self.episode_started_sim_s or 0.0)
            ),
            "seed": self.episode_seed,
            "skill_counts": dict(self.skill_counts),
            "collision_steps": {
                "defender": sum(bool(row["defender_collision"]) for row in self.records),
                "attacker": sum(bool(row["attacker_collision"]) for row in self.records),
            },
            "controller_env_obstacle_mask": False,
            "action_shield": False,
            "create3_reflexes_expected": not self.use_sim_time,
            "config": asdict(self.config),
            "checkpoints": self.checkpoint_metadata,
            "profile_metadata": copy.deepcopy(self.profile_metadata),
            "records": self.records,
        }
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        self.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.outcome_pub.publish(String(data=json.dumps({"reason": terminal.reason, "outcome": terminal.outcome})))
        self.get_logger().info(f"episode finished: {terminal.reason}; wrote {self.output_json}")

    def publish_stop(self) -> None:
        zero = _twist(VelocityCommand())
        self.defender_pub.publish(zero)
        self.attacker_pub.publish(zero)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-checkpoint", type=Path, default=DEFAULT_TOP)
    parser.add_argument("--protect-checkpoint", type=Path, default=DEFAULT_PROTECT)
    parser.add_argument("--chase-checkpoint", type=Path, default=DEFAULT_CHASE)
    parser.add_argument("--attacker-checkpoint", type=Path, default=DEFAULT_ATTACKER)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps", type=int, default=449)
    parser.add_argument("--seed", type=int, default=20260326)
    parser.add_argument("--capture-distance-m", type=float, default=0.45)
    parser.add_argument("--target-distance-m", type=float, default=0.35)
    parser.add_argument("--startup-timeout-s", type=float, default=30.0)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "coppeliasim/scenes/trackmaker_turtlebot4_v2_1_scene.json",
    )
    parser.add_argument("--wall-time", action="store_true", help="use wall time instead of CoppeliaSim /clock")
    return parser.parse_args(argv)


def _load_models(args: argparse.Namespace) -> tuple[Any, dict[str, Any], LearnedAttackerPolicy, dict[str, Any]]:
    specs = {
        "top": CheckpointSpec(args.top_checkpoint, "hrl_top_dual_gru_raw", CHECKSUMS["top"]),
        "protect": CheckpointSpec(args.protect_checkpoint, "mlp_ctde", CHECKSUMS["protect"]),
        "chase": CheckpointSpec(args.chase_checkpoint, "nmn_dual_gru_raw", CHECKSUMS["chase"]),
        "attacker": CheckpointSpec(
            args.attacker_checkpoint,
            "attacker_nmn_mlp_multistyle_ppo",
            CHECKSUMS["attacker"],
        ),
    }
    metadata = {name: validate_checkpoint(spec) for name, spec in specs.items()}
    device = torch.device("cpu")
    top_model = load_hrl_top_model(specs["top"].path.resolve(), "hrl_top_dual_gru_raw", device)
    bottom_models = {
        "protect": load_skill_model(specs["protect"].path, "mlp_ctde", device=device, global_model=False),
        "chase": load_skill_model(specs["chase"].path, "nmn_dual_gru_raw", device=device, global_model=False),
    }
    top_model.network.eval()
    for model in bottom_models.values():
        model.network.eval()
    attacker = LearnedAttackerPolicy(
        str(specs["attacker"].path),
        device="cpu",
        alias="rl_ppo_goal_rush",
        reward_style="goal_rush",
    )
    return top_model, bottom_models, attacker, metadata


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = replace(
        DeploymentConfig(),
        max_steps=max(1, int(args.max_steps)),
        capture_distance_m=max(0.0, float(args.capture_distance_m)),
        target_distance_m=max(0.0, float(args.target_distance_m)),
        startup_timeout_s=max(1.0, float(args.startup_timeout_s)),
    )
    top_model, bottom_models, attacker, metadata = _load_models(args)
    rclpy.init(args=[])
    node = TrackMakerPolicyNode(
        config=config,
        top_model=top_model,
        bottom_models=bottom_models,
        attacker_policy=attacker,
        checkpoint_metadata=metadata,
        output_json=args.output_json,
        episode_seed=args.seed,
        obstacles=list(json.loads(args.manifest.read_text(encoding="utf-8"))["obstacles"]),
        use_sim_time=not args.wall_time,
    )
    try:
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(10):
            node.publish_stop()
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.05)
        node.destroy_node()
        rclpy.shutdown()
    valid_reasons = {
        "defender_caught_attacker",
        "attacker_caught_target",
        "timeout_defender_wins",
        "defender_collision",
    }
    return 0 if node.terminal is not None and node.terminal.reason in valid_reasons else 2


if __name__ == "__main__":
    raise SystemExit(main())
