#!/usr/bin/env python3
"""Export a portable multi-excitation V2.1 calibration dataset over ROS 2."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coppelia_env.digital_twin import load_profile  # noqa: E402
from coppelia_env.digital_twin_calibration import (  # noqa: E402
    CALIBRATION_DATA_VERSION,
    CalibrationPhase,
    calibration_plan,
    dataset_checksum,
    validate_calibration_dataset,
)

try:
    import rclpy
    from diagnostic_msgs.msg import DiagnosticArray
    from geometry_msgs.msg import PoseStamped, Twist
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool, String
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Run this exporter in the ros2humble environment") from exc


ROLES = ("defender", "attacker")


def _yaw(message: PoseStamped) -> float:
    q = message.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _twist(linear: float = 0.0, angular: float = 0.0) -> Twist:
    message = Twist()
    message.linear.x = float(linear)
    message.angular.z = float(angular)
    return message


def _uint8(value: Any) -> int:
    if isinstance(value, (bytes, bytearray)):
        return int(value[0]) if value else 0
    return int(value)


class CalibrationExporter(Node):
    def __init__(self, profile: dict[str, Any]) -> None:
        super().__init__(
            "trackmaker_v2_calibration_exporter",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
            automatically_declare_parameters_from_overrides=True,
        )
        self.profile = profile
        self.command_pubs = {role: self.create_publisher(Twist, f"/{role}/cmd_vel", 10) for role in ROLES}
        self.poses: dict[str, dict[str, float]] = {}
        self.collisions = {role: False for role in ROLES}
        self.joints: dict[str, dict[str, dict[str, Any]]] = {role: {} for role in ROLES}
        self.actuator: dict[str, dict[str, Any]] = {}
        self.events: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
        self.profile_metadata: dict[str, Any] = {}
        self.diagnostics: list[dict[str, Any]] = []
        self.current_phase = "startup"
        self.last_sample_time = {role: -math.inf for role in ROLES}
        self.samples: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
        self.phase_records: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLES}
        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(String, "/demo/profile_metadata", self._profile, latched)
        self.create_subscription(DiagnosticArray, "/diagnostics", self._diagnostic, 10)
        for role in ROLES:
            self.create_subscription(
                PoseStamped,
                f"/tracking/{role}/pose",
                lambda msg, role=role: self._pose(role, msg),
                10,
            )
            self.create_subscription(
                Bool,
                f"/{role}/collision",
                lambda msg, role=role: self.collisions.__setitem__(role, bool(msg.data)),
                10,
            )
            self.create_subscription(
                JointState,
                f"/{role}/joint_targets",
                lambda msg, role=role: self._joint(role, "target", msg),
                10,
            )
            self.create_subscription(
                JointState,
                f"/{role}/joint_states",
                lambda msg, role=role: self._joint(role, "actual", msg),
                10,
            )
            self.create_subscription(
                String,
                f"/{role}/actuator_state",
                lambda msg, role=role: self._actuator(role, msg),
                10,
            )
            self.create_subscription(
                String,
                f"/{role}/actuator_events",
                lambda msg, role=role: self._event(role, msg),
                50,
            )

    def simulation_time(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _pose(self, role: str, message: PoseStamped) -> None:
        self.poses[role] = {
            "x": float(message.pose.position.x),
            "y": float(message.pose.position.y),
            "z": float(message.pose.position.z),
            "yaw": _yaw(message),
            "stamp_s": float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9,
        }

    def _joint(self, role: str, kind: str, message: JointState) -> None:
        if len(message.velocity) != 2:
            return
        self.joints[role][kind] = {
            "stamp_s": float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9,
            "velocity_radps": [float(value) for value in message.velocity],
        }

    def _actuator(self, role: str, message: String) -> None:
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if isinstance(value, dict) and value.get("role") == role:
            self.actuator[role] = value

    def _event(self, role: str, message: String) -> None:
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if isinstance(value, dict):
            value["calibration_phase"] = self.current_phase
            self.events[role].append(value)

    def _profile(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if isinstance(value, dict):
            self.profile_metadata = value

    def _diagnostic(self, message: DiagnosticArray) -> None:
        snapshot = {
            "time_s": self.simulation_time(),
            "status": [
                {
                    "name": status.name,
                    "level": _uint8(status.level),
                    "message": status.message,
                    "values": {item.key: item.value for item in status.values},
                }
                for status in message.status
                if status.name.startswith("trackmaker_bridge/")
            ],
        }
        if snapshot["status"]:
            self.diagnostics.append(snapshot)

    def ready(self) -> bool:
        return bool(
            self.simulation_time() > 0.0
            and self.profile_metadata
            and len(self.poses) == 2
            and all(set(self.joints[role]) == {"target", "actual"} for role in ROLES)
            and set(self.actuator) == set(ROLES)
        )

    def publish(self, active_role: str | None = None, linear: float = 0.0, angular: float = 0.0, *, silence_active: bool = False) -> None:
        for role, publisher in self.command_pubs.items():
            if role == active_role and silence_active:
                continue
            publisher.publish(_twist(linear, angular) if role == active_role else _twist())

    def sample(self, role: str, phase: CalibrationPhase, phase_elapsed_s: float, command: tuple[float, float]) -> None:
        if role not in self.actuator or role not in self.poses or set(self.joints[role]) != {"target", "actual"}:
            return
        time_s = float(self.actuator[role].get("time_s", self.simulation_time()))
        if time_s <= self.last_sample_time[role] + 1e-12:
            return
        self.last_sample_time[role] = time_s
        pose = dict(self.poses[role])
        pose.pop("stamp_s", None)
        self.samples[role].append(
            {
                "time_s": time_s,
                "phase": phase.name,
                "phase_category": phase.category,
                "phase_elapsed_s": float(phase_elapsed_s),
                "command_published": bool(phase.publish_commands),
                "command": {"linear_mps": float(command[0]), "angular_radps": float(command[1])},
                "pose": pose,
                "joint_target_radps": list(self.joints[role]["target"]["velocity_radps"]),
                "joint_actual_radps": list(self.joints[role]["actual"]["velocity_radps"]),
                "actuator": dict(self.actuator[role]),
                "collision": bool(self.collisions[role]),
            }
        )

    def run_phase(self, role: str, phase: CalibrationPhase) -> None:
        self.current_phase = f"{role}:{phase.name}"
        start = self.simulation_time()
        next_publish = start
        first_index = len(self.samples[role])
        while rclpy.ok() and self.simulation_time() - start < phase.duration_s:
            elapsed = self.simulation_time() - start
            command = phase.command(elapsed)
            if self.simulation_time() + 1e-12 >= next_publish:
                self.publish(role, *command, silence_active=not phase.publish_commands)
                next_publish = self.simulation_time() + 0.05
            rclpy.spin_once(self, timeout_sec=0.02)
            self.sample(role, phase, elapsed, command)
        self.publish()
        self.phase_records[role].append(
            {
                **asdict(phase),
                "simulation_start_s": start,
                "simulation_duration_s": self.simulation_time() - start,
                "samples": len(self.samples[role]) - first_index,
            }
        )

    def graceful_stop(self, duration_s: float = 0.75) -> dict[str, Any]:
        self.current_phase = "graceful_shutdown"
        start = self.simulation_time()
        next_publish = start
        wall_deadline = time.monotonic() + 3.0
        while rclpy.ok() and self.simulation_time() - start < duration_s and time.monotonic() < wall_deadline:
            if self.simulation_time() + 1e-12 >= next_publish:
                self.publish()
                next_publish = self.simulation_time() + 0.05
            rclpy.spin_once(self, timeout_sec=0.02)
        actual = {
            role: list(self.joints.get(role, {}).get("actual", {}).get("velocity_radps", [math.inf, math.inf]))
            for role in ROLES
        }
        stopped = all(abs(value) <= 0.10 for values in actual.values() for value in values)
        return {
            "zero_command_sent": True,
            "requested_duration_s": float(duration_s),
            "simulation_duration_s": self.simulation_time() - start,
            "actual_wheel_radps": actual,
            "actual_wheel_stopped": stopped,
        }


def _quality(dataset: dict[str, Any]) -> dict[str, Any]:
    roles = {}
    for role, role_data in dataset["roles"].items():
        samples = role_data["samples"]
        contacts = [sample.get("actuator", {}).get("contact", {}) for sample in samples]
        supported = [bool(contact.get("support_ok")) for contact in contacts]
        chassis_floor = [bool(contact.get("chassis_floor")) for contact in contacts]
        outage_watchdog = [
            bool(sample.get("actuator", {}).get("watchdog_active"))
            for sample in samples
            if sample["phase"] == "watchdog_outage"
        ]
        phases_complete = all(int(phase["samples"]) > 0 for phase in role_data["phases"])
        roles[role] = {
            "samples": len(samples),
            "events": len(role_data["events"]),
            "support_fraction": sum(supported) / max(len(supported), 1),
            "chassis_floor_contact_seen": any(chassis_floor),
            "collision_seen": any(bool(sample["collision"]) for sample in samples),
            "watchdog_outage_observed": any(outage_watchdog),
            "all_phases_sampled": phases_complete,
        }
        roles[role]["passed"] = bool(
            roles[role]["samples"] >= 200
            and roles[role]["events"] > 0
            and roles[role]["support_fraction"] >= 0.95
            and not roles[role]["chassis_floor_contact_seen"]
            and not roles[role]["collision_seen"]
            and roles[role]["watchdog_outage_observed"]
            and phases_complete
        )
    return {"roles": roles, "passed": all(value["passed"] for value in roles.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        type=Path,
        default=PROJECT_ROOT / "coppeliasim/profiles/trackmaker_turtlebot4_v2_1_prior.json",
    )
    parser.add_argument("--startup-timeout-s", type=float, default=30.0)
    args = parser.parse_args()
    profile = load_profile(args.profile)
    rclpy.init(args=[])
    node = CalibrationExporter(profile)
    wall_start = time.monotonic()
    graceful: dict[str, Any] = {"zero_command_sent": False, "actual_wheel_stopped": False}
    try:
        deadline = wall_start + max(1.0, float(args.startup_timeout_s))
        next_startup_publish = wall_start
        while rclpy.ok() and not node.ready():
            if time.monotonic() >= next_startup_publish:
                node.publish()
                next_startup_publish = time.monotonic() + 0.05
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() >= deadline:
                raise RuntimeError("timed out waiting for profile, pose, wheel, actuator, and /clock topics")
        if node.profile_metadata.get("checksum") != profile["checksum"]:
            raise RuntimeError("live profile checksum does not match exporter input")
        for role in ROLES:
            for phase in calibration_plan(profile["robots"][role]):
                node.run_phase(role, phase)
        graceful = node.graceful_stop()
        source_profile = {
            key: profile[key]
            for key in ("schema_version", "profile_id", "provenance", "calibration_state", "seed", "checksum")
        }
        dataset = {
            "schema_version": CALIBRATION_DATA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_profile": source_profile,
            "live_profile_metadata": node.profile_metadata,
            "seed": int(profile["seed"]),
            "sampling": {
                "source": "ros2",
                "clock": "/clock",
                "nominal_hz": 20.0,
                "portable_json": True,
                "rosbag_recorded_by_launcher": True,
            },
            "roles": {
                role: {
                    "phases": node.phase_records[role],
                    "samples": node.samples[role],
                    "events": node.events[role],
                }
                for role in ROLES
            },
            "diagnostics": node.diagnostics,
            "graceful_shutdown": graceful,
            "wall_duration_s": time.monotonic() - wall_start,
        }
        dataset["quality"] = _quality(dataset)
        dataset["passed"] = bool(dataset["quality"]["passed"] and graceful["actual_wheel_stopped"])
        dataset["dataset_checksum"] = dataset_checksum(dataset)
        validate_calibration_dataset(dataset)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
        summary = {
            "passed": dataset["passed"],
            "dataset_checksum": dataset["dataset_checksum"],
            "source_profile": source_profile,
            "quality": dataset["quality"],
            "graceful_shutdown": graceful,
            "wall_duration_s": dataset["wall_duration_s"],
            "output": str(args.output.resolve()),
        }
        print(json.dumps(summary, indent=2))
        return 0 if dataset["passed"] else 1
    finally:
        for _ in range(10):
            node.publish()
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.05)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
