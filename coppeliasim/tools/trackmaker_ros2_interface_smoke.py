#!/usr/bin/env python3
"""Validate live TrackMaker ROS 2 topic types, frames, timestamps and rates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import time

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState, LaserScan
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


class InterfaceProbe(Node):
    def __init__(self) -> None:
        super().__init__("trackmaker_interface_probe")
        self.counts: Counter[str] = Counter()
        self.errors: list[str] = []
        self.tf_children: set[str] = set()
        self.profile_metadata: dict = {}
        self.actuator_states: dict[str, dict] = {}
        self.diagnostic_names: set[str] = set()
        self.first_clock_s: float | None = None
        self.last_clock_s = -math.inf
        for role in ("defender", "attacker", "target"):
            self.create_subscription(
                PoseStamped,
                f"/tracking/{role}/pose",
                lambda msg, role=role: self.pose(role, msg),
                10,
            )
        for role in ("defender", "attacker"):
            self.create_subscription(
                LaserScan,
                f"/{role}/scan",
                lambda msg, role=role: self.scan(role, msg),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                Twist,
                f"/{role}/cmd_vel",
                lambda _msg, role=role: self.counts.update([f"cmd:{role}"]),
                10,
            )
            self.create_subscription(
                JointState,
                f"/{role}/joint_targets",
                lambda msg, role=role: self.joint(role, "target", msg),
                10,
            )
            self.create_subscription(
                JointState,
                f"/{role}/joint_states",
                lambda msg, role=role: self.joint(role, "actual", msg),
                10,
            )
            self.create_subscription(
                String,
                f"/{role}/actuator_state",
                lambda msg, role=role: self.actuator(role, msg),
                10,
            )
            self.create_subscription(
                String,
                f"/{role}/actuator_events",
                lambda _msg, role=role: self.counts.update([f"event:{role}"]),
                10,
            )
        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(String, "/demo/profile_metadata", self.profile, latched)
        self.create_subscription(DiagnosticArray, "/diagnostics", self.diagnostics, 10)
        self.create_subscription(Clock, "/clock", self.clock, 10)
        self.create_subscription(TFMessage, "/tf", self.tf, 10)

    def clock(self, message: Clock) -> None:
        self.counts.update(["clock"])
        stamp = float(message.clock.sec) + float(message.clock.nanosec) * 1e-9
        if self.first_clock_s is None:
            self.first_clock_s = stamp
        if stamp + 1e-12 < self.last_clock_s:
            self.errors.append("/clock moved backwards")
        self.last_clock_s = stamp

    def pose(self, role: str, message: PoseStamped) -> None:
        self.counts.update([f"pose:{role}"])
        if message.header.frame_id != "map":
            self.errors.append(f"pose:{role} frame_id={message.header.frame_id!r}")
        if message.header.stamp.sec < 0 or message.header.stamp.nanosec < 0:
            self.errors.append(f"pose:{role} has invalid timestamp")

    def scan(self, role: str, message: LaserScan) -> None:
        self.counts.update([f"scan:{role}"])
        if message.header.frame_id != f"{role}/laser":
            self.errors.append(f"scan:{role} frame_id={message.header.frame_id!r}")
        if len(message.ranges) != 64:
            self.errors.append(f"scan:{role} ranges={len(message.ranges)}")

    def tf(self, message: TFMessage) -> None:
        self.counts.update(["tf"])
        for transform in message.transforms:
            if transform.header.frame_id != "map":
                self.errors.append(f"tf parent={transform.header.frame_id!r}")
            self.tf_children.add(transform.child_frame_id)

    def joint(self, role: str, kind: str, message: JointState) -> None:
        self.counts.update([f"joint_{kind}:{role}"])
        if message.header.frame_id != f"{role}/base_link":
            self.errors.append(f"joint_{kind}:{role} frame_id={message.header.frame_id!r}")
        if list(message.name) != ["left_wheel_joint", "right_wheel_joint"]:
            self.errors.append(f"joint_{kind}:{role} names={list(message.name)!r}")
        if len(message.velocity) != 2 or not all(math.isfinite(float(value)) for value in message.velocity):
            self.errors.append(f"joint_{kind}:{role} invalid velocity")

    def actuator(self, role: str, message: String) -> None:
        self.counts.update([f"actuator:{role}"])
        try:
            value = json.loads(message.data)
            required = {
                "time_s",
                "role",
                "requested_linear_mps",
                "requested_angular_radps",
                "filtered_linear_mps",
                "filtered_angular_radps",
                "target_wheel_radps",
                "actual_wheel_radps",
                "watchdog_active",
                "contact",
                "last_actuator_delay_s",
                "deadline_miss_count",
            }
            if value.get("role") != role or not required.issubset(value):
                raise ValueError("missing fields or role mismatch")
            self.actuator_states[role] = value
        except (json.JSONDecodeError, ValueError) as exc:
            self.errors.append(f"actuator:{role} invalid JSON: {exc}")

    def profile(self, message: String) -> None:
        self.counts.update(["profile_metadata"])
        try:
            value = json.loads(message.data)
            if value.get("provenance") not in {"prior", "measured"}:
                raise ValueError("invalid provenance")
            if not str(value.get("checksum", "")).startswith("sha256:") or "runtime" not in value:
                raise ValueError("missing checksum or runtime introspection")
            self.profile_metadata = value
        except (json.JSONDecodeError, ValueError) as exc:
            self.errors.append(f"profile metadata invalid: {exc}")

    def diagnostics(self, message: DiagnosticArray) -> None:
        self.counts.update(["diagnostics"])
        self.diagnostic_names.update(status.name for status in message.status)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-commands", action="store_true")
    args = parser.parse_args()
    duration = max(1.0, float(args.duration_s))
    rclpy.init(args=[])
    node = InterfaceProbe()
    started = time.monotonic()
    while rclpy.ok() and time.monotonic() - started < duration:
        rclpy.spin_once(node, timeout_sec=0.1)
    wall_duration = time.monotonic() - started
    simulation_duration = (
        node.last_clock_s - node.first_clock_s
        if node.first_clock_s is not None and math.isfinite(node.last_clock_s)
        else 0.0
    )
    if simulation_duration <= 0.0:
        node.errors.append("/clock did not advance")
    rate_duration = simulation_duration if simulation_duration > 0.0 else wall_duration
    rates = {name: count / rate_duration for name, count in sorted(node.counts.items())}
    wall_rates = {name: count / wall_duration for name, count in sorted(node.counts.items())}
    required = {
        "pose:defender": 4.0,
        "pose:attacker": 4.0,
        "pose:target": 4.0,
        "scan:defender": 2.0,
        "scan:attacker": 2.0,
        "clock": 4.0,
        "tf": 4.0,
        "joint_target:defender": 15.0,
        "joint_target:attacker": 15.0,
        "joint_actual:defender": 15.0,
        "joint_actual:attacker": 15.0,
        "actuator:defender": 15.0,
        "actuator:attacker": 15.0,
        "diagnostics": 4.0,
        "profile_metadata": 0.1,
    }
    if args.require_commands:
        required.update({"cmd:defender": 15.0, "cmd:attacker": 15.0})
    for name, minimum_hz in required.items():
        if rates.get(name, 0.0) < minimum_hz:
            node.errors.append(f"{name} rate={rates.get(name, 0.0):.2f}Hz < {minimum_hz:.2f}Hz")
    required_tf_children = {
        "defender/base_link",
        "defender/laser",
        "attacker/base_link",
        "attacker/laser",
        "target/base_link",
    }
    missing_tf = required_tf_children - node.tf_children
    if missing_tf:
        node.errors.append(f"missing TF children: {sorted(missing_tf)}")
    required_diagnostics = {"trackmaker_bridge/defender", "trackmaker_bridge/attacker"}
    missing_diagnostics = required_diagnostics - node.diagnostic_names
    if missing_diagnostics:
        node.errors.append(f"missing diagnostics: {sorted(missing_diagnostics)}")
    report = {
        "requested_wall_duration_s": duration,
        "observed_wall_duration_s": wall_duration,
        "simulation_duration_s": simulation_duration,
        "rates_time_base": "simulation_clock",
        "counts": dict(node.counts),
        "rates_hz": rates,
        "wall_rates_hz": wall_rates,
        "tf_children": sorted(node.tf_children),
        "diagnostic_names": sorted(node.diagnostic_names),
        "profile_metadata": node.profile_metadata,
        "actuator_states": node.actuator_states,
        "errors": node.errors,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
