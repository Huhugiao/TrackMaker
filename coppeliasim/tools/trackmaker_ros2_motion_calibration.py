#!/usr/bin/env python3
"""Measure differential-drive response against simulation time, not wall time."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from std_msgs.msg import Bool
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Run this tool in the ros2humble environment") from exc


ROLES = ("defender", "attacker")


def _yaw(msg: PoseStamped) -> float:
    q = msg.pose.orientation
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _angle_delta(first: float, second: float) -> float:
    return (second - first + math.pi) % (2.0 * math.pi) - math.pi


class MotionCalibration(Node):
    def __init__(self) -> None:
        super().__init__(
            "trackmaker_motion_calibration",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
            automatically_declare_parameters_from_overrides=True,
        )
        self.poses: dict[str, tuple[float, float, float, float]] = {}
        self.collisions = {role: False for role in ROLES}
        self.command_pubs = {role: self.create_publisher(Twist, f"/{role}/cmd_vel", 10) for role in ROLES}
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

    def _pose(self, role: str, msg: PoseStamped) -> None:
        self.poses[role] = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, _yaw(msg))

    def simulation_time(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def publish(self, active_role: str | None = None, linear: float = 0.0, angular: float = 0.0) -> None:
        for role, publisher in self.command_pubs.items():
            msg = Twist()
            if role == active_role:
                msg.linear.x = float(linear)
                msg.angular.z = float(angular)
            publisher.publish(msg)

    def phase(
        self,
        role: str,
        name: str,
        duration_s: float,
        linear: float,
        angular: float,
        *,
        publish_commands: bool = True,
    ) -> dict:
        start_sim = self.simulation_time()
        start_wall = time.monotonic()
        samples = []
        while rclpy.ok() and self.simulation_time() - start_sim < duration_s:
            if publish_commands:
                self.publish(role, linear, angular)
            rclpy.spin_once(self, timeout_sec=0.01)
            if role in self.poses:
                samples.append((self.simulation_time(), *self.poses[role], self.collisions[role]))
        self.publish()
        if not samples:
            raise RuntimeError(f"no pose samples during {role}:{name}")
        return {
            "name": name,
            "command": {"linear_mps": linear, "angular_radps": angular},
            "requested_duration_s": duration_s,
            "simulation_duration_s": samples[-1][0] - samples[0][0],
            "wall_duration_s": time.monotonic() - start_wall,
            "samples": samples,
        }


def _summarize(phases: list[dict]) -> dict:
    by_name = {phase["name"]: phase for phase in phases}

    def endpoints(name: str):
        samples = by_name[name]["samples"]
        return samples[0], samples[-1]

    straight_start, straight_end = endpoints("straight")
    sx, sy, sz, syaw = straight_start[1:5]
    ex, ey, ez, eyaw = straight_end[1:5]
    dx, dy = ex - sx, ey - sy
    forward = dx * math.cos(syaw) + dy * math.sin(syaw)
    lateral = -dx * math.sin(syaw) + dy * math.cos(syaw)
    straight_expected = by_name["straight"]["command"]["linear_mps"] * by_name["straight"]["simulation_duration_s"]

    turn_results = {}
    for name in ("left", "right"):
        samples = by_name[name]["samples"]
        unwrapped = 0.0
        for first, second in zip(samples, samples[1:]):
            unwrapped += _angle_delta(first[4], second[4])
        expected = by_name[name]["command"]["angular_radps"] * by_name[name]["simulation_duration_s"]
        translation = math.hypot(samples[-1][1] - samples[0][1], samples[-1][2] - samples[0][2])
        turn_results[name] = {
            "yaw_rad": unwrapped,
            "expected_yaw_rad": expected,
            "response_ratio": unwrapped / expected,
            "translation_m": translation,
        }

    all_samples = [sample for phase in phases for sample in phase["samples"]]
    watchdog = by_name["watchdog_silence"]["samples"]
    watchdog_start = watchdog[0]
    watchdog_after_timeout = min(watchdog, key=lambda sample: abs(sample[0] - watchdog_start[0] - 0.6))
    watchdog_end = watchdog[-1]
    watchdog_travel = math.hypot(watchdog_end[1] - watchdog_start[1], watchdog_end[2] - watchdog_start[2])
    watchdog_post_timeout_drift = math.hypot(
        watchdog_end[1] - watchdog_after_timeout[1],
        watchdog_end[2] - watchdog_after_timeout[2],
    )
    stop_drifts = []
    for phase in phases:
        if phase["name"].startswith("stop"):
            first, last = phase["samples"][0], phase["samples"][-1]
            stop_drifts.append(math.hypot(last[1] - first[1], last[2] - first[2]))
    summary = {
        "straight": {
            "forward_m": forward,
            "lateral_m": lateral,
            "expected_forward_m": straight_expected,
            "response_ratio": forward / straight_expected,
            "yaw_drift_rad": _angle_delta(syaw, eyaw),
        },
        "turns": turn_results,
        "z_min_m": min(sample[3] for sample in all_samples),
        "z_max_m": max(sample[3] for sample in all_samples),
        "max_stop_drift_m": max(stop_drifts, default=0.0),
        "watchdog_silence_travel_m": watchdog_travel,
        "watchdog_post_timeout_drift_m": watchdog_post_timeout_drift,
        "collision_seen": any(bool(sample[5]) for sample in all_samples),
    }
    summary["passed"] = bool(
        0.85 <= summary["straight"]["response_ratio"] <= 1.15
        and abs(summary["straight"]["lateral_m"]) <= 0.03
        and abs(summary["straight"]["yaw_drift_rad"]) <= 0.08
        and all(0.85 <= value["response_ratio"] <= 1.15 for value in turn_results.values())
        and all(value["translation_m"] <= 0.03 for value in turn_results.values())
        and summary["z_min_m"] >= 0.03
        and summary["z_max_m"] - summary["z_min_m"] <= 0.01
        and summary["max_stop_drift_m"] <= 0.02
        and summary["watchdog_silence_travel_m"] <= 0.07
        and summary["watchdog_post_timeout_drift_m"] <= 0.015
        and not summary["collision_seen"]
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rclpy.init(args=[])
    node = MotionCalibration()
    phases_by_role: dict[str, list[dict]] = {}
    wall_start = time.monotonic()
    try:
        deadline = wall_start + 30.0
        while rclpy.ok() and (len(node.poses) < 2 or node.simulation_time() <= 0.0):
            node.publish()
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() >= deadline:
                raise RuntimeError("timed out waiting for /clock and robot poses")
        for role in ROLES:
            phases_by_role[role] = [
                node.phase(role, "stop_before", 0.5, 0.0, 0.0),
                node.phase(role, "watchdog_prime", 0.5, 0.1, 0.0),
                node.phase(role, "watchdog_silence", 1.0, 0.0, 0.0, publish_commands=False),
                node.phase(role, "straight", 3.0, 0.1, 0.0),
                node.phase(role, "stop_after_straight", 0.5, 0.0, 0.0),
                node.phase(role, "left", 2.0, 0.0, 0.5),
                node.phase(role, "stop_after_left", 0.5, 0.0, 0.0),
                node.phase(role, "right", 2.0, 0.0, -0.5),
                node.phase(role, "stop_final", 0.75, 0.0, 0.0),
            ]
        summaries = {role: _summarize(phases) for role, phases in phases_by_role.items()}
        sim_duration = sum(phase["simulation_duration_s"] for phases in phases_by_role.values() for phase in phases)
        wall_duration = time.monotonic() - wall_start
        result = {
            "passed": all(summary["passed"] for summary in summaries.values()),
            "simulation_duration_s": sim_duration,
            "wall_duration_s": wall_duration,
            "mean_realtime_factor": sim_duration / wall_duration,
            "robots": summaries,
            "phases": phases_by_role,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "phases"}, indent=2))
        return 0 if result["passed"] else 1
    finally:
        for _ in range(10):
            node.publish()
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.02)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
