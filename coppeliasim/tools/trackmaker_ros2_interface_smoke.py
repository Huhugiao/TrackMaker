#!/usr/bin/env python3
"""Validate live TrackMaker ROS 2 topic types, frames, timestamps and rates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage


class InterfaceProbe(Node):
    def __init__(self) -> None:
        super().__init__("trackmaker_interface_probe")
        self.counts: Counter[str] = Counter()
        self.errors: list[str] = []
        self.tf_children: set[str] = set()
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
        self.create_subscription(Clock, "/clock", lambda _msg: self.counts.update(["clock"]), 10)
        self.create_subscription(TFMessage, "/tf", self.tf, 10)

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    duration = max(1.0, float(args.duration_s))
    rclpy.init(args=[])
    node = InterfaceProbe()
    started = time.monotonic()
    while rclpy.ok() and time.monotonic() - started < duration:
        rclpy.spin_once(node, timeout_sec=0.1)
    rates = {name: count / duration for name, count in sorted(node.counts.items())}
    required = {
        "pose:defender": 4.0,
        "pose:attacker": 4.0,
        "pose:target": 4.0,
        "scan:defender": 2.0,
        "scan:attacker": 2.0,
        "cmd:defender": 15.0,
        "cmd:attacker": 15.0,
        "clock": 4.0,
        "tf": 4.0,
    }
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
    report = {
        "duration_s": duration,
        "counts": dict(node.counts),
        "rates_hz": rates,
        "tf_children": sorted(node.tf_children),
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
