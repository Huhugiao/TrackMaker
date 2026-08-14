#!/usr/bin/env python3
"""Compare live CoppeliaSim LaserScan rays with the Gym obstacle geometry."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coppelia_env.ros2_deployment import (  # noqa: E402
    DeploymentConfig,
    Pose2D,
    _ray_obstacle_distance,
    normalize_laser_scan,
    quaternion_to_yaw,
    world_pose_to_training,
)

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from tf2_msgs.msg import TFMessage
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Run in the ros2humble environment") from exc


class Samples(Node):
    def __init__(self) -> None:
        super().__init__("trackmaker_lidar_parity")
        self.poses: dict[str, Pose2D] = {}
        self.laser_positions: dict[str, tuple[float, float]] = {}
        self.scans: dict[str, LaserScan] = {}
        self.create_subscription(TFMessage, "/tf", self.tf, qos_profile_sensor_data)
        for role in ("defender", "attacker"):
            self.create_subscription(
                PoseStamped,
                f"/tracking/{role}/pose",
                lambda msg, role=role: self.pose(role, msg),
                qos_profile_sensor_data,
            )
            self.create_subscription(
                LaserScan,
                f"/{role}/scan",
                lambda msg, role=role: self.scans.__setitem__(role, msg),
                qos_profile_sensor_data,
            )

    def pose(self, role: str, msg: PoseStamped) -> None:
        p, q = msg.pose.position, msg.pose.orientation
        self.poses[role] = Pose2D(p.x, p.y, quaternion_to_yaw(q.x, q.y, q.z, q.w))

    def tf(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            if transform.child_frame_id in ("defender/laser", "attacker/laser"):
                role = transform.child_frame_id.split("/", 1)[0]
                p = transform.transform.translation
                self.laser_positions[role] = (p.x, p.y)


def expected_ranges(
    pose: Pose2D,
    origin_xy: tuple[float, float],
    obstacles: list[dict],
    config: DeploymentConfig,
) -> np.ndarray:
    state = world_pose_to_training(Pose2D(origin_xy[0], origin_xy[1], pose.yaw), config)
    ox = state["x"] + 0.5 * config.pixel_size_px
    oy = state["y"] + 0.5 * config.pixel_size_px
    heading = math.radians(state["theta"])
    ranges = []
    for index in range(config.radar_rays):
        angle = heading + 2.0 * math.pi * index / config.radar_rays
        dx, dy = math.cos(angle), math.sin(angle)
        distance_px = min(
            (_ray_obstacle_distance(ox, oy, dx, dy, obstacle) for obstacle in obstacles),
            default=math.inf,
        )
        ranges.append(min(config.radar_range_m, distance_px * config.scale_m_per_px))
    return np.asarray(ranges, dtype=np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "coppeliasim/scenes/trackmaker_turtlebot4_scene.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = DeploymentConfig()
    node: Samples | None = None
    rclpy.init()
    try:
        node = Samples()
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and (
            len(node.poses) < 2 or len(node.scans) < 2 or len(node.laser_positions) < 2
        ):
            rclpy.spin_once(node, timeout_sec=0.1)
        if len(node.poses) < 2 or len(node.scans) < 2 or len(node.laser_positions) < 2:
            raise RuntimeError("timed out waiting for pose, scan, and laser TF")

        report: dict[str, object] = {"roles": {}}
        passed = True
        for role in ("defender", "attacker"):
            msg = node.scans[role]
            normalized = normalize_laser_scan(
                msg.ranges,
                angle_min=msg.angle_min,
                angle_increment=msg.angle_increment,
                range_min=msg.range_min,
                range_max=msg.range_max,
                config=config,
            )
            observed = (normalized.astype(np.float64) + 1.0) * 0.5 * config.radar_range_m
            expected = expected_ranges(
                node.poses[role], node.laser_positions[role], list(manifest["obstacles"]), config
            )
            error = np.abs(observed - expected)
            worst = np.argsort(error)[-8:][::-1]
            role_passed = bool(np.max(error) <= 0.05)
            passed = passed and role_passed
            report["roles"][role] = {
                "max_abs_error_m": float(np.max(error)),
                "mean_abs_error_m": float(np.mean(error)),
                "p95_abs_error_m": float(np.percentile(error, 95)),
                "rays_over_0_05_m": int(np.count_nonzero(error > 0.05)),
                "laser_offset_from_base_m": float(
                    math.dist(
                        node.laser_positions[role],
                        (node.poses[role].x, node.poses[role].y),
                    )
                ),
                "worst_rays": [
                    {
                        "index": int(index),
                        "observed_m": float(observed[index]),
                        "expected_m": float(expected[index]),
                        "error_m": float(error[index]),
                    }
                    for index in worst
                ],
                "passed": role_passed,
            }
        report["passed"] = passed
        text = json.dumps(report, ensure_ascii=False, indent=2)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0 if passed else 1
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
