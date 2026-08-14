#!/usr/bin/env python3
"""Verify that a recorded rosbag contains the episode trajectory and interfaces."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

try:
    from geometry_msgs.msg import PoseStamped
    from rclpy.serialization import deserialize_message
    import rosbag2_py
    from std_msgs.msg import Bool
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Run in the ros2humble environment") from exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(args.bag), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    counts: Counter[str] = Counter()
    collision_true_counts: Counter[str] = Counter()
    bag_poses: dict[str, list[tuple[float, float, float, float]]] = {"defender": [], "attacker": []}
    pose_topics = {f"/tracking/{role}/pose": role for role in bag_poses}
    collision_topics = {f"/{role}/collision": role for role in bag_poses}
    while reader.has_next():
        topic, data, _timestamp = reader.read_next()
        counts[topic] += 1
        role = pose_topics.get(topic)
        if role is not None:
            msg = deserialize_message(data, PoseStamped)
            stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            bag_poses[role].append(
                (stamp, float(msg.pose.position.x), float(msg.pose.position.y), float(msg.pose.position.z))
            )
        collision_role = collision_topics.get(topic)
        if collision_role is not None and deserialize_message(data, Bool).data:
            collision_true_counts[collision_role] += 1

    episode = json.loads(args.episode.read_text(encoding="utf-8"))
    required = {
        "/clock": "rosgraph_msgs/msg/Clock",
        "/tf": "tf2_msgs/msg/TFMessage",
        "/tracking/defender/pose": "geometry_msgs/msg/PoseStamped",
        "/tracking/attacker/pose": "geometry_msgs/msg/PoseStamped",
        "/tracking/target/pose": "geometry_msgs/msg/PoseStamped",
        "/defender/scan": "sensor_msgs/msg/LaserScan",
        "/attacker/scan": "sensor_msgs/msg/LaserScan",
        "/defender/cmd_vel": "geometry_msgs/msg/Twist",
        "/attacker/cmd_vel": "geometry_msgs/msg/Twist",
    }
    errors = [
        f"{topic}: expected {message_type}, got {topic_types.get(topic)}"
        for topic, message_type in required.items()
        if topic_types.get(topic) != message_type
    ]
    trajectory: dict[str, dict[str, float | int]] = {}
    for role, samples in bag_poses.items():
        if not samples:
            errors.append(f"no bag poses for {role}")
            continue
        record_points = [
            (float(record["poses"][role]["x"]), float(record["poses"][role]["y"]))
            for record in episode["records"]
        ]
        nearest_errors = [min(math.dist(point, sample[1:3]) for sample in samples) for point in record_points]
        max_error = max(nearest_errors, default=math.inf)
        if max_error > 1e-6:
            errors.append(f"{role} trajectory max nearest error {max_error:.9g}m")
        movements = [
            (math.dist(first[1:3], second[1:3]), second[0] - first[0])
            for first, second in zip(samples, samples[1:])
            if second[0] > first[0]
        ]
        max_speed = max((distance / dt for distance, dt in movements), default=0.0)
        speed_limit = {"defender": 0.234, "attacker": 0.180}[role] * 1.15
        z_variation = max(sample[3] for sample in samples) - min(sample[3] for sample in samples)
        if max_speed > speed_limit:
            errors.append(f"{role} apparent speed {max_speed:.6g}m/s > {speed_limit:.6g}m/s")
        if z_variation > 0.002:
            errors.append(f"{role} z variation {z_variation:.6g}m > 0.002m")
        trajectory[role] = {
            "bag_pose_count": len(samples),
            "episode_pose_count": len(record_points),
            "max_nearest_error_m": max_error,
            "max_20hz_step_m": max((distance for distance, _dt in movements), default=0.0),
            "max_20hz_speed_mps": max_speed,
            "speed_limit_with_15pct_physics_tolerance_mps": speed_limit,
            "z_variation_m": z_variation,
        }

    paired_samples = zip(bag_poses["defender"], bag_poses["attacker"])
    min_robot_distance = min(
        (math.dist(defender[1:3], attacker[1:3]) for defender, attacker in paired_samples),
        default=math.inf,
    )
    # The imported Create 3 base collision cylinder has radius 0.164 m.
    physical_contact_distance = 0.328
    if min_robot_distance < physical_contact_distance:
        errors.append(
            f"robot center distance {min_robot_distance:.6g}m < physical contact threshold "
            f"{physical_contact_distance:.6g}m"
        )
    if episode.get("reason") == "defender_caught_attacker" and any(collision_true_counts.values()):
        errors.append(f"capture episode contains obstacle collision messages: {dict(collision_true_counts)}")

    report = {
        "required_topic_types": required,
        "message_counts": dict(sorted(counts.items())),
        "collision_true_message_counts": {
            role: int(collision_true_counts[role]) for role in ("defender", "attacker")
        },
        "min_robot_center_distance_m": min_robot_distance,
        "physical_contact_distance_m": physical_contact_distance,
        "trajectory": trajectory,
        "errors": errors,
        "passed": not errors,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
