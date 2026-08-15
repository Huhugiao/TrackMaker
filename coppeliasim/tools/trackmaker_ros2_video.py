#!/usr/bin/env python3
"""Record the CoppeliaSim overhead ROS image as an annotated 1080p H.264 MP4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


class DemoRecorder(Node):
    def __init__(self, output: Path, fps: float) -> None:
        super().__init__("trackmaker_demo_recorder")
        self.output = output.resolve()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.fps = max(1.0, float(fps))
        self.latest_stamp: float | None = None
        self.episode_started_stamp: float | None = None
        self.skill = "waiting"
        self.outcome = "running"
        self.outcome_at: float | None = None
        self.profile: dict = {}
        self.actuator = {"defender": {}, "attacker": {}}
        self.frames_written = 0
        self.failed = False
        self.camera_logged = False

        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "1920x1080",
            "-r",
            f"{self.fps:g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.output),
        ]
        self.ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE)
        self.create_subscription(Image, "/demo/camera/image_raw", self._image, qos_profile_sensor_data)
        self.create_subscription(String, "/demo/selected_skill", self._skill, 10)
        latched = QoSProfile(depth=1)
        latched.reliability = ReliabilityPolicy.RELIABLE
        latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(String, "/demo/outcome", self._outcome, latched)
        self.create_subscription(String, "/demo/profile_metadata", self._profile, latched)
        for role in ("defender", "attacker"):
            self.create_subscription(
                String,
                f"/{role}/actuator_state",
                lambda message, role=role: self._actuator(role, message),
                10,
            )

    def _profile(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if isinstance(value, dict):
            self.profile = value

    def _actuator(self, role: str, message: String) -> None:
        try:
            value = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if isinstance(value, dict):
            self.actuator[role] = value

    def _image(self, message: Image) -> None:
        channels = 3
        expected = int(message.height) * int(message.step)
        data = np.frombuffer(message.data, dtype=np.uint8)
        if message.encoding not in {"rgb8", "bgr8"} or data.size < expected:
            self.get_logger().error(f"unsupported camera encoding or size: {message.encoding}")
            self.failed = True
            return
        rows = data[:expected].reshape(int(message.height), int(message.step))
        frame = rows[:, : int(message.width) * channels].reshape(int(message.height), int(message.width), channels)
        if message.encoding == "rgb8":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # CoppeliaSim vision-sensor buffers use a bottom-left image origin.
        frame = cv2.rotate(cv2.flip(frame, 0), cv2.ROTATE_180)
        self.latest_stamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
        if not self.camera_logged:
            self.get_logger().info("received camera; recording 1920x1080 H.264")
            self.camera_logged = True
        if self.episode_started_stamp is not None:
            target_frames = int((self.latest_stamp - self.episode_started_stamp) * self.fps) + 1
            while self.frames_written < target_frames:
                timeline_stamp = self.episode_started_stamp + self.frames_written / self.fps
                self._write_frame(frame, timeline_stamp)

    def _skill(self, message: String) -> None:
        self.skill = message.data or "waiting"
        if self.episode_started_stamp is None and self.latest_stamp is not None:
            self.episode_started_stamp = self.latest_stamp

    def _outcome(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            self.outcome = str(payload.get("reason", "terminal"))
        except json.JSONDecodeError:
            self.outcome = message.data or "terminal"
        self.outcome_at = self.latest_stamp

    def _write_frame(self, image: np.ndarray, stamp: float) -> None:
        if self.failed or self.ffmpeg.stdin is None:
            return
        frame = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        elapsed = max(0.0, stamp - (self.episode_started_stamp or stamp))
        overlay = frame.copy()
        cv2.rectangle(overlay, (24, 20), (1125, 350), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0.0, frame)
        cv2.circle(frame, (55, 58), 13, (225, 125, 45), -1)
        cv2.putText(frame, "Defender", (82, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.circle(frame, (280, 58), 13, (50, 65, 225), -1)
        cv2.putText(frame, "Attacker", (307, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.circle(frame, (505, 58), 13, (40, 210, 230), -1)
        cv2.putText(frame, "Target", (532, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"HRL skill: {self.skill}    t={elapsed:05.1f}s",
            (42, 116),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (245, 245, 245),
            2,
        )
        color = (90, 220, 255) if self.outcome != "running" else (215, 215, 215)
        cv2.putText(frame, f"Outcome: {self.outcome}", (42, 163), cv2.FONT_HERSHEY_SIMPLEX, 0.82, color, 2)
        profile_id = str(self.profile.get("profile_id", "profile pending"))
        provenance = str(self.profile.get("provenance", "unknown"))
        calibration = str(self.profile.get("calibration_state", "unknown"))
        cv2.putText(
            frame,
            f"Profile: {profile_id}  [{provenance}/{calibration}]",
            (42, 208),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (225, 225, 225),
            2,
        )
        for row, role in enumerate(("defender", "attacker")):
            state = self.actuator[role]
            requested = (float(state.get("requested_linear_mps", 0.0)), float(state.get("requested_angular_radps", 0.0)))
            filtered = (float(state.get("filtered_linear_mps", 0.0)), float(state.get("filtered_angular_radps", 0.0)))
            wheels = [float(value) for value in state.get("actual_wheel_radps", [0.0, 0.0])]
            runtime_robot = self.profile.get("runtime", {}).get("robots", {}).get(role, {})
            radius = float(runtime_robot.get("wheel_radius_m", 0.0))
            track = max(float(runtime_robot.get("wheel_separation_m", 1.0)), 1e-9)
            actual = (0.5 * radius * sum(wheels), radius * (wheels[1] - wheels[0]) / track)
            contact = state.get("contact", {})
            text = (
                f"{role[0].upper()} req v/w={requested[0]:+.3f}/{requested[1]:+.3f}  "
                f"filt={filtered[0]:+.3f}/{filtered[1]:+.3f}  actual={actual[0]:+.3f}/{actual[1]:+.3f}  "
                f"wheel={wheels[0]:+.2f}/{wheels[1]:+.2f}  support={contact.get('support_ok', False)} "
                f"watchdog={state.get('watchdog_active', True)}"
            )
            cv2.putText(
                frame,
                text,
                (42, 255 + row * 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
        try:
            self.ffmpeg.stdin.write(frame.tobytes())
            self.frames_written += 1
        except (BrokenPipeError, OSError) as exc:
            self.get_logger().error(f"ffmpeg pipe failed: {exc}")
            self.failed = True

    @property
    def complete(self) -> bool:
        return (
            self.outcome_at is not None
            and self.latest_stamp is not None
            and self.latest_stamp - self.outcome_at >= 1.0
        )

    def close(self) -> int:
        if self.ffmpeg.stdin is not None:
            self.ffmpeg.stdin.close()
        return self.ffmpeg.wait(timeout=30)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()
    rclpy.init(args=[])
    node = DemoRecorder(args.output, args.fps)
    try:
        while rclpy.ok() and not node.complete and not node.failed:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    code = node.close()
    if node.frames_written == 0:
        print("no camera frames were recorded")
        return 2
    if code != 0 or node.failed:
        return 3
    print(f"wrote {node.frames_written} frames to {node.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
