"""TurtleBot4 geometry and control helpers for CoppeliaSim integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class TurtleBot4Spec:
    base_link: str
    left_wheel_joint: str
    right_wheel_joint: str
    lidar_link: str
    rgb_camera_frame: str
    wheel_radius_m: float
    wheel_separation_m: float

    def to_manifest(self) -> dict[str, float | str]:
        return asdict(self)


def _require_named(root: ET.Element, tag: str, name: str) -> ET.Element:
    for node in root.findall(tag):
        if node.attrib.get("name") == name:
            return node
    raise ValueError(f"TurtleBot4 URDF missing {tag} named {name!r}")


def _joint_xyz(root: ET.Element, joint_name: str) -> tuple[float, float, float]:
    joint = _require_named(root, "joint", joint_name)
    origin = joint.find("origin")
    raw = origin.attrib.get("xyz", "") if origin is not None else ""
    parts = [float(v) for v in raw.split()]
    if len(parts) != 3:
        raise ValueError(f"joint {joint_name!r} origin xyz must contain 3 floats")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _wheel_radius(root: ET.Element, wheel_link: str) -> float:
    link = _require_named(root, "link", wheel_link)
    for path in (
        "collision/geometry/cylinder",
        "visual/geometry/cylinder",
    ):
        node = link.find(path)
        if node is not None and node.attrib.get("radius") is not None:
            radius = float(node.attrib["radius"])
            if radius > 0.0:
                return radius
    raise ValueError(f"wheel link {wheel_link!r} missing positive cylinder radius")


def load_turtlebot4_spec(urdf_path: str | Path) -> TurtleBot4Spec:
    path = Path(urdf_path)
    root = ET.parse(path).getroot()

    base_link = "base_link"
    left_wheel_joint = "left_wheel_joint"
    right_wheel_joint = "right_wheel_joint"
    lidar_link = "rplidar_link"
    rgb_camera_frame = "oakd_rgb_camera_frame"
    for link_name in (base_link, lidar_link, rgb_camera_frame):
        _require_named(root, "link", link_name)
    for joint_name in (left_wheel_joint, right_wheel_joint):
        _require_named(root, "joint", joint_name)

    _left_x, left_y, _left_z = _joint_xyz(root, "wheel_drop_left_joint")
    _right_x, right_y, _right_z = _joint_xyz(root, "wheel_drop_right_joint")
    wheel_separation = abs(float(left_y) - float(right_y))
    if wheel_separation <= 0.0:
        raise ValueError("TurtleBot4 wheel separation must be positive")

    return TurtleBot4Spec(
        base_link=base_link,
        left_wheel_joint=left_wheel_joint,
        right_wheel_joint=right_wheel_joint,
        lidar_link=lidar_link,
        rgb_camera_frame=rgb_camera_frame,
        wheel_radius_m=_wheel_radius(root, "left_wheel"),
        wheel_separation_m=wheel_separation,
    )


def _clip(value: float, limit: float | None) -> float:
    out = float(value)
    if limit is None:
        return out
    limit = abs(float(limit))
    return max(-limit, min(limit, out))


def cmd_vel_to_wheel_speeds(
    linear_mps: float,
    angular_radps: float,
    spec: TurtleBot4Spec,
    max_linear_mps: float | None = None,
    max_angular_radps: float | None = None,
) -> tuple[float, float]:
    """Convert body-frame cmd_vel to left/right wheel angular speeds in rad/s."""

    v = _clip(float(linear_mps), max_linear_mps)
    w = _clip(float(angular_radps), max_angular_radps)
    radius = max(float(spec.wheel_radius_m), 1e-9)
    half_sep = 0.5 * float(spec.wheel_separation_m)
    left = (v - half_sep * w) / radius
    right = (v + half_sep * w) / radius
    if not (math.isfinite(left) and math.isfinite(right)):
        raise ValueError("non-finite wheel speed generated")
    return float(left), float(right)
