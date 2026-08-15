"""Calibratable TurtleBot4 digital-twin profile and actuator primitives.

The module has no ROS or CoppeliaSim dependency.  It is the executable
contract shared by scene generation, offline calibration, and unit tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

import numpy as np


PROFILE_SCHEMA_VERSION = "2.1"
PROFILE_PROVENANCE = frozenset({"prior", "measured"})
ROLES = ("defender", "attacker")
WHEEL_LINKS = frozenset({"left_wheel", "right_wheel"})
CASTER_LINK = "front_caster_link"
CHASSIS_EXCLUDED_LINKS = WHEEL_LINKS | {CASTER_LINK}


@dataclass(frozen=True)
class RigidBodyAggregate:
    mass_kg: float
    center_of_mass_m: tuple[float, float, float]
    inertia_kg_m2: tuple[float, ...]
    links: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mass_kg": self.mass_kg,
            "center_of_mass_m": list(self.center_of_mass_m),
            "inertia_kg_m2": list(self.inertia_kg_m2),
            "aggregated_links": list(self.links),
        }


@dataclass(frozen=True)
class InputTiming:
    fresh: bool
    skew_s: float
    max_age_s: float
    reason: str


@dataclass(frozen=True)
class ActuatorConfig:
    wheel_radius_m: float
    wheel_separation_m: float
    response_space: str
    max_linear_mps: float
    max_angular_radps: float
    fixed_delay_s: float
    uniform_jitter_s: float
    packet_loss_probability: float
    dropout_windows_s: tuple[tuple[float, float], ...]
    left_gain: float
    right_gain: float
    left_deadband_mps: float
    right_deadband_mps: float
    time_constant_s: float
    acceleration_mps2: float
    braking_mps2: float
    watchdog_timeout_s: float
    target_force_nm: float

    @classmethod
    def from_robot_profile(cls, robot: Mapping[str, Any]) -> "ActuatorConfig":
        actuator = robot["actuator"]
        return cls(
            wheel_radius_m=float(robot["wheel_radius_m"]),
            wheel_separation_m=float(robot["wheel_separation_m"]),
            response_space=str(actuator["response_space"]),
            max_linear_mps=float(actuator["max_linear_mps"]),
            max_angular_radps=float(actuator["max_angular_radps"]),
            fixed_delay_s=float(actuator["fixed_delay_s"]),
            uniform_jitter_s=float(actuator["uniform_jitter_s"]),
            packet_loss_probability=float(actuator["packet_loss_probability"]),
            dropout_windows_s=tuple(
                (float(window[0]), float(window[1]))
                for window in actuator.get("dropout_windows_s", [])
            ),
            left_gain=float(actuator["left_gain"]),
            right_gain=float(actuator["right_gain"]),
            left_deadband_mps=float(actuator["left_deadband_mps"]),
            right_deadband_mps=float(actuator["right_deadband_mps"]),
            time_constant_s=float(actuator["time_constant_s"]),
            acceleration_mps2=float(actuator["acceleration_mps2"]),
            braking_mps2=float(actuator["braking_mps2"]),
            watchdog_timeout_s=float(actuator["watchdog_timeout_s"]),
            target_force_nm=float(actuator["target_force_nm"]),
        )


class ParkMillerRng:
    """Small RNG with identical arithmetic in Python and Lua doubles."""

    _MODULUS = 2_147_483_647
    _MULTIPLIER = 16_807

    def __init__(self, seed: int) -> None:
        self.state = int(seed) % self._MODULUS
        if self.state <= 0:
            self.state += self._MODULUS - 1

    def random(self) -> float:
        self.state = (self.state * self._MULTIPLIER) % self._MODULUS
        return self.state / self._MODULUS


def _numbers(raw: str | None, count: int, default: Sequence[float]) -> np.ndarray:
    if not raw:
        return np.asarray(default, dtype=np.float64)
    values = np.asarray([float(value) for value in raw.split()], dtype=np.float64)
    if values.shape != (count,) or not np.all(np.isfinite(values)):
        raise ValueError(f"expected {count} finite values, got {raw!r}")
    return values


def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _origin(node: ET.Element | None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    if node is None:
        return transform
    transform[:3, :3] = _rpy_matrix(_numbers(node.attrib.get("rpy"), 3, (0.0, 0.0, 0.0)))
    transform[:3, 3] = _numbers(node.attrib.get("xyz"), 3, (0.0, 0.0, 0.0))
    return transform


def _link_transforms(root: ET.Element, root_link: str) -> dict[str, np.ndarray]:
    children: dict[str, list[tuple[str, np.ndarray]]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.attrib.get("link")
        child_name = child.attrib.get("link")
        if parent_name and child_name:
            children.setdefault(parent_name, []).append((child_name, _origin(joint.find("origin"))))

    transforms = {str(root_link): np.eye(4, dtype=np.float64)}
    stack = [str(root_link)]
    while stack:
        parent_name = stack.pop()
        for child_name, transform in children.get(parent_name, []):
            if child_name in transforms:
                raise ValueError(f"URDF contains multiple paths to link {child_name!r}")
            transforms[child_name] = transforms[parent_name] @ transform
            stack.append(child_name)
    return transforms


def aggregate_urdf_inertials(
    urdf_path: str | Path,
    *,
    root_link: str = "base_link",
    excluded_links: Iterable[str] = (),
) -> RigidBodyAggregate:
    """Aggregate selected URDF link inertials in ``root_link`` coordinates."""

    path = Path(urdf_path)
    root = ET.parse(path).getroot()
    transforms = _link_transforms(root, root_link)
    excluded = frozenset(str(name) for name in excluded_links)
    entries: list[tuple[str, float, np.ndarray, np.ndarray]] = []
    for link in root.findall("link"):
        name = str(link.attrib.get("name", ""))
        if not name or name in excluded or name not in transforms:
            continue
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_node = inertial.find("mass")
        inertia_node = inertial.find("inertia")
        if mass_node is None or inertia_node is None:
            continue
        mass = float(mass_node.attrib.get("value", "nan"))
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError(f"link {name!r} has invalid mass")
        inertia = np.asarray(
            [
                [float(inertia_node.attrib.get("ixx", "nan")), float(inertia_node.attrib.get("ixy", 0.0)), float(inertia_node.attrib.get("ixz", 0.0))],
                [float(inertia_node.attrib.get("ixy", 0.0)), float(inertia_node.attrib.get("iyy", "nan")), float(inertia_node.attrib.get("iyz", 0.0))],
                [float(inertia_node.attrib.get("ixz", 0.0)), float(inertia_node.attrib.get("iyz", 0.0)), float(inertia_node.attrib.get("izz", "nan"))],
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(inertia)):
            raise ValueError(f"link {name!r} has non-finite inertia")
        inertial_transform = transforms[name] @ _origin(inertial.find("origin"))
        rotation = inertial_transform[:3, :3]
        entries.append((name, mass, inertial_transform[:3, 3], rotation @ inertia @ rotation.T))

    if not entries:
        raise ValueError(f"URDF {path} has no selected inertials below {root_link!r}")
    total_mass = sum(entry[1] for entry in entries)
    center = sum((mass * position for _name, mass, position, _inertia in entries), np.zeros(3)) / total_mass
    aggregate = np.zeros((3, 3), dtype=np.float64)
    for _name, mass, position, inertia in entries:
        offset = position - center
        aggregate += inertia + mass * ((offset @ offset) * np.eye(3) - np.outer(offset, offset))
    aggregate = 0.5 * (aggregate + aggregate.T)
    eigenvalues = np.linalg.eigvalsh(aggregate)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError(f"aggregated inertia is not positive definite: {eigenvalues.tolist()}")
    return RigidBodyAggregate(
        mass_kg=float(total_mass),
        center_of_mass_m=tuple(float(value) for value in center),
        inertia_kg_m2=tuple(float(value) for value in aggregate.reshape(-1)),
        links=tuple(sorted(entry[0] for entry in entries)),
    )


def urdf_link_inertial(urdf_path: str | Path, link_name: str) -> RigidBodyAggregate:
    """Return one link's inertial expressed in that link's frame."""

    root = ET.parse(Path(urdf_path)).getroot()
    link = next((item for item in root.findall("link") if item.attrib.get("name") == link_name), None)
    if link is None or link.find("inertial") is None:
        raise ValueError(f"URDF link {link_name!r} has no inertial")
    inertial = link.find("inertial")
    assert inertial is not None
    mass_node = inertial.find("mass")
    inertia_node = inertial.find("inertia")
    if mass_node is None or inertia_node is None:
        raise ValueError(f"URDF link {link_name!r} has incomplete inertial")
    mass = _finite_number(mass_node.attrib.get("value"), f"link {link_name}.mass", positive=True)
    transform = _origin(inertial.find("origin"))
    raw = np.asarray(
        [
            [float(inertia_node.attrib.get("ixx", "nan")), float(inertia_node.attrib.get("ixy", 0.0)), float(inertia_node.attrib.get("ixz", 0.0))],
            [float(inertia_node.attrib.get("ixy", 0.0)), float(inertia_node.attrib.get("iyy", "nan")), float(inertia_node.attrib.get("iyz", 0.0))],
            [float(inertia_node.attrib.get("ixz", 0.0)), float(inertia_node.attrib.get("iyz", 0.0)), float(inertia_node.attrib.get("izz", "nan"))],
        ],
        dtype=np.float64,
    )
    rotation = transform[:3, :3]
    inertia = rotation @ raw @ rotation.T
    return RigidBodyAggregate(
        mass_kg=mass,
        center_of_mass_m=tuple(float(value) for value in transform[:3, 3]),
        inertia_kg_m2=tuple(float(value) for value in inertia.reshape(-1)),
        links=(str(link_name),),
    )


def canonical_profile_bytes(profile: Mapping[str, Any]) -> bytes:
    payload = copy.deepcopy(dict(profile))
    payload.pop("checksum", None)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def profile_checksum(profile: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_profile_bytes(profile)).hexdigest()


def _finite_number(value: Any, path: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{path} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{path} must be nonnegative")
    return result


def _finite_vector(value: Any, size: int, path: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{path} must contain {size} values")
    return [_finite_number(item, f"{path}[{index}]") for index, item in enumerate(value)]


def validate_profile(profile: Mapping[str, Any], *, verify_checksum: bool = True) -> dict[str, Any]:
    """Strictly validate the V2.1 fields used at runtime."""

    value = copy.deepcopy(dict(profile))
    if value.get("$schema") != "trackmaker_digital_twin_profile.schema.json":
        raise ValueError("$schema must identify the V2.1 digital-twin schema")
    if value.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {PROFILE_SCHEMA_VERSION!r}")
    if not isinstance(value.get("profile_id"), str) or not value["profile_id"].strip():
        raise ValueError("profile_id must be a non-empty string")
    if value.get("provenance") not in PROFILE_PROVENANCE:
        raise ValueError("provenance must be 'prior' or 'measured'")
    if value.get("calibration_state") not in {"uncalibrated", "calibrated"}:
        raise ValueError("calibration_state must be uncalibrated or calibrated")
    if value["provenance"] == "prior" and value["calibration_state"] != "uncalibrated":
        raise ValueError("prior profiles must remain marked uncalibrated")
    if value["provenance"] == "measured" and value["calibration_state"] != "calibrated":
        raise ValueError("measured profiles must be marked calibrated")
    if isinstance(value.get("seed"), bool) or not isinstance(value.get("seed"), int):
        raise ValueError("seed must be an integer")
    checksum = value.get("checksum")
    if not isinstance(checksum, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", checksum) is None:
        raise ValueError("checksum must be sha256:<64 lowercase hex digits>")
    if verify_checksum and checksum != profile_checksum(value):
        raise ValueError(f"profile checksum mismatch: expected {profile_checksum(value)}, got {checksum}")

    simulation = value.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError("simulation must be an object")
    if simulation.get("engine") != "Bullet" or int(simulation.get("engine_index", -1)) != 0:
        raise ValueError("simulation engine must be Bullet with engine_index=0")
    if isinstance(simulation.get("engine_version"), bool) or not isinstance(simulation.get("engine_version"), int):
        raise ValueError("simulation.engine_version must be an integer")
    if not isinstance(simulation.get("realtime"), bool):
        raise ValueError("simulation.realtime must be boolean")
    outer = _finite_number(simulation.get("outer_step_s"), "simulation.outer_step_s", positive=True)
    inner = _finite_number(simulation.get("physics_step_s"), "simulation.physics_step_s", positive=True)
    if not math.isclose(outer, 0.05, abs_tol=1e-12) or not math.isclose(inner, 0.005, abs_tol=1e-12):
        raise ValueError("V2.1 requires 50 ms outer and 5 ms physics steps")
    if int(simulation.get("solver_iterations", 0)) != 100:
        raise ValueError("V2.1 requires 100 solver iterations")
    _finite_vector(simulation.get("gravity_mps2"), 3, "simulation.gravity_mps2")

    materials = value.get("materials")
    if not isinstance(materials, dict):
        raise ValueError("materials must be an object")
    for name in ("floor", "obstacle", "chassis", "wheel", "caster"):
        material = materials.get(name)
        if not isinstance(material, dict):
            raise ValueError(f"materials.{name} must be an object")
        _finite_number(material.get("friction"), f"materials.{name}.friction", nonnegative=True)
        _finite_number(material.get("restitution"), f"materials.{name}.restitution", nonnegative=True)
        _finite_number(material.get("linear_damping"), f"materials.{name}.linear_damping", nonnegative=True)
        _finite_number(material.get("angular_damping"), f"materials.{name}.angular_damping", nonnegative=True)

    robots = value.get("robots")
    if not isinstance(robots, dict) or set(robots) != set(ROLES):
        raise ValueError("robots must contain exactly defender and attacker")
    for role in ROLES:
        robot = robots[role]
        if not isinstance(robot, dict):
            raise ValueError(f"robots.{role} must be an object")
        _finite_number(robot.get("mass_kg"), f"robots.{role}.mass_kg", positive=True)
        _finite_vector(robot.get("center_of_mass_m"), 3, f"robots.{role}.center_of_mass_m")
        inertia = np.asarray(_finite_vector(robot.get("inertia_kg_m2"), 9, f"robots.{role}.inertia_kg_m2")).reshape(3, 3)
        if not np.allclose(inertia, inertia.T, atol=1e-10) or np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
            raise ValueError(f"robots.{role}.inertia_kg_m2 must be symmetric positive definite")
        _finite_number(robot.get("wheel_radius_m"), f"robots.{role}.wheel_radius_m", positive=True)
        _finite_number(robot.get("wheel_separation_m"), f"robots.{role}.wheel_separation_m", positive=True)
        chassis = robot.get("chassis")
        caster = robot.get("caster")
        wheel = robot.get("wheel")
        geometry = robot.get("geometry")
        actuator = robot.get("actuator")
        if not all(isinstance(section, dict) for section in (chassis, caster, wheel, geometry, actuator)):
            raise ValueError(f"robots.{role} chassis, caster, wheel, geometry, and actuator must be objects")
        _finite_number(chassis.get("mass_kg"), f"robots.{role}.chassis.mass_kg", positive=True)
        if robot.get("inertia_frame") != "base_link" or chassis.get("inertia_frame") != "base_link":
            raise ValueError(f"robots.{role} aggregate inertias must use base_link frame")
        _finite_vector(chassis.get("center_of_mass_m"), 3, f"robots.{role}.chassis.center_of_mass_m")
        _finite_vector(chassis.get("inertia_kg_m2"), 9, f"robots.{role}.chassis.inertia_kg_m2")
        if not isinstance(chassis.get("aggregated_links"), list) or not chassis["aggregated_links"]:
            raise ValueError(f"robots.{role}.chassis.aggregated_links must be non-empty")
        _finite_number(caster.get("mass_kg"), f"robots.{role}.caster.mass_kg", positive=True)
        _finite_vector(caster.get("center_of_mass_m"), 3, f"robots.{role}.caster.center_of_mass_m")
        _finite_vector(caster.get("inertia_kg_m2"), 9, f"robots.{role}.caster.inertia_kg_m2")
        if caster.get("inertia_frame") != "caster_link":
            raise ValueError(f"robots.{role}.caster.inertia_frame must be caster_link")
        _finite_number(wheel.get("mass_kg_each"), f"robots.{role}.wheel.mass_kg_each", positive=True)
        if wheel.get("inertia_frame") != "wheel_link":
            raise ValueError(f"robots.{role}.wheel.inertia_frame must be wheel_link")
        _finite_vector(wheel.get("inertia_kg_m2_each"), 9, f"robots.{role}.wheel.inertia_kg_m2_each")
        component_mass = float(chassis["mass_kg"]) + float(caster["mass_kg"]) + 2.0 * float(wheel["mass_kg_each"])
        if not math.isclose(float(robot["mass_kg"]), component_mass, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"robots.{role}.mass_kg must equal chassis + caster + two wheels")
        for key in ("base", "shell", "bumper", "caster", "left_wheel", "right_wheel"):
            if not isinstance(geometry.get(key), dict):
                raise ValueError(f"robots.{role}.geometry.{key} must be an object")
        if geometry.get("visual_mesh_respondable") is not False:
            raise ValueError(f"robots.{role}.geometry.visual_mesh_respondable must be false")
        if geometry.get("wheel_drop_enabled") is not False:
            raise ValueError(f"robots.{role}.geometry.wheel_drop_enabled must be false")
        if geometry.get("create3_reflex_simulated") is not False:
            raise ValueError(f"robots.{role}.geometry.create3_reflex_simulated must be false")
        if actuator.get("response_space") != "body_twist":
            raise ValueError(f"robots.{role}.actuator.response_space must be body_twist")
        caster_geometry = geometry["caster"]
        if (
            caster_geometry.get("collision_model") != "equivalent_front_slider"
            or caster_geometry.get("mount_model") != "passive_spherical_joint"
            or caster_geometry.get("contact_material") != "caster"
        ):
            raise ValueError(f"robots.{role}.geometry.caster must use the V2.1 passive support model")
        for key in (
            "max_linear_mps",
            "max_angular_radps",
            "fixed_delay_s",
            "uniform_jitter_s",
            "packet_loss_probability",
            "left_gain",
            "right_gain",
            "left_deadband_mps",
            "right_deadband_mps",
            "time_constant_s",
            "acceleration_mps2",
            "braking_mps2",
            "watchdog_timeout_s",
            "target_force_nm",
        ):
            _finite_number(actuator.get(key), f"robots.{role}.actuator.{key}", nonnegative=True)
        loss = float(actuator["packet_loss_probability"])
        if not 0.0 <= loss <= 1.0:
            raise ValueError(f"robots.{role}.actuator.packet_loss_probability must be in [0,1]")
        windows = actuator.get("dropout_windows_s")
        if not isinstance(windows, list):
            raise ValueError(f"robots.{role}.actuator.dropout_windows_s must be a list")
        previous_end = -math.inf
        for index, window in enumerate(windows):
            bounds = _finite_vector(window, 2, f"robots.{role}.actuator.dropout_windows_s[{index}]")
            if bounds[0] < 0.0 or bounds[1] <= bounds[0] or bounds[0] < previous_end:
                raise ValueError(f"robots.{role}.actuator.dropout windows must be ordered and non-overlapping")
            previous_end = bounds[1]
    return value


def load_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path).expanduser().resolve()
    return validate_profile(json.loads(profile_path.read_text(encoding="utf-8")))


def _default_geometry(wheel_radius: float, wheel_separation: float) -> dict[str, Any]:
    wheel_z = 0.0402
    base_z = 0.0492
    caster_radius = 0.025
    caster_preload = 0.0
    caster_z = round(wheel_z - wheel_radius + caster_radius - caster_preload, 12)
    nominal_base_z = 0.02 - (wheel_z - wheel_radius)
    return {
        "nominal_base_z_m": nominal_base_z,
        "base_clearance_m": nominal_base_z + base_z - 0.03 - 0.02,
        "base": {"shape": "cylinder", "center_m": [0.0, 0.0, base_z], "radius_m": 0.164, "height_m": 0.06},
        "shell": {"shape": "cylinder", "center_m": [0.0, 0.0, 0.19], "radius_m": 0.14, "height_m": 0.25},
        "bumper": {"shape": "cuboid", "center_m": [0.105, 0.0, 0.072], "size_m": [0.14, 0.285, 0.065]},
        "caster": {
            "shape": "sphere",
            "center_m": [0.125, 0.0, caster_z],
            "radius_m": caster_radius,
            "preload_m": caster_preload,
            "collision_model": "equivalent_front_slider",
            "mount_model": "passive_spherical_joint",
            "contact_material": "caster",
        },
        "left_wheel": {"shape": "cylinder", "center_m": [0.0, wheel_separation * 0.5, wheel_z], "radius_m": wheel_radius, "width_m": 0.015},
        "right_wheel": {"shape": "cylinder", "center_m": [0.0, -wheel_separation * 0.5, wheel_z], "radius_m": wheel_radius, "width_m": 0.015},
        "visual_mesh_respondable": False,
        "wheel_drop_enabled": False,
        "create3_reflex_simulated": False,
    }


def build_prior_profile(urdf_path: str | Path, *, seed: int = 20260815) -> dict[str, Any]:
    """Build the checked-in prior profile from the prepared TurtleBot4 URDF."""

    from coppelia_env.turtlebot4 import load_turtlebot4_spec

    path = Path(urdf_path).resolve()
    spec = load_turtlebot4_spec(path)
    full = aggregate_urdf_inertials(path)
    chassis = aggregate_urdf_inertials(path, excluded_links=CHASSIS_EXCLUDED_LINKS)
    caster = urdf_link_inertial(path, CASTER_LINK)
    left_wheel = urdf_link_inertial(path, "left_wheel")
    right_wheel = urdf_link_inertial(path, "right_wheel")
    if not math.isclose(left_wheel.mass_kg, right_wheel.mass_kg, rel_tol=1e-9):
        raise ValueError("left and right URDF wheel masses differ")
    wheel_inertia = np.mean(
        [np.asarray(left_wheel.inertia_kg_m2), np.asarray(right_wheel.inertia_kg_m2)], axis=0
    ).tolist()
    geometry = _default_geometry(spec.wheel_radius_m, spec.wheel_separation_m)
    base_robot = {
        "mass_kg": full.mass_kg,
        "center_of_mass_m": list(full.center_of_mass_m),
        "inertia_kg_m2": list(full.inertia_kg_m2),
        "inertia_frame": "base_link",
        "wheel_radius_m": spec.wheel_radius_m,
        "wheel_separation_m": spec.wheel_separation_m,
        "chassis": {**chassis.to_dict(), "inertia_frame": "base_link"},
        "caster": {
            "mass_kg": caster.mass_kg,
            "center_of_mass_m": list(caster.center_of_mass_m),
            "inertia_kg_m2": list(caster.inertia_kg_m2),
            "inertia_frame": "caster_link",
        },
        "wheel": {
            "mass_kg_each": left_wheel.mass_kg,
            "inertia_kg_m2_each": wheel_inertia,
            "inertia_frame": "wheel_link",
        },
        "geometry": geometry,
    }
    profile: dict[str, Any] = {
        "$schema": "trackmaker_digital_twin_profile.schema.json",
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_id": f"trackmaker_turtlebot4_v2_1_body_twist_prior_{int(seed)}",
        "provenance": "prior",
        "calibration_state": "uncalibrated",
        "seed": int(seed),
        "checksum": "",
        "source": {
            "kind": "urdf_and_engineering_prior",
            "urdf": str(path.relative_to(Path(__file__).resolve().parents[1])),
            "urdf_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "generator": "coppelia_env.digital_twin.build_prior_profile",
        },
        "simulation": {
            "engine": "Bullet",
            "engine_index": 0,
            "engine_version": 0,
            "outer_step_s": 0.05,
            "physics_step_s": 0.005,
            "solver_iterations": 100,
            "gravity_mps2": [0.0, 0.0, -9.81],
            "realtime": True,
        },
        "materials": {
            "floor": {"friction": 0.85, "restitution": 0.0, "linear_damping": 0.0, "angular_damping": 0.0},
            "obstacle": {"friction": 0.70, "restitution": 0.0, "linear_damping": 0.0, "angular_damping": 0.0},
            "chassis": {"friction": 0.55, "restitution": 0.0, "linear_damping": 0.02, "angular_damping": 0.02},
            "wheel": {"friction": 1.15, "restitution": 0.0, "linear_damping": 0.0, "angular_damping": 0.0},
            "caster": {"friction": 0.0, "restitution": 0.0, "linear_damping": 0.01, "angular_damping": 0.01},
        },
        "robots": {},
    }
    actuator_common = {
        "response_space": "body_twist",
        "fixed_delay_s": 0.02,
        "uniform_jitter_s": 0.005,
        "packet_loss_probability": 0.0,
        "dropout_windows_s": [],
        "left_gain": 0.96,
        "right_gain": 0.96,
        "left_deadband_mps": 0.003,
        "right_deadband_mps": 0.003,
        "time_constant_s": 0.08,
        "acceleration_mps2": 0.05,
        "braking_mps2": 1.8,
        "watchdog_timeout_s": 0.5,
        "target_force_nm": 0.05,
    }
    for role, max_linear, max_angular in (
        ("defender", 0.234, math.radians(54.0)),
        ("attacker", 0.180, math.radians(108.0)),
    ):
        robot = copy.deepcopy(base_robot)
        robot["actuator"] = {
            **actuator_common,
            "max_linear_mps": max_linear,
            "max_angular_radps": max_angular,
        }
        profile["robots"][role] = robot
    profile["checksum"] = profile_checksum(profile)
    return validate_profile(profile)


def evaluate_input_timing(
    stamps: Iterable[float],
    *,
    now: float,
    stale_after_s: float,
    max_skew_s: float = 0.05,
    required_count: int = 5,
) -> InputTiming:
    values = [float(value) for value in stamps]
    current = float(now)
    if len(values) != int(required_count):
        return InputTiming(False, math.inf, math.inf, "missing")
    if not math.isfinite(current) or not all(math.isfinite(value) for value in values):
        return InputTiming(False, math.inf, math.inf, "non_finite")
    ages = [current - value for value in values]
    skew = max(values) - min(values)
    max_age = max(ages)
    if any(age < -1e-9 for age in ages):
        return InputTiming(False, skew, max_age, "future")
    if max_age > max(0.0, float(stale_after_s)):
        return InputTiming(False, skew, max_age, "stale")
    if skew > max(0.0, float(max_skew_s)) + 1e-12:
        return InputTiming(False, skew, max_age, "skew")
    return InputTiming(True, skew, max_age, "fresh")


class DeterministicActuator:
    """Reference cmd_vel queue and body-velocity actuator used by tests/fitting."""

    def __init__(self, config: ActuatorConfig, *, seed: int, start_time_s: float = 0.0) -> None:
        self.config = config
        self.rng = ParkMillerRng(seed)
        self.start_time_s = float(start_time_s)
        self.sequence = 0
        self.queue: list[dict[str, Any]] = []
        self.active_command = (0.0, 0.0)
        self.lag_state = [0.0, 0.0]
        self.body_linear_state = 0.0
        self.output_state = [0.0, 0.0]
        self.last_execution_time_s = -math.inf
        self.watchdog_active = True
        self.last_target_wheel_radps = (0.0, 0.0)

    def enqueue(self, receive_time_s: float, linear_mps: float, angular_radps: float) -> dict[str, Any]:
        receive = float(receive_time_s)
        finite = all(math.isfinite(float(value)) for value in (receive, linear_mps, angular_radps))
        requested = (float(linear_mps) if finite else 0.0, float(angular_radps) if finite else 0.0)
        clamped = (
            max(-self.config.max_linear_mps, min(self.config.max_linear_mps, requested[0])),
            max(-self.config.max_angular_radps, min(self.config.max_angular_radps, requested[1])),
        )
        jitter = (2.0 * self.rng.random() - 1.0) * self.config.uniform_jitter_s
        scheduled = max(receive, receive + self.config.fixed_delay_s + jitter)
        elapsed = receive - self.start_time_s
        preset_dropout = any(first <= elapsed < second for first, second in self.config.dropout_windows_s)
        random_dropout = self.rng.random() < self.config.packet_loss_probability
        dropped = preset_dropout or random_dropout
        self.sequence += 1
        event = {
            "sequence": self.sequence,
            "receive_time_s": receive,
            "scheduled_time_s": scheduled,
            "execute_time_s": None,
            "requested_linear_mps": requested[0],
            "requested_angular_radps": requested[1],
            "clamped_linear_mps": clamped[0],
            "clamped_angular_radps": clamped[1],
            "finite": finite,
            "jitter_s": jitter,
            "dropped": dropped,
            "drop_reason": "preset_outage" if preset_dropout else ("packet_loss" if random_dropout else ""),
        }
        if not dropped:
            self.queue.append(event)
            self.queue.sort(key=lambda item: (item["scheduled_time_s"], item["sequence"]))
        return copy.deepcopy(event)

    @staticmethod
    def _deadband(value: float, gain: float, deadband: float) -> float:
        return math.copysign(max(0.0, abs(value) - deadband) * gain, value) if value else 0.0

    @staticmethod
    def _slew(current: float, desired: float, dt: float, accel: float, brake: float) -> float:
        accelerating = current * desired >= 0.0 and abs(desired) > abs(current)
        limit = accel if accelerating else brake
        delta = max(-limit * dt, min(limit * dt, desired - current))
        return current + delta

    def step(
        self,
        now_s: float,
        dt_s: float,
        *,
        actual_wheel_radps: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        now = float(now_s)
        dt = max(0.0, float(dt_s))
        executed: list[dict[str, Any]] = []
        while self.queue and self.queue[0]["scheduled_time_s"] <= now + 1e-12:
            event = self.queue.pop(0)
            event["execute_time_s"] = now
            self.active_command = (event["clamped_linear_mps"], event["clamped_angular_radps"])
            self.last_execution_time_s = now
            executed.append(copy.deepcopy(event))
        self.watchdog_active = now - self.last_execution_time_s > self.config.watchdog_timeout_s
        linear, angular = (0.0, 0.0) if self.watchdog_active else self.active_command
        if self.config.time_constant_s <= 1e-12:
            alpha = 1.0
        else:
            alpha = 1.0 - math.exp(-dt / self.config.time_constant_s)
        self.lag_state[0] += alpha * (linear - self.lag_state[0])
        self.lag_state[1] += alpha * (angular - self.lag_state[1])
        self.body_linear_state = self._slew(
            self.body_linear_state,
            self.lag_state[0],
            dt,
            self.config.acceleration_mps2,
            self.config.braking_mps2,
        )
        half_track = 0.5 * self.config.wheel_separation_m
        requested_sides = (
            self.body_linear_state - half_track * self.lag_state[1],
            self.body_linear_state + half_track * self.lag_state[1],
        )
        self.output_state[0] = self._deadband(
            requested_sides[0], self.config.left_gain, self.config.left_deadband_mps
        )
        self.output_state[1] = self._deadband(
            requested_sides[1], self.config.right_gain, self.config.right_deadband_mps
        )
        radius = max(self.config.wheel_radius_m, 1e-12)
        target = (self.output_state[0] / radius, self.output_state[1] / radius)
        actual = target if actual_wheel_radps is None else tuple(float(value) for value in actual_wheel_radps)
        if len(actual) != 2 or not all(math.isfinite(value) for value in actual):
            raise ValueError("actual_wheel_radps must contain two finite values")
        self.last_target_wheel_radps = target
        filtered_linear = 0.5 * (self.output_state[0] + self.output_state[1])
        filtered_angular = (self.output_state[1] - self.output_state[0]) / max(self.config.wheel_separation_m, 1e-12)
        return {
            "time_s": now,
            "executed_events": executed,
            "watchdog_active": self.watchdog_active,
            "requested_linear_mps": linear,
            "requested_angular_radps": angular,
            "filtered_linear_mps": filtered_linear,
            "filtered_angular_radps": filtered_angular,
            "target_wheel_radps": target,
            "actual_wheel_radps": actual,
            "queue_depth": len(self.queue),
        }


def runtime_mismatches(
    profile: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    absolute_tolerance: float = 1e-6,
) -> list[str]:
    """Compare startup introspection data with the loaded profile."""

    expected = validate_profile(profile)
    errors: list[str] = []

    def close(path: str, actual: Any, wanted: Any, tolerance: float = absolute_tolerance) -> None:
        if isinstance(wanted, (int, float)) and not isinstance(wanted, bool):
            try:
                if not math.isclose(float(actual), float(wanted), rel_tol=1e-6, abs_tol=tolerance):
                    errors.append(f"{path}: expected {wanted!r}, got {actual!r}")
            except (TypeError, ValueError):
                errors.append(f"{path}: expected numeric {wanted!r}, got {actual!r}")
        elif actual != wanted:
            errors.append(f"{path}: expected {wanted!r}, got {actual!r}")

    simulation = expected["simulation"]
    close("profile_checksum", runtime.get("profile_checksum"), expected["checksum"])
    close("engine_index", runtime.get("engine_index"), simulation["engine_index"])
    close("engine_version", runtime.get("engine_version"), simulation["engine_version"])
    close("outer_step_s", runtime.get("outer_step_s"), simulation["outer_step_s"], 1e-9)
    close("physics_step_s", runtime.get("physics_step_s"), simulation["physics_step_s"], 1e-9)
    close("solver_iterations", runtime.get("solver_iterations"), simulation["solver_iterations"])
    gravity = runtime.get("gravity_mps2")
    if not isinstance(gravity, (list, tuple)) or len(gravity) != 3:
        errors.append("gravity_mps2: missing vector")
    else:
        for index, wanted in enumerate(simulation["gravity_mps2"]):
            close(f"gravity_mps2[{index}]", gravity[index], wanted)
    robots = runtime.get("robots")
    if not isinstance(robots, Mapping):
        return errors + ["robots: missing runtime data"]
    for role in ROLES:
        actual = robots.get(role)
        if not isinstance(actual, Mapping):
            errors.append(f"robots.{role}: missing runtime data")
            continue
        robot = expected["robots"][role]
        close(f"robots.{role}.chassis_mass_kg", actual.get("chassis_mass_kg"), robot["chassis"]["mass_kg"], 1e-5)
        close(f"robots.{role}.caster_mass_kg", actual.get("caster_mass_kg"), robot["caster"]["mass_kg"], 1e-6)
        close(
            f"robots.{role}.caster_mount_model",
            actual.get("caster_mount_model"),
            robot["geometry"]["caster"]["mount_model"],
        )
        close(f"robots.{role}.wheel_radius_m", actual.get("wheel_radius_m"), robot["wheel_radius_m"], 1e-5)
        close(f"robots.{role}.wheel_separation_m", actual.get("wheel_separation_m"), robot["wheel_separation_m"], 1e-5)
        close(f"robots.{role}.target_force_nm", actual.get("target_force_nm"), robot["actuator"]["target_force_nm"], 1e-5)
    return errors


def profile_metadata(profile: Mapping[str, Any]) -> dict[str, Any]:
    value = validate_profile(profile)
    return {
        "schema_version": value["schema_version"],
        "profile_id": value["profile_id"],
        "provenance": value["provenance"],
        "calibration_state": value["calibration_state"],
        "actuator_response_space": "body_twist",
        "seed": value["seed"],
        "checksum": value["checksum"],
    }


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate the TrackMaker V2.1 engineering-prior profile")
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    profile = build_prior_profile(args.urdf, seed=args.seed)
    text = json.dumps(profile, indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(text, end="")
        return 0
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing profile: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
