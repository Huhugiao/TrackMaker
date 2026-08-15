"""Portable calibration plans, synthetic data, and SciPy fitting for V2.1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from coppelia_env.digital_twin import (
    ActuatorConfig,
    DeterministicActuator,
    profile_checksum,
    validate_profile,
)


CALIBRATION_DATA_VERSION = "trackmaker.digital_twin_calibration.v2.1"
REQUIRED_CATEGORIES = frozenset(
    {"straight", "turn", "arc", "step", "ramp", "emergency_stop", "watchdog_outage"}
)


@dataclass(frozen=True)
class CalibrationPhase:
    name: str
    category: str
    duration_s: float
    mode: str = "constant"
    linear_start_mps: float = 0.0
    linear_end_mps: float = 0.0
    angular_start_radps: float = 0.0
    angular_end_radps: float = 0.0
    publish_commands: bool = True

    def command(self, elapsed_s: float) -> tuple[float, float]:
        fraction = 0.0
        if self.mode == "ramp":
            fraction = max(0.0, min(1.0, float(elapsed_s) / max(float(self.duration_s), 1e-12)))
        elif self.mode != "constant":
            raise ValueError(f"unsupported calibration phase mode: {self.mode}")
        linear = self.linear_start_mps + fraction * (self.linear_end_mps - self.linear_start_mps)
        angular = self.angular_start_radps + fraction * (self.angular_end_radps - self.angular_start_radps)
        return float(linear), float(angular)


def calibration_plan(robot_profile: Mapping[str, Any]) -> list[CalibrationPhase]:
    """Return a bounded excitation plan covering every required V2.1 behavior."""

    actuator = robot_profile["actuator"]
    vmax = float(actuator["max_linear_mps"])
    wmax = float(actuator["max_angular_radps"])
    return [
        CalibrationPhase("settle", "step", 0.50),
        CalibrationPhase("straight_sub_deadband", "straight", 0.80, linear_start_mps=0.002, linear_end_mps=0.002),
        CalibrationPhase("straight_low", "straight", 0.90, linear_start_mps=min(0.015, 0.15 * vmax), linear_end_mps=min(0.015, 0.15 * vmax)),
        CalibrationPhase("straight_mid", "straight", 2.00, linear_start_mps=0.45 * vmax, linear_end_mps=0.45 * vmax),
        CalibrationPhase("straight_high", "straight", 2.00, linear_start_mps=0.82 * vmax, linear_end_mps=0.82 * vmax),
        CalibrationPhase("straight_reverse", "straight", 2.40, linear_start_mps=-0.45 * vmax, linear_end_mps=-0.45 * vmax),
        CalibrationPhase("reverse_ramp_stop", "ramp", 3.00, mode="ramp", linear_start_mps=-0.45 * vmax),
        CalibrationPhase("turn_settle", "step", 0.50),
        CalibrationPhase("turn_left_low", "turn", 1.50, angular_start_radps=0.28 * wmax, angular_end_radps=0.28 * wmax),
        CalibrationPhase("turn_left_high", "turn", 3.50, angular_start_radps=0.72 * wmax, angular_end_radps=0.72 * wmax),
        CalibrationPhase("turn_right_low", "turn", 1.50, angular_start_radps=-0.28 * wmax, angular_end_radps=-0.28 * wmax),
        CalibrationPhase("turn_right_high", "turn", 3.50, angular_start_radps=-0.72 * wmax, angular_end_radps=-0.72 * wmax),
        CalibrationPhase("arc_left", "arc", 2.50, linear_start_mps=0.42 * vmax, linear_end_mps=0.42 * vmax, angular_start_radps=0.42 * wmax, angular_end_radps=0.42 * wmax),
        CalibrationPhase("arc_right", "arc", 2.50, linear_start_mps=0.42 * vmax, linear_end_mps=0.42 * vmax, angular_start_radps=-0.42 * wmax, angular_end_radps=-0.42 * wmax),
        CalibrationPhase("step_zero", "step", 0.50),
        CalibrationPhase("step_positive", "step", 3.80, linear_start_mps=0.75 * vmax, linear_end_mps=0.75 * vmax),
        CalibrationPhase("ramp_up", "ramp", 4.00, mode="ramp", linear_end_mps=0.80 * vmax),
        CalibrationPhase("emergency_stop_prime", "emergency_stop", 3.80, linear_start_mps=0.80 * vmax, linear_end_mps=0.80 * vmax),
        CalibrationPhase("emergency_stop", "emergency_stop", 1.00),
        CalibrationPhase("watchdog_prime", "watchdog_outage", 2.80, linear_start_mps=0.55 * vmax, linear_end_mps=0.55 * vmax),
        CalibrationPhase("watchdog_outage", "watchdog_outage", 1.10, publish_commands=False),
        CalibrationPhase("final_stop", "step", 1.00),
    ]


def validate_calibration_dataset(dataset: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(dataset))
    if value.get("schema_version") != CALIBRATION_DATA_VERSION:
        raise ValueError(f"unsupported calibration dataset schema: {value.get('schema_version')!r}")
    source = value.get("source_profile")
    if not isinstance(source, dict) or source.get("provenance") not in {"prior", "measured"}:
        raise ValueError("calibration dataset must identify its source profile")
    if source.get("provenance") == "prior" and source.get("calibration_state") != "uncalibrated":
        raise ValueError("prior calibration data must remain marked uncalibrated")
    graceful = value.get("graceful_shutdown")
    if not isinstance(graceful, dict) or not graceful.get("zero_command_sent"):
        raise ValueError("calibration dataset must record a graceful zero command")
    if not graceful.get("actual_wheel_stopped"):
        raise ValueError("calibration dataset must record stopped actual wheel speeds")
    roles = value.get("roles")
    if not isinstance(roles, dict) or set(roles) != {"defender", "attacker"}:
        raise ValueError("calibration dataset must contain defender and attacker")
    for role, role_data in roles.items():
        samples = role_data.get("samples") if isinstance(role_data, dict) else None
        phases = role_data.get("phases") if isinstance(role_data, dict) else None
        events = role_data.get("events") if isinstance(role_data, dict) else None
        if not isinstance(samples, list) or len(samples) < 20:
            raise ValueError(f"{role} requires at least 20 samples")
        if not isinstance(phases, list) or not isinstance(events, list):
            raise ValueError(f"{role} phases and events must be lists")
        categories = {str(phase.get("category")) for phase in phases if isinstance(phase, dict)}
        missing = REQUIRED_CATEGORIES - categories
        if missing:
            raise ValueError(f"{role} calibration plan is missing categories: {sorted(missing)}")
        times = [float(sample["time_s"]) for sample in samples]
        if not all(math.isfinite(item) for item in times) or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f"{role} sample times must be finite and strictly increasing")
    declared_checksum = value.get("dataset_checksum")
    if not isinstance(declared_checksum, str) or declared_checksum != dataset_checksum(value):
        raise ValueError("calibration dataset checksum mismatch")
    return value


def dataset_checksum(dataset: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(dataset))
    payload.pop("dataset_checksum", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _phase_at(plan: Sequence[CalibrationPhase], time_s: float) -> tuple[CalibrationPhase, float]:
    cursor = 0.0
    for phase in plan:
        if time_s < cursor + phase.duration_s - 1e-12:
            return phase, time_s - cursor
        cursor += phase.duration_s
    return plan[-1], plan[-1].duration_s


def synthetic_calibration_dataset(
    prior_profile: Mapping[str, Any],
    *,
    truth_by_role: Mapping[str, Mapping[str, float]],
    seed: int = 4107,
    sample_dt_s: float = 0.005,
) -> dict[str, Any]:
    """Generate exact deterministic calibration data with known parameters."""

    prior = validate_profile(prior_profile)
    dataset: dict[str, Any] = {
        "schema_version": CALIBRATION_DATA_VERSION,
        "source_profile": {
            key: prior[key]
            for key in ("schema_version", "profile_id", "provenance", "calibration_state", "seed", "checksum")
        },
        "seed": int(seed),
        "sampling": {"source": "synthetic", "sample_dt_s": float(sample_dt_s)},
        "roles": {},
        "graceful_shutdown": {"zero_command_sent": True, "actual_wheel_stopped": True},
    }
    for role_index, role in enumerate(("defender", "attacker")):
        truth = dict(truth_by_role[role])
        base_config = ActuatorConfig.from_robot_profile(prior["robots"][role])
        config_values = dict(base_config.__dict__)
        for key in (
            "left_gain",
            "right_gain",
            "left_deadband_mps",
            "right_deadband_mps",
            "time_constant_s",
            "acceleration_mps2",
            "braking_mps2",
            "fixed_delay_s",
            "wheel_radius_m",
            "wheel_separation_m",
        ):
            if key in truth:
                config_values[key] = float(truth[key])
        config = ActuatorConfig(**config_values)
        plan = calibration_plan(prior["robots"][role])
        actuator = DeterministicActuator(config, seed=seed + 1009 * role_index)
        total = sum(phase.duration_s for phase in plan)
        samples: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        pose_x = pose_y = pose_yaw = 0.0
        publish_period = 0.05
        next_publish = 0.0
        steps = int(math.ceil(total / sample_dt_s)) + 1
        for index in range(steps):
            now = min(total, index * sample_dt_s)
            phase, phase_elapsed = _phase_at(plan, min(now, max(total - 1e-12, 0.0)))
            linear, angular = phase.command(phase_elapsed)
            if now + 1e-12 >= next_publish:
                if phase.publish_commands:
                    received = actuator.enqueue(now, linear, angular)
                    received["phase"] = phase.name
                    if received["dropped"]:
                        events.append(received)
                next_publish += publish_period
            state = actuator.step(now, sample_dt_s)
            for event in state["executed_events"]:
                event["phase"] = phase.name
                events.append(event)
            left, right = (float(item) for item in state["actual_wheel_radps"])
            forward = config.wheel_radius_m * 0.5 * (left + right)
            yaw_rate = config.wheel_radius_m * (right - left) / config.wheel_separation_m
            pose_x += forward * math.cos(pose_yaw) * sample_dt_s
            pose_y += forward * math.sin(pose_yaw) * sample_dt_s
            pose_yaw += yaw_rate * sample_dt_s
            samples.append(
                {
                    "time_s": now,
                    "phase": phase.name,
                    "phase_category": phase.category,
                    "phase_elapsed_s": phase_elapsed,
                    "command_published": phase.publish_commands,
                    "command": {"linear_mps": linear, "angular_radps": angular},
                    "pose": {"x": pose_x, "y": pose_y, "z": 0.0647, "yaw": pose_yaw},
                    "joint_target_radps": list(state["target_wheel_radps"]),
                    "joint_actual_radps": list(state["actual_wheel_radps"]),
                    "actuator": {key: value for key, value in state.items() if key != "executed_events"},
                    "collision": False,
                }
            )
            if now >= total:
                break
        dataset["roles"][role] = {
            "phases": [asdict(phase) for phase in plan],
            "samples": samples,
            "events": events,
            "known_truth": truth,
        }
    dataset["dataset_checksum"] = dataset_checksum(dataset)
    return validate_calibration_dataset(dataset)


def _body_command_observations(
    samples: Sequence[Mapping[str, Any]],
    wheel_radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times, requested, observed = [], [], []
    for sample in samples:
        actuator = sample.get("actuator", {})
        values = (actuator.get("requested_linear_mps"), actuator.get("requested_angular_radps"))
        wheels = sample.get("joint_actual_radps")
        if (
            not all(value is not None and math.isfinite(float(value)) for value in values)
            or not isinstance(wheels, (list, tuple))
            or len(wheels) != 2
            or not all(math.isfinite(float(value)) for value in wheels)
        ):
            continue
        linear, angular = (float(value) for value in values)
        times.append(float(sample["time_s"]))
        requested.append((linear, angular))
        observed.append((float(wheels[0]) * wheel_radius, float(wheels[1]) * wheel_radius))
    return np.asarray(times), np.asarray(requested), np.asarray(observed)


def _simulate_sides(
    times: np.ndarray,
    requested: np.ndarray,
    parameters: Sequence[float],
    physics_step_s: float,
    wheel_separation_m: float,
) -> np.ndarray:
    left_gain, right_gain, left_deadband, right_deadband, tau, acceleration, braking = parameters
    body_lag = np.zeros(2, dtype=np.float64)
    body_linear = 0.0
    predicted = np.zeros_like(requested, dtype=np.float64)
    previous = float(times[0]) if len(times) else 0.0
    for index, (now, sides) in enumerate(zip(times, requested)):
        duration = max(0.0, float(now) - previous)
        substeps = max(1, int(round(duration / max(physics_step_s, 1e-9))))
        dt = duration / substeps if duration > 0.0 else physics_step_s
        for _ in range(substeps):
            alpha = 1.0 if tau <= 1e-12 else 1.0 - math.exp(-dt / tau)
            body_lag += alpha * (sides - body_lag)
            accelerating = body_linear * body_lag[0] >= 0.0 and abs(body_lag[0]) > abs(body_linear)
            limit = acceleration if accelerating else braking
            body_linear += float(np.clip(body_lag[0] - body_linear, -limit * dt, limit * dt))
        side_request = np.asarray(
            [
                body_linear - 0.5 * wheel_separation_m * body_lag[1],
                body_linear + 0.5 * wheel_separation_m * body_lag[1],
            ],
            dtype=np.float64,
        )
        predicted[index] = (
            math.copysign(max(0.0, abs(side_request[0]) - left_deadband) * left_gain, side_request[0])
            if side_request[0]
            else 0.0,
            math.copysign(max(0.0, abs(side_request[1]) - right_deadband) * right_gain, side_request[1])
            if side_request[1]
            else 0.0,
        )
        previous = float(now)
    return predicted


def _split_indices(count: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(count)
    train_end = max(1, int(math.floor(0.60 * count)))
    validation_end = max(train_end + 1, int(math.floor(0.80 * count)))
    validation_end = min(validation_end, count - 1)
    return {
        "train": np.sort(order[:train_end]),
        "validation": np.sort(order[train_end:validation_end]),
        "test": np.sort(order[validation_end:]),
    }


def _fit_wheel_geometry(samples: Sequence[Mapping[str, Any]], train_sample_indices: set[int]) -> tuple[float, float]:
    average_wheel, wheel_difference, forward_velocity, yaw_velocity = [], [], [], []
    for index in range(1, len(samples)):
        if index not in train_sample_indices or index - 1 not in train_sample_indices:
            continue
        previous, current = samples[index - 1], samples[index]
        dt = float(current["time_s"]) - float(previous["time_s"])
        if dt <= 1e-9:
            continue
        wheels_a = np.asarray(previous["joint_actual_radps"], dtype=np.float64)
        wheels_b = np.asarray(current["joint_actual_radps"], dtype=np.float64)
        wheels = 0.5 * (wheels_a + wheels_b)
        pose_a, pose_b = previous["pose"], current["pose"]
        yaw_a, yaw_b = float(pose_a["yaw"]), float(pose_b["yaw"])
        yaw_delta = (yaw_b - yaw_a + math.pi) % (2.0 * math.pi) - math.pi
        dx, dy = float(pose_b["x"]) - float(pose_a["x"]), float(pose_b["y"]) - float(pose_a["y"])
        heading = yaw_a + 0.5 * yaw_delta
        forward = (dx * math.cos(heading) + dy * math.sin(heading)) / dt
        average_wheel.append(0.5 * float(wheels[0] + wheels[1]))
        wheel_difference.append(float(wheels[1] - wheels[0]))
        forward_velocity.append(forward)
        yaw_velocity.append(yaw_delta / dt)
    average = np.asarray(average_wheel)
    difference = np.asarray(wheel_difference)
    forward = np.asarray(forward_velocity)
    yaw = np.asarray(yaw_velocity)
    linear_mask = np.abs(average) > 0.05
    angular_mask = np.abs(difference) > 0.05
    if np.count_nonzero(linear_mask) < 10 or np.count_nonzero(angular_mask) < 10:
        raise ValueError("insufficient wheel excitation for effective geometry fit")
    radius = float(np.dot(average[linear_mask], forward[linear_mask]) / np.dot(average[linear_mask], average[linear_mask]))
    radius_over_track = float(
        np.dot(difference[angular_mask], yaw[angular_mask])
        / np.dot(difference[angular_mask], difference[angular_mask])
    )
    if radius <= 0.0 or radius_over_track <= 0.0:
        raise ValueError("effective wheel geometry fit is non-physical")
    return radius, radius / radius_over_track


def fit_calibration_dataset(
    dataset: Mapping[str, Any],
    prior_profile: Mapping[str, Any],
    *,
    split_seed: int = 90210,
) -> dict[str, Any]:
    """Fit both robots and report deterministic 60/20/20 holdout metrics."""

    from scipy.optimize import least_squares

    data = validate_calibration_dataset(dataset)
    prior = validate_profile(prior_profile)
    if data["source_profile"]["checksum"] != prior["checksum"]:
        raise ValueError("dataset source checksum does not match prior profile")
    physics_step = float(prior["simulation"]["physics_step_s"])
    fitted: dict[str, Any] = {}
    for role_index, role in enumerate(("defender", "attacker")):
        samples = data["roles"][role]["samples"]
        robot = prior["robots"][role]
        split = _split_indices(len(samples), split_seed + role_index)
        radius, separation = _fit_wheel_geometry(samples, set(int(item) for item in split["train"]))
        times, requested, observed = _body_command_observations(samples, radius)
        if len(times) != len(samples):
            raise ValueError(f"{role} has incomplete actual wheel observations")
        initial = np.asarray(
            [
                robot["actuator"]["left_gain"],
                robot["actuator"]["right_gain"],
                robot["actuator"]["left_deadband_mps"],
                robot["actuator"]["right_deadband_mps"],
                robot["actuator"]["time_constant_s"],
                robot["actuator"]["acceleration_mps2"],
                robot["actuator"]["braking_mps2"],
            ],
            dtype=np.float64,
        )
        train = split["train"]

        def residual(parameters: np.ndarray) -> np.ndarray:
            predicted = _simulate_sides(times, requested, parameters, physics_step, separation)
            return ((predicted[train] - observed[train]) / 0.05).reshape(-1)

        optimization = least_squares(
            residual,
            initial,
            bounds=(
                np.asarray([0.5, 0.5, 0.0, 0.0, 0.005, 0.01, 0.15]),
                np.asarray([1.5, 1.5, 0.03, 0.03, 0.50, 5.0, 8.0]),
            ),
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=300,
        )
        if not optimization.success:
            raise ValueError(f"{role} actuator fit did not converge: {optimization.message}")
        prediction = _simulate_sides(times, requested, optimization.x, physics_step, separation)
        metrics = {}
        for split_name, indices in split.items():
            error = prediction[indices] - observed[indices]
            metrics[split_name] = {
                "samples": int(len(indices)),
                "rmse_mps": float(np.sqrt(np.mean(np.square(error)))),
                "mae_mps": float(np.mean(np.abs(error))),
            }
        delay_values = []
        for event in data["roles"][role]["events"]:
            if event.get("execute_time_s") is None or event.get("dropped"):
                continue
            scheduled = float(event.get("scheduled_time_s", event["execute_time_s"]))
            receive = float(event["receive_time_s"])
            jitter = float(event.get("jitter_s", 0.0))
            delay_values.append(max(0.0, scheduled - receive - jitter))
        if not delay_values:
            raise ValueError(f"{role} contains no executed actuator events")
        values = optimization.x
        fitted[role] = {
            "parameters": {
                "left_gain": float(values[0]),
                "right_gain": float(values[1]),
                "left_deadband_mps": float(values[2]),
                "right_deadband_mps": float(values[3]),
                "time_constant_s": float(values[4]),
                "acceleration_mps2": float(values[5]),
                "braking_mps2": float(values[6]),
                "wheel_radius_m": radius,
                "wheel_separation_m": separation,
                "fixed_delay_s": float(np.median(delay_values)),
            },
            "split": {"method": "deterministic_random_samples", "ratios": [0.60, 0.20, 0.20], "seed": split_seed + role_index},
            "metrics": metrics,
            "optimizer": {
                "success": bool(optimization.success),
                "status": int(optimization.status),
                "cost": float(optimization.cost),
                "nfev": int(optimization.nfev),
                "message": str(optimization.message),
            },
        }
    return {
        "schema_version": "trackmaker.digital_twin_fit.v2.1",
        "dataset_checksum": data.get("dataset_checksum", dataset_checksum(data)),
        "source_profile_checksum": prior["checksum"],
        "split": "60/20/20",
        "roles": fitted,
    }


def measured_profile_from_fit(
    prior_profile: Mapping[str, Any],
    fit: Mapping[str, Any],
    *,
    profile_id: str,
    calibration_seed: int,
) -> dict[str, Any]:
    """Create a separate measured profile; the input prior is never mutated."""

    prior = validate_profile(prior_profile)
    measured = copy.deepcopy(prior)
    measured["profile_id"] = str(profile_id)
    measured["provenance"] = "measured"
    measured["calibration_state"] = "calibrated"
    measured["seed"] = int(calibration_seed)
    measured["source"] = {
        "kind": "offline_ros2_calibration_fit",
        "urdf": prior["source"].get("urdf"),
        "urdf_sha256": prior["source"].get("urdf_sha256"),
        "prior_profile_id": prior["profile_id"],
        "prior_profile_checksum": prior["checksum"],
        "dataset_checksum": fit["dataset_checksum"],
        "fitter": "coppelia_env.digital_twin_calibration.fit_calibration_dataset",
        "split": "60/20/20",
    }
    for role in ("defender", "attacker"):
        parameters = fit["roles"][role]["parameters"]
        robot = measured["robots"][role]
        for key in (
            "left_gain",
            "right_gain",
            "left_deadband_mps",
            "right_deadband_mps",
            "time_constant_s",
            "acceleration_mps2",
            "braking_mps2",
            "fixed_delay_s",
        ):
            robot["actuator"][key] = float(parameters[key])
        radius = float(parameters["wheel_radius_m"])
        separation = float(parameters["wheel_separation_m"])
        robot["wheel_radius_m"] = radius
        robot["wheel_separation_m"] = separation
        geometry = robot["geometry"]
        geometry["left_wheel"]["radius_m"] = radius
        geometry["right_wheel"]["radius_m"] = radius
        geometry["left_wheel"]["center_m"][1] = 0.5 * separation
        geometry["right_wheel"]["center_m"][1] = -0.5 * separation
        wheel_z = float(geometry["left_wheel"]["center_m"][2])
        geometry["nominal_base_z_m"] = 0.02 - (wheel_z - radius)
        base = geometry["base"]
        geometry["base_clearance_m"] = (
            geometry["nominal_base_z_m"]
            + float(base["center_m"][2])
            - 0.5 * float(base["height_m"])
            - 0.02
        )
    measured["checksum"] = profile_checksum(measured)
    return validate_profile(measured)


def relative_error(actual: float, expected: float) -> float:
    return abs(float(actual) - float(expected)) / max(abs(float(expected)), 1e-12)
