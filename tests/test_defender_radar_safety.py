from pathlib import Path
import inspect
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.defender_radar_safety import RadarSafetyFilter
from envs.tad_env import TADEnv
from attacker.defender_pool import BaseDefenderController, create_defender_controller


def _obs_with_ranges(default_distance=300.0):
    obs = np.zeros(71, dtype=np.float32)
    obs[7:71] = 2.0 * float(default_distance) / 300.0 - 1.0
    return obs


def test_clear_path_preserves_action_exactly():
    safety = RadarSafetyFilter()
    obs = _obs_with_ranges()
    action = np.asarray([0.23, 0.91], dtype=np.float32)

    applied, diagnostics = safety.filter_action(obs, action, max_turn=6.0, max_speed=2.6)

    assert applied is action
    assert diagnostics["mode"] == "pass"
    assert not diagnostics["intervened"]


def test_obstacle_ahead_changes_only_steering():
    safety = RadarSafetyFilter(horizon_steps=16, turn_samples=25)
    obs = _obs_with_ranges()
    obs[7] = 2.0 * 26.0 / 300.0 - 1.0
    action = np.asarray([0.0, 1.0], dtype=np.float32)

    applied, diagnostics = safety.filter_action(obs, action, max_turn=6.0, max_speed=2.6)

    assert diagnostics["mode"] in {"steer", "unavoidable"}
    assert diagnostics["intervened"]
    assert applied[1] == action[1]
    assert applied[0] != action[0]


def test_unavoidable_case_never_reduces_speed():
    safety = RadarSafetyFilter(safety_margin=2.0)
    obs = _obs_with_ranges(default_distance=9.0)
    action = np.asarray([0.0, 0.8], dtype=np.float32)

    applied, diagnostics = safety.filter_action(obs, action, max_turn=6.0, max_speed=2.6)

    assert diagnostics["mode"] == "unavoidable"
    assert applied[1] == action[1]


def test_legacy_map_based_masks_are_not_exposed():
    assert "hard_action_mask" not in inspect.signature(TADEnv).parameters
    assert not hasattr(TADEnv, "_apply_hard_action_mask")
    assert not hasattr(BaseDefenderController, "_apply_defender_hard_obstacle_mask")
    try:
        create_defender_controller(
            "protect",
            env=object(),
            controller_config={"apply_controller_obstacle_mask": True},
        )
    except ValueError as exc:
        assert "retired" in str(exc)
    else:
        raise AssertionError("legacy controller mask config must fail loudly")


if __name__ == "__main__":
    test_clear_path_preserves_action_exactly()
    test_obstacle_ahead_changes_only_steering()
    test_unavoidable_case_never_reduces_speed()
    test_legacy_map_based_masks_are_not_exposed()
