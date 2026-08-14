from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.rewards import reward_calculate_chase, reward_calculate_tad  # noqa: E402
from eval.run_defender_hierarchy_attacker_matrix import (  # noqa: E402
    DEFENDER_VARIANTS,
    protect_chase_union_analysis,
    resolved_defender_variants,
)


def test_chase_rollout_ignores_target_contact_but_formal_tad_ends():
    actor = {"x": 0.0, "y": 0.0}
    target = {"x": 100.0, "y": 100.0}
    chase_reward, chase_terminated, _, chase_info = reward_calculate_chase(
        actor, actor, target, attacker_captured=True
    )
    tad_reward, tad_terminated, _, tad_info = reward_calculate_tad(
        actor, actor, target, attacker_captured=True
    )

    assert not chase_terminated
    assert chase_info.get("reason") is None
    assert tad_terminated
    assert tad_info["reason"] == "attacker_caught_target"
    assert tad_reward < chase_reward


def test_chase_checkpoint_override_is_local_to_standalone_chase():
    variants = resolved_defender_variants("/tmp/new_chase.pth")
    assert variants["chase_skill"].chase_checkpoint == "/tmp/new_chase.pth"
    assert variants["protect_skill"] == DEFENDER_VARIANTS["protect_skill"]
    assert variants["hrl"] == DEFENDER_VARIANTS["hrl"]


def test_union_analysis_marks_high_margin_common_failure():
    base = {
        "attacker": "heuristic_default",
        "seed": 7,
        "initial_target_time_margin": 25.0,
        "defender_success": False,
        "defender_collision": False,
        "reason": "attacker_caught_target",
    }
    summaries, pairs = protect_chase_union_analysis([
        {**base, "defender": "protect_skill"},
        {**base, "defender": "chase_skill"},
    ])
    assert pairs[0]["failure_class"] == "high_margin_strategy_failure"
    assert summaries[0]["high_margin_collision_free_common_failure_count"] == 1


if __name__ == "__main__":
    test_chase_rollout_ignores_target_contact_but_formal_tad_ends()
    test_chase_checkpoint_override_is_local_to_standalone_chase()
    test_union_analysis_marks_high_margin_common_failure()
