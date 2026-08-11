"""
奖励函数模块
"""

import math
from configs import map_config


PROTECT_STEP_TIME_PENALTY = -0.04


def apply_timeout_defender_win(info=None):
    """统一超时结算：超时按 defender 胜利处理，奖励值不在这里改动。"""
    timeout_info = dict(info) if info is not None else {}
    timeout_info['reason'] = 'timeout_defender_wins'
    timeout_info['win'] = True
    return timeout_info


def _guidance_reward_to_attacker(
    defender,
    attacker,
    prev_defender,
    prev_attacker,
    initial_dist_def_att,
    reward_scale=10.0,
):
    """Chase引导奖励: 鼓励defender持续接近attacker。"""
    if prev_defender is None or prev_attacker is None or initial_dist_def_att is None:
        return 0.0

    agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
    capture_radius = agent_radius * 2.0
    if initial_dist_def_att <= capture_radius:
        return 0.0

    curr_dx = (defender['x'] + map_config.pixel_size * 0.5) - (attacker['x'] + map_config.pixel_size * 0.5)
    curr_dy = (defender['y'] + map_config.pixel_size * 0.5) - (attacker['y'] + map_config.pixel_size * 0.5)
    curr_dist = math.hypot(curr_dx, curr_dy)

    prev_dx = (prev_defender['x'] + map_config.pixel_size * 0.5) - (prev_attacker['x'] + map_config.pixel_size * 0.5)
    prev_dy = (prev_defender['y'] + map_config.pixel_size * 0.5) - (prev_attacker['y'] + map_config.pixel_size * 0.5)
    prev_dist = math.hypot(prev_dx, prev_dy)

    prev_boundary_dist = max(0.0, prev_dist - capture_radius)
    curr_boundary_dist = max(0.0, curr_dist - capture_radius)
    initial_boundary_dist = max(0.0, initial_dist_def_att - capture_radius)
    if initial_boundary_dist <= 0.0:
        return 0.0

    distance_progress = (prev_boundary_dist - curr_boundary_dist) / initial_boundary_dist
    return float(distance_progress * float(reward_scale))


def _guidance_reward_to_target(
    defender,
    target,
    prev_defender,
    initial_dist_def_tgt,
    reward_scale=20.0,
):
    """Protect引导奖励: 鼓励defender持续接近target。"""
    if prev_defender is None or initial_dist_def_tgt is None or initial_dist_def_tgt <= 0.0:
        return 0.0

    target_radius = float(getattr(map_config, 'target_radius', 16.0))
    agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
    reach_radius = target_radius + agent_radius
    curr_dx = (defender['x'] + map_config.pixel_size * 0.5) - (target['x'] + map_config.pixel_size * 0.5)
    curr_dy = (defender['y'] + map_config.pixel_size * 0.5) - (target['y'] + map_config.pixel_size * 0.5)
    curr_dist = math.hypot(curr_dx, curr_dy)

    prev_dx = (prev_defender['x'] + map_config.pixel_size * 0.5) - (target['x'] + map_config.pixel_size * 0.5)
    prev_dy = (prev_defender['y'] + map_config.pixel_size * 0.5) - (target['y'] + map_config.pixel_size * 0.5)
    prev_dist = math.hypot(prev_dx, prev_dy)

    prev_boundary_dist = max(0.0, prev_dist - reach_radius)
    curr_boundary_dist = max(0.0, curr_dist - reach_radius)
    initial_boundary_dist = max(0.0, initial_dist_def_tgt)
    if initial_boundary_dist <= 0.0:
        return 0.0

    distance_progress = (prev_boundary_dist - curr_boundary_dist) / initial_boundary_dist
    return float(distance_progress * float(reward_scale))


def reward_calculate_tad(defender, attacker, target, prev_defender=None, prev_attacker=None,
                         defender_collision=False, attacker_collision=False,
                         defender_captured=False, attacker_captured=False,
                         capture_progress_defender=0, capture_progress_attacker=0,
                         capture_required_steps=0, radar=None, initial_dist_def_tgt=None,
                         initial_dist_def_att=None):
    """TAD 标准奖励函数"""
    info = {
        'capture_progress_defender': int(capture_progress_defender),
        'capture_progress_attacker': int(capture_progress_attacker),
        'capture_required_steps': int(capture_required_steps),
        'defender_collision': bool(defender_collision),
        'attacker_collision': bool(attacker_collision)
    }

    reward = -0.04
    terminated = False

    success_reward = float(getattr(map_config, 'success_reward', 20.0))

    if defender_captured:
        terminated = True
        info['reason'] = 'defender_caught_attacker'
        info['win'] = True
        reward += 1.5*success_reward
    elif attacker_captured:
        terminated = True
        info['reason'] = 'attacker_caught_target'
        info['win'] = False
        reward -= success_reward
    elif defender_collision:
        terminated = True
        reward -= success_reward
        info['reason'] = 'defender_collision'
        info['win'] = False

    return float(reward), bool(terminated), False, info

def reward_calculate_protect(defender, attacker, target, prev_defender=None, prev_attacker=None,
                             defender_collision=False, attacker_collision=False,
                             defender_captured=False, attacker_captured=False,
                             capture_progress_defender=0, capture_progress_attacker=0,
                             capture_required_steps=0, radar=None,
                             initial_dist_def_tgt=None, initial_dist_def_att=None):
    """Protect奖励：TAD 终端项 + 每步惩罚 + chase/protect dense guidance。"""
    info = {
        'capture_progress_defender': int(capture_progress_defender),
        'capture_progress_attacker': int(capture_progress_attacker),
        'capture_required_steps': int(capture_required_steps),
        'defender_collision': bool(defender_collision),
        'attacker_collision': bool(attacker_collision),
    }

    reward = float(PROTECT_STEP_TIME_PENALTY)
    terminated = False
    success_reward = float(getattr(map_config, 'success_reward', 20.0))

    if defender_captured:
        terminated = True
        info['reason'] = 'defender_caught_attacker'
        info['win'] = True
        reward += 1.5 * success_reward
    elif attacker_captured:
        terminated = True
        info['reason'] = 'attacker_caught_target'
        info['win'] = False
        reward -= success_reward
    elif defender_collision:
        terminated = True
        info['reason'] = 'defender_collision'
        info['win'] = False
        reward -= success_reward

    reward += _guidance_reward_to_attacker(
        defender=defender,
        attacker=attacker,
        prev_defender=prev_defender,
        prev_attacker=prev_attacker,
        initial_dist_def_att=initial_dist_def_att,
        reward_scale=10.0,
    )
    reward += _guidance_reward_to_target(
        defender=defender,
        target=target,
        prev_defender=prev_defender,
        initial_dist_def_tgt=initial_dist_def_tgt,
        reward_scale=20.0,
    )
    return float(reward), bool(terminated), False, info


def reward_calculate_chase(defender, attacker, target, prev_defender=None, prev_attacker=None,
                           defender_collision=False, attacker_collision=False,
                           defender_captured=False, attacker_captured=False,
                           capture_progress_defender=0, capture_progress_attacker=0,
                           capture_required_steps=0, radar=None, prev_radar=None,
                           initial_dist_def_att=None, initial_dist_def_tgt=None):
    """纯追逃奖励函数 - 用于训练 Chase 技能"""
    info = {
        'capture_progress_defender': int(capture_progress_defender),
        'capture_progress_attacker': int(capture_progress_attacker),
        'capture_required_steps': int(capture_required_steps),
        'defender_collision': bool(defender_collision),
        'attacker_collision': bool(attacker_collision)
    }

    reward = 0.0
    terminated = False

    # Chase shaping remains unchanged from the successful legacy run.  The
    # full TAD terminal contract is handled here as well: a Target breach is
    # an immediate Defender loss, rather than a transition that can continue
    # and later be mislabelled as a timeout.
    reward -= 0.08

    # 计算defender到attacker的距离
    dx_def_att = (defender['x'] + map_config.pixel_size * 0.5) - (attacker['x'] + map_config.pixel_size * 0.5)
    dy_def_att = (defender['y'] + map_config.pixel_size * 0.5) - (attacker['y'] + map_config.pixel_size * 0.5)
    curr_dist_def_att = math.hypot(dx_def_att, dy_def_att)

    agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
    capture_radius = agent_radius * 2

    # 微分距离奖励：按进度比例给奖励，总计10.0
    if prev_defender is not None and prev_attacker is not None and initial_dist_def_att is not None and initial_dist_def_att > capture_radius:
        prev_dx_def_att = (prev_defender['x'] + map_config.pixel_size * 0.5) - (prev_attacker['x'] + map_config.pixel_size * 0.5)
        prev_dy_def_att = (prev_defender['y'] + map_config.pixel_size * 0.5) - (prev_attacker['y'] + map_config.pixel_size * 0.5)
        prev_dist_def_att = math.hypot(prev_dx_def_att, prev_dy_def_att)

        prev_boundary_dist = max(0.0, prev_dist_def_att - capture_radius)
        curr_boundary_dist = max(0.0, curr_dist_def_att - capture_radius)
        initial_boundary_dist = max(0.0, initial_dist_def_att - capture_radius)

        if initial_boundary_dist > 0:
            distance_progress = (prev_boundary_dist - curr_boundary_dist) / initial_boundary_dist
            distance_reward = distance_progress * 10.0
            reward += distance_reward

    # 终止奖励
    if defender_captured:
        terminated = True
        info['reason'] = 'defender_caught_attacker'
        info['win'] = True
        reward += 20.0
    elif attacker_captured:
        terminated = True
        info['reason'] = 'attacker_caught_target'
        info['win'] = False
        reward -= float(getattr(map_config, 'success_reward', 20.0))
    elif defender_collision:
        terminated = True
        reward -= 10.0
        info['reason'] = 'defender_collision'
        info['win'] = False

    return float(reward), bool(terminated), False, info
