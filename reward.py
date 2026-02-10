"""
奖励函数模块
"""

import math
import map_config


def apply_timeout_defender_win(info=None):
    """统一超时结算：超时按 defender 胜利处理，奖励值不在这里改动。"""
    timeout_info = dict(info) if info is not None else {}
    timeout_info['reason'] = 'timeout_defender_wins'
    timeout_info['win'] = True
    return timeout_info


def reward_calculate_tad(defender, attacker, target, prev_defender=None, prev_attacker=None,
                         defender_collision=False, attacker_collision=False,
                         defender_captured=False, attacker_captured=False,
                         capture_progress_defender=0, capture_progress_attacker=0,
                         capture_required_steps=0, radar=None, initial_dist_def_tgt=None):
    """TAD 标准奖励函数"""
    info = {
        'capture_progress_defender': int(capture_progress_defender),
        'capture_progress_attacker': int(capture_progress_attacker),
        'capture_required_steps': int(capture_required_steps),
        'defender_collision': bool(defender_collision),
        'attacker_collision': bool(attacker_collision)
    }

    reward = 0.0
    terminated = False

    reward -= 0.04

    success_reward = float(getattr(map_config, 'success_reward', 20.0))

    if defender_captured:
        terminated = True
        info['reason'] = 'defender_caught_attacker'
        info['win'] = True
        reward += success_reward
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


def reward_calculate_chase(defender, attacker, target, prev_defender=None, prev_attacker=None,
                           defender_collision=False, attacker_collision=False,
                           defender_captured=False, attacker_captured=False,
                           capture_progress_defender=0, capture_progress_attacker=0,
                           capture_required_steps=0, radar=None, initial_dist_def_att=None):
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

    # 时间惩罚：每步-0.04
    reward -= 0.04

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
        reward += 10.0
    elif attacker_captured:
        # chase 模式下 attacker 捕获 target：只终止并判负，不额外惩罚
        terminated = True
        info['reason'] = 'attacker_caught_target'
        info['win'] = False
    elif defender_collision:
        terminated = True
        reward -= 10.0
        info['reason'] = 'defender_collision'
        info['win'] = False

    return float(reward), bool(terminated), False, info


def reward_calculate_protect1(defender, attacker, target, prev_defender=None, prev_attacker=None,
                              defender_collision=False, attacker_collision=False,
                              defender_captured=False, attacker_captured=False,
                              capture_progress_defender=0, capture_progress_attacker=0,
                              capture_required_steps=0, radar=None, initial_dist_def_tgt=None,
                              collision_cooldown=0):
    """Protect阶段1奖励函数：学会导航到target"""
    info = {
        'capture_progress_defender': int(capture_progress_defender),
        'capture_progress_attacker': int(capture_progress_attacker),
        'capture_required_steps': int(capture_required_steps),
        'defender_collision': bool(defender_collision),
        'attacker_collision': bool(attacker_collision)
    }

    reward = 0.0
    terminated = False

    # 计算defender到target的距离
    dx_def_tgt = (defender['x'] + map_config.pixel_size * 0.5) - (target['x'] + map_config.pixel_size * 0.5)
    dy_def_tgt = (defender['y'] + map_config.pixel_size * 0.5) - (target['y'] + map_config.pixel_size * 0.5)
    curr_dist_def_tgt = math.hypot(dx_def_tgt, dy_def_tgt)

    success_reward = 20
    target_radius = float(getattr(map_config, 'target_radius', 16.0))
    agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
    reach_radius = target_radius + agent_radius

    # 距离奖励：按到target边缘(reach_radius)的进度比例给奖励
    # initial_dist_def_tgt 由 env.reset 提供，已是初始边缘距离
    if prev_defender is not None and initial_dist_def_tgt is not None and initial_dist_def_tgt > 0:
        prev_dx_def_tgt = (prev_defender['x'] + map_config.pixel_size * 0.5) - (target['x'] + map_config.pixel_size * 0.5)
        prev_dy_def_tgt = (prev_defender['y'] + map_config.pixel_size * 0.5) - (target['y'] + map_config.pixel_size * 0.5)
        prev_dist_def_tgt = math.hypot(prev_dx_def_tgt, prev_dy_def_tgt)

        prev_boundary_dist = max(0.0, prev_dist_def_tgt - reach_radius)
        curr_boundary_dist = max(0.0, curr_dist_def_tgt - reach_radius)
        initial_boundary_dist = max(0.0, initial_dist_def_tgt)

        if initial_boundary_dist > 0:
            distance_progress = (prev_boundary_dist - curr_boundary_dist) / initial_boundary_dist
            distance_reward = distance_progress * success_reward
            reward += distance_reward

    # 终止条件判断
    if curr_dist_def_tgt <= reach_radius:
        terminated = True
        info['reason'] = 'defender_reached_target'
        info['win'] = True
        reward += 0.2 * success_reward

    # 碰撞惩罚：不终止episode，惩罚-5，10步冷却
    if defender_collision and collision_cooldown == 0:
        reward -= 5.0
        info['collision_penalty_applied'] = True

    return float(reward), bool(terminated), False, info


def reward_calculate_protect2(defender, attacker, target, prev_defender=None, prev_attacker=None,
                              defender_collision=False, attacker_collision=False,
                              defender_captured=False, attacker_captured=False,
                              capture_progress_defender=0, capture_progress_attacker=0,
                              capture_required_steps=0, radar=None, initial_dist_def_tgt=None):
    """Protect阶段2奖励函数：到达target后保护它"""
    info = {
        'capture_progress_defender': int(capture_progress_defender),
        'capture_progress_attacker': int(capture_progress_attacker),
        'capture_required_steps': int(capture_required_steps),
        'defender_collision': bool(defender_collision),
        'attacker_collision': bool(attacker_collision)
    }

    reward = 0.0
    terminated = False

    # 计算defender到target的距离
    dx_def_tgt = (defender['x'] + map_config.pixel_size * 0.5) - (target['x'] + map_config.pixel_size * 0.5)
    dy_def_tgt = (defender['y'] + map_config.pixel_size * 0.5) - (target['y'] + map_config.pixel_size * 0.5)
    curr_dist_def_tgt = math.hypot(dx_def_tgt, dy_def_tgt)

    success_reward = float(getattr(map_config, 'success_reward', 20.0))
    target_radius = float(getattr(map_config, 'target_radius', 16.0))
    agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
    reach_radius = target_radius + agent_radius

    # 距离奖励：按到target边缘(reach_radius)的进度比例给奖励
    # initial_dist_def_tgt 由 env.reset 提供，已是初始边缘距离
    if prev_defender is not None and initial_dist_def_tgt is not None and initial_dist_def_tgt > 0:
        prev_dx_def_tgt = (prev_defender['x'] + map_config.pixel_size * 0.5) - (target['x'] + map_config.pixel_size * 0.5)
        prev_dy_def_tgt = (prev_defender['y'] + map_config.pixel_size * 0.5) - (target['y'] + map_config.pixel_size * 0.5)
        prev_dist_def_tgt = math.hypot(prev_dx_def_tgt, prev_dy_def_tgt)

        prev_boundary_dist = max(0.0, prev_dist_def_tgt - reach_radius)
        curr_boundary_dist = max(0.0, curr_dist_def_tgt - reach_radius)
        initial_boundary_dist = max(0.0, initial_dist_def_tgt)

        if initial_boundary_dist > 0:
            distance_progress = (prev_boundary_dist - curr_boundary_dist) / initial_boundary_dist
            distance_reward = distance_progress * success_reward
            reward += distance_reward

    # 终止条件判断
    if defender_captured:
        terminated = True
        info['reason'] = 'defender_caught_attacker'
        info['win'] = True
        reward += 0.5 * success_reward
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
