"""
奖励函数模块
"""

import math
import map_config


OBSTACLE_SHAPING_OMEGA = 1.5
OBSTACLE_SHAPING_K = 12.0


def _radar_min_distance_norm(radar):
    """将雷达最小读数[-1,1]转换为归一化距离[0,1]。"""
    if radar is None:
        return None
    try:
        min_reading = float(min(radar))
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, 0.5 * (min_reading + 1.0)))


def _obstacle_proximity_delta_reward(radar, prev_radar, omega=OBSTACLE_SHAPING_OMEGA, k=OBSTACLE_SHAPING_K):
    """
    基于最近障碍距离差分的非线性避障奖励（非线性，非反函数）。

    公式:
      d_t = clip((min(radar_t)+1)/2, 0, 1)
      phi(d) = log(1+k*d) / log(1+k)
      r_t = omega * (phi(d_t) - phi(d_{t-1}))

    - 距离不变时奖励约为0（近似零和）
    - 靠近障碍为负，远离障碍为正
    - d越小越敏感
    """
    d_curr = _radar_min_distance_norm(radar)
    d_prev = _radar_min_distance_norm(prev_radar)
    if d_curr is None or d_prev is None:
        return 0.0

    kk = max(float(k), 1e-6)
    denom = math.log1p(kk)
    phi_curr = math.log1p(kk * d_curr) / denom
    phi_prev = math.log1p(kk * d_prev) / denom
    return float(omega) * (phi_curr - phi_prev)


def apply_timeout_defender_win(info=None):
    """统一超时结算：超时按 defender 胜利处理，奖励值不在这里改动。"""
    timeout_info = dict(info) if info is not None else {}
    timeout_info['reason'] = 'timeout_defender_wins'
    timeout_info['win'] = True
    return timeout_info


def _guidance_reward_to_attacker(defender, attacker, prev_defender, prev_attacker, initial_dist_def_att):
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
    return float(distance_progress * 10.0)


def _guidance_reward_to_target(defender, target, prev_defender, initial_dist_def_tgt):
    """Protect2引导奖励: 鼓励defender持续接近target。"""
    if prev_defender is None or initial_dist_def_tgt is None or initial_dist_def_tgt <= 0.0:
        return 0.0

    target_radius = float(getattr(map_config, 'target_radius', 16.0))
    agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
    reach_radius = target_radius + agent_radius
    success_reward = float(getattr(map_config, 'success_reward', 20.0))

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
    return float(distance_progress * success_reward)


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

    reward = -0.0
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


def reward_calculate_baseline(defender, attacker, target, prev_defender=None, prev_attacker=None,
                              defender_collision=False, attacker_collision=False,
                              defender_captured=False, attacker_captured=False,
                              capture_progress_defender=0, capture_progress_attacker=0,
                              capture_required_steps=0, radar=None,
                              initial_dist_def_tgt=None, initial_dist_def_att=None):
    """
    Baseline端到端奖励:
      TAD现有奖励 + 时间惩罚(-0.04) + 引导奖励(到attacker + 到target)
    """
    reward, terminated, truncated, info = reward_calculate_tad(
        defender=defender,
        attacker=attacker,
        target=target,
        prev_defender=prev_defender,
        prev_attacker=prev_attacker,
        defender_collision=defender_collision,
        attacker_collision=attacker_collision,
        defender_captured=defender_captured,
        attacker_captured=attacker_captured,
        capture_progress_defender=capture_progress_defender,
        capture_progress_attacker=capture_progress_attacker,
        capture_required_steps=capture_required_steps,
        radar=radar,
        initial_dist_def_tgt=initial_dist_def_tgt,
    )

    reward -= 0.04
    reward += _guidance_reward_to_attacker(
        defender=defender,
        attacker=attacker,
        prev_defender=prev_defender,
        prev_attacker=prev_attacker,
        initial_dist_def_att=initial_dist_def_att,
    )
    reward += _guidance_reward_to_target(
        defender=defender,
        target=target,
        prev_defender=prev_defender,
        initial_dist_def_tgt=initial_dist_def_tgt,
    )

    return float(reward), bool(terminated), bool(truncated), info


def reward_calculate_chase(defender, attacker, target, prev_defender=None, prev_attacker=None,
                           defender_collision=False, attacker_collision=False,
                           defender_captured=False, attacker_captured=False,
                           capture_progress_defender=0, capture_progress_attacker=0,
                           capture_required_steps=0, radar=None, prev_radar=None, initial_dist_def_att=None):
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

    # 时间惩罚：
    reward -= 0.08

    # 暂时关闭: 避障差分奖励（每步）
    # reward += _obstacle_proximity_delta_reward(radar, prev_radar)

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
    # elif attacker_captured:
    #     # chase 模式下 attacker 捕获 target：只终止并判负，不额外惩罚
    #     terminated = True
    #     info['reason'] = 'attacker_caught_target'
    #     info['win'] = False
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
                              capture_required_steps=0, radar=None, prev_radar=None, initial_dist_def_tgt=None):
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

    # 暂时关闭: 避障差分奖励（每步）
    # reward += _obstacle_proximity_delta_reward(radar, prev_radar)

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
