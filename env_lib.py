import math
import numpy as np
import map_config
import pygame
import pygame.gfxdraw
from numba import njit

_OCC_GRID = None
_OCC_CELL = None
_RECT_OBS = None
_CIRCLE_OBS = None
_SEGMENT_OBS = None
_OBS_COMPILED = False

@njit(fastmath=True)
def _numba_ray_cast_kernel(ox, oy, angle, max_range, grid, cell_size, nx, ny, pad_cells):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    if abs(cos_a) < 1e-9: cos_a = 1e-9 if cos_a >= 0 else -1e-9
    if abs(sin_a) < 1e-9: sin_a = 1e-9 if sin_a >= 0 else -1e-9

    gx = ox / cell_size
    gy = oy / cell_size
    ix = int(gx)
    iy = int(gy)

    if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
        return 0.0

    if pad_cells > 0:
        x1 = max(0, ix - pad_cells)
        x2 = min(nx, ix + pad_cells + 1)
        y1 = max(0, iy - pad_cells)
        y2 = min(ny, iy + pad_cells + 1)
        if np.any(grid[y1:y2, x1:x2]):
            return 0.0
    else:
        if grid[iy, ix]:
            return 0.0

    step_x = 1 if cos_a > 0 else -1
    step_y = 1 if sin_a > 0 else -1

    cell_x = float(ix)
    cell_y = float(iy)

    if step_x > 0:
        dist_to_vx = (cell_x + 1.0 - gx) * cell_size
    else:
        dist_to_vx = (gx - cell_x) * cell_size

    if step_y > 0:
        dist_to_vy = (cell_y + 1.0 - gy) * cell_size
    else:
        dist_to_vy = (gy - cell_y) * cell_size

    tMaxX = dist_to_vx / abs(cos_a)
    tMaxY = dist_to_vy / abs(sin_a)

    tDeltaX = cell_size / abs(cos_a)
    tDeltaY = cell_size / abs(sin_a)

    dist = 0.0

    while dist <= max_range:
        if tMaxX < tMaxY:
            dist = tMaxX
            tMaxX += tDeltaX
            ix += step_x
        else:
            dist = tMaxY
            tMaxY += tDeltaY
            iy += step_y

        if ix < 0 or ix >= nx or iy < 0 or iy >= ny:
            break

        if pad_cells > 0:
            x1 = max(0, ix - pad_cells)
            x2 = min(nx, ix + pad_cells + 1)
            y1 = max(0, iy - pad_cells)
            y2 = min(ny, iy + pad_cells + 1)
            if np.any(grid[y1:y2, x1:x2]):
                return min(dist, max_range)
        else:
            if grid[iy, ix]:
                return min(dist, max_range)

    return max_range

@njit(fastmath=True, cache=True)
def _numba_ray_rect(ox, oy, dx, dy, rx, ry, rw, rh):
    # *** 修复：避免探测到障碍物背面边缘 ***
    # 方法：如果起点在矩形内部或边界上，增加一个较大的排除区
    t_min = 1e30
    eps = 1e-9
    exclude_dist = 1.0  # 排除距离起点1.0以内的交点（增加到1.0像素）

    # 检查起点是否在矩形内部
    inside = rx <= ox <= rx + rw and ry <= oy <= ry + rh

    if abs(dx) > eps:
        # 左边
        t = (rx - ox) / dx
        y = oy + t * dy
        if t >= 0.0 and ry <= y <= ry + rh:
            if inside:
                # 在内部，排除很近的交点（可能是起点所在的边）
                if t > exclude_dist and t < t_min:
                    t_min = t
            else:
                if t < t_min:
                    t_min = t

        # 右边
        t = (rx + rw - ox) / dx
        y = oy + t * dy
        if t >= 0.0 and ry <= y <= ry + rh:
            if inside:
                if t > exclude_dist and t < t_min:
                    t_min = t
            else:
                if t < t_min:
                    t_min = t

    if abs(dy) > eps:
        # 上边
        t = (ry - oy) / dy
        x = ox + t * dx
        if t >= 0.0 and rx <= x <= rx + rw:
            if inside:
                if t > exclude_dist and t < t_min:
                    t_min = t
            else:
                if t < t_min:
                    t_min = t

        # 下边
        t = (ry + rh - oy) / dy
        x = ox + t * dx
        if t >= 0.0 and rx <= x <= rx + rw:
            if inside:
                if t > exclude_dist and t < t_min:
                    t_min = t
            else:
                if t < t_min:
                    t_min = t

    return t_min

@njit(fastmath=True, cache=True)
def _numba_ray_circle(ox, oy, dx, dy, cx, cy, r):
    fx = ox - cx
    fy = oy - cy
    b = 2.0 * (dx * fx + dy * fy)
    c = (fx * fx + fy * fy) - r * r
    disc = b * b - 4.0 * c

    if disc < 0.0:
        return 1e30

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0

    if t2 < 0.0:
        return 1e30
    if t1 > 0.0:
        return t1
    return t2

@njit(fastmath=True, cache=True)
def _numba_ray_segment_line(ox, oy, dx, dy, x1, y1, x2, y2):
    rx = x2 - x1
    ry = y2 - y1
    cross = dx * ry - dy * rx

    if abs(cross) < 1e-9:
        return 1e30

    dp_x = x1 - ox
    dp_y = y1 - oy

    t = (dp_x * ry - dp_y * rx) / cross
    u = (dp_x * dy - dp_y * dx) / cross

    if 0.0 <= u <= 1.0 and t >= 0.0:
        return t
    return 1e30

@njit(fastmath=True, cache=True)
def _numba_ray_capsule(ox, oy, dx, dy, p1x, p1y, p2x, p2y, r):
    dist = _numba_ray_circle(ox, oy, dx, dy, p1x, p1y, r)
    dist = min(dist, _numba_ray_circle(ox, oy, dx, dy, p2x, p2y, r))

    vx = p2x - p1x
    vy = p2y - p1y
    length = math.sqrt(vx * vx + vy * vy)

    if length < 1e-9:
        return dist

    nx = -vy / length
    ny = vx / length

    off_x = nx * r
    off_y = ny * r

    t1 = _numba_ray_segment_line(ox, oy, dx, dy, p1x + off_x, p1y + off_y, p2x + off_x, p2y + off_y)
    t2 = _numba_ray_segment_line(ox, oy, dx, dy, p1x - off_x, p1y - off_y, p2x - off_x, p2y - off_y)

    return min(dist, t1, t2)

@njit(fastmath=True, cache=True)
def _numba_ray_all_obstacles(ox, oy, angle, max_range, rects, circles, segments, padding):
    dx = math.cos(angle)
    dy = math.sin(angle)
    min_dist = max_range

    for i in range(rects.shape[0]):
        rx, ry, rw, rh = rects[i, 0] - padding, rects[i, 1] - padding, rects[i, 2] + 2.0 * padding, rects[i, 3] + 2.0 * padding
        d = _numba_ray_rect(ox, oy, dx, dy, rx, ry, rw, rh)
        if d < min_dist: min_dist = d

    for i in range(circles.shape[0]):
        d = _numba_ray_circle(ox, oy, dx, dy, circles[i, 0], circles[i, 1], circles[i, 2] + padding)
        if d < min_dist: min_dist = d

    for i in range(segments.shape[0]):
        r = segments[i, 4] * 0.5 + padding
        d = _numba_ray_capsule(ox, oy, dx, dy, segments[i, 0], segments[i, 1], segments[i, 2], segments[i, 3], r)
        if d < min_dist: min_dist = d

    return min_dist

@njit(fastmath=True, cache=True)
def _numba_ray_batch_analytical(ox, oy, angles, max_range, rects, circles, segments, padding):
    n = len(angles)
    res = np.empty(n, dtype=np.float32)
    for i in range(n):
        res[i] = _numba_ray_all_obstacles(ox, oy, angles[i], max_range, rects, circles, segments, padding)
    return res

@njit(fastmath=True, cache=True)
def _numba_point_blocked_analytical(px, py, padding, rects, circles, segments):
    for i in range(rects.shape[0]):
        rx, ry, rw, rh = rects[i, 0] - padding, rects[i, 1] - padding, rects[i, 2] + 2.0 * padding, rects[i, 3] + 2.0 * padding
        if rx <= px <= rx + rw and ry <= py <= ry + rh: return True

    for i in range(circles.shape[0]):
        dx, dy, r = px - circles[i, 0], py - circles[i, 1], circles[i, 2] + padding
        if dx * dx + dy * dy <= r * r: return True

    for i in range(segments.shape[0]):
        x1, y1, x2, y2, thick = segments[i, 0], segments[i, 1], segments[i, 2], segments[i, 3], segments[i, 4]
        r = thick * 0.5 + padding
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1
        seg_len2 = vx * vx + vy * vy
        if seg_len2 <= 1e-9:
            dist2 = (px - x1)**2 + (py - y1)**2
        else:
            t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
            dist2 = (px - (x1 + t * vx))**2 + (py - (y1 + t * vy))**2
        if dist2 <= r * r: return True
    return False

def build_occupancy(width=None, height=None, cell=None, obstacles=None):
    global _OCC_GRID, _OCC_CELL, _RECT_OBS, _CIRCLE_OBS, _SEGMENT_OBS, _OBS_COMPILED
    width = float(width or map_config.width)
    height = float(height or map_config.height)
    cell = float(cell or getattr(map_config, 'occ_cell', getattr(map_config, 'pixel_size', 8.0)))
    obstacles = obstacles or getattr(map_config, 'obstacles', [])

    nx = int(math.ceil(width / cell))
    ny = int(math.ceil(height / cell))
    grid = np.zeros((ny, nx), dtype=np.bool_)

    for obs in obstacles:
        typ = obs.get('type')
        if typ == 'rect':
            x1 = max(0, int(math.floor(obs['x'] / cell)))
            y1 = max(0, int(math.floor(obs['y'] / cell)))
            x2 = min(nx, int(math.ceil((obs['x'] + obs['w']) / cell)))
            y2 = min(ny, int(math.ceil((obs['y'] + obs['h']) / cell)))
            if x2 > x1 and y2 > y1:
                grid[y1:y2, x1:x2] = True
        elif typ == 'circle':
            x1 = max(0, int(math.floor((obs['cx'] - obs['r']) / cell)))
            y1 = max(0, int(math.floor((obs['cy'] - obs['r']) / cell)))
            x2 = min(nx, int(math.ceil((obs['cx'] + obs['r']) / cell)))
            y2 = min(ny, int(math.ceil((obs['cy'] + obs['r']) / cell)))
            rr = obs['r'] ** 2
            for iy in range(y1, y2):
                cyc = (iy + 0.5) * cell
                dy2 = (cyc - obs['cy']) ** 2
                for ix in range(x1, x2):
                    cxc = (ix + 0.5) * cell
                    if (cxc - obs['cx']) ** 2 + dy2 <= rr:
                        grid[iy, ix] = True
        elif typ == 'segment':
            pad = obs.get('thick', 8.0) * 0.5
            xmin, xmax = min(obs['x1'], obs['x2']) - pad, max(obs['x1'], obs['x2']) + pad
            ymin, ymax = min(obs['y1'], obs['y2']) - pad, max(obs['y1'], obs['y2']) + pad
            ix1 = max(0, int(math.floor(xmin / cell)))
            iy1 = max(0, int(math.floor(ymin / cell)))
            ix2 = min(nx, int(math.ceil(xmax / cell)))
            iy2 = min(ny, int(math.ceil(ymax / cell)))
            r2 = pad ** 2
            vx, vy = obs['x2'] - obs['x1'], obs['y2'] - obs['y1']
            seg_len2 = vx * vx + vy * vy
            for iy in range(iy1, iy2):
                cy = (iy + 0.5) * cell
                for ix in range(ix1, ix2):
                    cx = (ix + 0.5) * cell
                    wx, wy = cx - obs['x1'], cy - obs['y1']
                    if seg_len2 <= 1e-9:
                        dist2 = (cx - obs['x1'])**2 + (cy - obs['y1'])**2
                    else:
                        t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
                        dist2 = (cx - (obs['x1'] + t * vx))**2 + (cy - (obs['y1'] + t * vy))**2
                    if dist2 <= r2:
                        grid[iy, ix] = True

    _OCC_GRID, _OCC_CELL, _OBS_COMPILED = grid, cell, True

    rects, circles, segments = [], [], []
    for obs in obstacles:
        typ = obs.get('type')
        if typ == 'rect':
            rects.append([float(obs['x']), float(obs['y']), float(obs['w']), float(obs['h'])])
        elif typ == 'circle':
            circles.append([float(obs['cx']), float(obs['cy']), float(obs['r'])])
        elif typ == 'segment':
            segments.append([float(obs['x1']), float(obs['y1']), float(obs['x2']), float(obs['y2']), float(obs.get('thick', 8.0))])
    _RECT_OBS = np.array(rects, dtype=np.float64) if rects else np.empty((0, 4), dtype=np.float64)
    _CIRCLE_OBS = np.array(circles, dtype=np.float64) if circles else np.empty((0, 3), dtype=np.float64)
    _SEGMENT_OBS = np.array(segments, dtype=np.float64) if segments else np.empty((0, 5), dtype=np.float64)

def ray_distance_grid(origin, angle_rad, max_range, padding=0.0):
    return _numba_ray_all_obstacles(float(origin[0]), float(origin[1]), float(angle_rad), float(max_range),
                                    _RECT_OBS, _CIRCLE_OBS, _SEGMENT_OBS, float(padding))

def ray_distances_multi(origin, angles_rad, max_range, padding=0.0):
    return _numba_ray_batch_analytical(float(origin[0]), float(origin[1]),
                                        np.asarray(angles_rad, dtype=np.float64),
                                        float(max_range),
                                        _RECT_OBS, _CIRCLE_OBS, _SEGMENT_OBS, float(padding))

def is_point_blocked(px, py, padding=0.0):
    return _numba_point_blocked_analytical(float(px), float(py), float(padding),
                                            _RECT_OBS, _CIRCLE_OBS, _SEGMENT_OBS)

def get_canvas(target, tracker, tracker_traj, target_traj, surface=None, fov_points=None, collision_info=None):
    w, h = map_config.width, map_config.height
    ss = getattr(map_config, 'ssaa', 1)
    if pygame is None:
        return np.zeros((h, w, 3), dtype=np.uint8)

    if surface is None:
        surface = pygame.Surface((w * ss, h * ss), flags=pygame.SRCALPHA)
    surface.fill(map_config.background_color)

    _draw_grid(surface)

    if collision_info and collision_info.get('collision'):
        _draw_obstacles(surface, exclude_obstacle=collision_info.get('collided_obstacle'))
    else:
        _draw_obstacles(surface)

    _draw_fov(surface, tracker, fov_points)
    _draw_capture_sector(surface, tracker)
    _draw_trail(surface, tracker_traj, map_config.trail_color_tracker, map_config.trail_width)
    _draw_trail(surface, target_traj, map_config.trail_color_target, map_config.trail_width)

    if collision_info and collision_info.get('collision'):
        collision_color = getattr(map_config, 'COLLISION_TRACKER_COLOR', (255, 50, 50, 255))
        _draw_agent(surface, tracker, collision_color, role='tracker')
    else:
        _draw_agent(surface, tracker, map_config.tracker_color, role='tracker')

    _draw_agent(surface, target, map_config.target_color, role='target')

    canvas = pygame.transform.smoothscale(surface, (w, h)) if ss > 1 else surface
    return pygame.surfarray.array3d(canvas).swapaxes(0, 1)

def _resolve_obstacle_collision(old_pos, new_pos):
    agent_radius = float(getattr(map_config, 'agent_radius', map_config.pixel_size * 0.5))
    center_x = new_pos['x'] + map_config.pixel_size * 0.5
    center_y = new_pos['y'] + map_config.pixel_size * 0.5

    if is_point_blocked(center_x, center_y, padding=agent_radius):
        new_pos['x'], new_pos['y'] = old_pos['x'], old_pos['y']
    return new_pos

def agent_move_velocity(agent, angle_delta, speed, max_speed, role=None):
    """
    更新智能体的位置和朝向
    
    Args:
        agent: 智能体状态字典
        angle_delta: 角速度（度/步）
        speed: 目标速度
        max_speed: 最大速度
        role: 'defender' 或 'attacker'
    
    Note:
        - Defender 的碰撞检测在 env.py 的 step() 中单独处理
        - Attacker 在此函数内处理碰撞（碰撞时回滚位置）
    """
    old_state = dict(agent)
    prev_speed = agent.get('v', 0.0)
    speed = float(np.clip(speed, 0.0, max_speed))

    # 根据角色获取参数
    if role == 'defender':
        max_turn = float(getattr(map_config, 'defender_max_angular_speed', 6.0))
    elif role == 'attacker':
        # Attacker 有加速度限制
        max_acc = float(getattr(map_config, 'attacker_max_acc', 0.6))
        speed_change = np.clip(speed - prev_speed, -max_acc, max_acc)
        speed = prev_speed + speed_change
        max_turn = float(getattr(map_config, 'attacker_max_angular_speed', 12.0))
    else:
        max_turn = 10.0

    # 更新朝向
    angle_delta = float(np.clip(angle_delta, -max_turn, max_turn))
    new_theta = (agent.get('theta', 0.0) + angle_delta) % 360.0
    agent['theta'] = float(new_theta)

    # 更新位置
    rad_theta = math.radians(new_theta)
    agent['x'] = float(np.clip(
        agent['x'] + speed * math.cos(rad_theta),
        0, map_config.width - map_config.pixel_size
    ))
    agent['y'] = float(np.clip(
        agent['y'] + speed * math.sin(rad_theta),
        0, map_config.height - map_config.pixel_size
    ))

    agent['v'] = speed

    # Attacker 碰撞时回滚位置（Defender 的碰撞在 env.step() 中单独检测）
    if role == 'attacker':
        return _resolve_obstacle_collision(old_state, agent)
    return agent

def reward_calculate_tad(defender, attacker, target, prev_defender=None, prev_attacker=None,
                         defender_collision=False, attacker_collision=False,
                         defender_captured=False, attacker_captured=False,
                         capture_progress_defender=0, capture_progress_attacker=0,
                         capture_required_steps=0, radar=None, initial_dist_def_tgt=None):
    info = {
        'capture_progress_defender': int(capture_progress_defender),
        'capture_progress_attacker': int(capture_progress_attacker),
        'capture_required_steps': int(capture_required_steps),
        'defender_collision': bool(defender_collision),
        'attacker_collision': bool(attacker_collision)
    }

    reward = 0.0
    terminated = False

    # 时间惩罚
    reward -= 0.01

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
    # attacker_collision 不影响 episode 终止，仅记录在 info 中

    return float(reward), bool(terminated), False, info


def reward_calculate_chase(defender, attacker, target, prev_defender=None, prev_attacker=None,
                           defender_collision=False, attacker_collision=False,
                           defender_captured=False, attacker_captured=False,
                           capture_progress_defender=0, capture_progress_attacker=0,
                           capture_required_steps=0, radar=None, initial_dist_def_att=None):
    """
    纯追逃奖励函数 - 用于训练 Chase 技能

    特点：
    - 纯追逃环境：忽略 target 的任何判定（attacker_captured 不触发终止）
    - 时间惩罚：每步-0.04（总共约-20）
    - defender捕获attacker: +10，episode结束
    - defender靠近attacker的微分奖励: 总计+10（按进度比例）
    - defender碰撞障碍物: -10，episode结束
    - attacker碰撞障碍物: 不结束episode
    - 超时算defender失败: 无额外奖励
    - 不使用GRU预测器
    """
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

    # 奖励配置
    capture_reward = 10.0  # 捕获奖励
    distance_total_reward = 10.0  # 微分距离奖励总计
    collision_penalty = 10.0  # 碰撞惩罚

    # 计算defender到attacker的距离
    dx_def_att = (defender['x'] + map_config.pixel_size * 0.5) - (attacker['x'] + map_config.pixel_size * 0.5)
    dy_def_att = (defender['y'] + map_config.pixel_size * 0.5) - (attacker['y'] + map_config.pixel_size * 0.5)
    curr_dist_def_att = math.hypot(dx_def_att, dy_def_att)

    agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
    capture_radius = agent_radius * 2  # 两个agent的半径之和

    # 微分距离奖励：按进度比例给奖励（类似protect1的导航奖励）
    if prev_defender is not None and prev_attacker is not None and initial_dist_def_att is not None and initial_dist_def_att > capture_radius:
        prev_dx_def_att = (prev_defender['x'] + map_config.pixel_size * 0.5) - (prev_attacker['x'] + map_config.pixel_size * 0.5)
        prev_dy_def_att = (prev_defender['y'] + map_config.pixel_size * 0.5) - (prev_attacker['y'] + map_config.pixel_size * 0.5)
        prev_dist_def_att = math.hypot(prev_dx_def_att, prev_dy_def_att)

        # 计算边界距离（到捕获半径的距离）
        prev_boundary_dist = max(0.0, prev_dist_def_att - capture_radius)
        curr_boundary_dist = max(0.0, curr_dist_def_att - capture_radius)
        initial_boundary_dist = max(0.0, initial_dist_def_att - capture_radius)

        if initial_boundary_dist > 0:
            distance_progress = (prev_boundary_dist - curr_boundary_dist) / initial_boundary_dist
            distance_reward = distance_progress * distance_total_reward
            reward += distance_reward

    # 终止奖励 - 纯追逃：只关心 defender 捕获 attacker，忽略 target
    if defender_captured:
        terminated = True
        info['reason'] = 'defender_caught_attacker'
        info['win'] = True
        reward += capture_reward
    elif defender_collision:
        terminated = True
        reward -= collision_penalty
        info['reason'] = 'defender_collision'
        info['win'] = False
    # 注意：attacker_captured (attacker到达target) 被忽略，不触发终止
    # attacker_collision 也不影响 episode 终止，仅记录在 info 中

    return float(reward), bool(terminated), False, info


    ss = float(getattr(map_config, 'ssaa', 1))
    target_radius = float(getattr(map_config, 'target_radius', 16))
    x_world = float(target['x']) + float(map_config.pixel_size) * 0.5
    y_world = float(target['y']) + float(map_config.pixel_size) * 0.5
    cx, cy = int(x_world * ss), int(y_world * ss)
    r_i = max(3, int(target_radius * ss))

    thickness = max(2, int(2.0 * ss))
    pygame.draw.circle(surface, color[:3], (cx, cy), r_i, thickness)

    pygame.draw.circle(surface, color[:3], (cx, cy), max(1, int(r_i * 0.3)))


def get_canvas(target, tracker, tracker_traj, target_traj, surface=None, fov_points=None, collision_info=None):
    w, h = map_config.width, map_config.height
    ss = getattr(map_config, 'ssaa', 1)
    if pygame is None:
        return np.zeros((h, w, 3), dtype=np.uint8)

    if surface is None:
        surface = pygame.Surface((w * ss, h * ss), flags=pygame.SRCALPHA)
    surface.fill(map_config.background_color)

    _draw_grid(surface)

    if collision_info and collision_info.get('collision'):
        _draw_obstacles(surface, exclude_obstacle=collision_info.get('collided_obstacle'))
    else:
        _draw_obstacles(surface)

    _draw_fov(surface, tracker, fov_points)
    _draw_capture_sector(surface, tracker)
    _draw_trail(surface, tracker_traj, map_config.trail_color_tracker, map_config.trail_width)
    _draw_trail(surface, target_traj, map_config.trail_color_target, map_config.trail_width)

    if collision_info and collision_info.get('collision'):
        collision_color = getattr(map_config, 'COLLISION_TRACKER_COLOR', (255, 50, 50, 255))
        _draw_agent(surface, tracker, collision_color, role='tracker')
    else:
        _draw_agent(surface, tracker, map_config.tracker_color, role='tracker')

    _draw_agent(surface, target, map_config.target_color, role='target')

    canvas = pygame.transform.smoothscale(surface, (w, h)) if ss > 1 else surface
    return pygame.surfarray.array3d(canvas).swapaxes(0, 1)

def _resolve_obstacle_collision(old_pos, new_pos):
    agent_radius = float(getattr(map_config, 'agent_radius', map_config.pixel_size * 0.5))
    center_x = new_pos['x'] + map_config.pixel_size * 0.5
    center_y = new_pos['y'] + map_config.pixel_size * 0.5

    if is_point_blocked(center_x, center_y, padding=agent_radius):
        new_pos['x'], new_pos['y'] = old_pos['x'], old_pos['y']
    return new_pos

def agent_move_velocity(agent, angle_delta, speed, max_speed, role=None):
    """
    更新智能体的位置和朝向
    
    Args:
        agent: 智能体状态字典
        angle_delta: 角速度（度/步）
        speed: 目标速度
        max_speed: 最大速度
        role: 'defender' 或 'attacker'
    
    Note:
        - Defender 的碰撞检测在 env.py 的 step() 中单独处理
        - Attacker 在此函数内处理碰撞（碰撞时回滚位置）
    """
    old_state = dict(agent)
    prev_speed = agent.get('v', 0.0)
    speed = float(np.clip(speed, 0.0, max_speed))

    # 根据角色获取参数
    if role == 'defender':
        max_turn = float(getattr(map_config, 'defender_max_angular_speed', 6.0))
    elif role == 'attacker':
        # Attacker 有加速度限制
        max_acc = float(getattr(map_config, 'attacker_max_acc', 0.6))
        speed_change = np.clip(speed - prev_speed, -max_acc, max_acc)
        speed = prev_speed + speed_change
        max_turn = float(getattr(map_config, 'attacker_max_angular_speed', 12.0))
    else:
        max_turn = 10.0

    # 更新朝向
    angle_delta = float(np.clip(angle_delta, -max_turn, max_turn))
    new_theta = (agent.get('theta', 0.0) + angle_delta) % 360.0
    agent['theta'] = float(new_theta)

    # 更新位置
    rad_theta = math.radians(new_theta)
    agent['x'] = float(np.clip(
        agent['x'] + speed * math.cos(rad_theta),
        0, map_config.width - map_config.pixel_size
    ))
    agent['y'] = float(np.clip(
        agent['y'] + speed * math.sin(rad_theta),
        0, map_config.height - map_config.pixel_size
    ))

    agent['v'] = speed

    # Attacker 碰撞时回滚位置（Defender 的碰撞在 env.step() 中单独检测）
    if role == 'attacker':
        return _resolve_obstacle_collision(old_state, agent)
    return agent

def reward_calculate_tad(defender, attacker, target, prev_defender=None, prev_attacker=None,
                         defender_collision=False, attacker_collision=False,
                         defender_captured=False, attacker_captured=False,
                         capture_progress_defender=0, capture_progress_attacker=0,
                         capture_required_steps=0, radar=None, initial_dist_def_tgt=None):
    info = {
        'capture_progress_defender': int(capture_progress_defender),
        'capture_progress_attacker': int(capture_progress_attacker),
        'capture_required_steps': int(capture_required_steps),
        'defender_collision': bool(defender_collision),
        'attacker_collision': bool(attacker_collision)
    }

    reward = 0.0
    terminated = False

    # 时间惩罚
    reward -= 0.01

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
    # attacker_collision 不影响 episode 终止，仅记录在 info 中

    return float(reward), bool(terminated), False, info


def reward_calculate_chase(defender, attacker, target, prev_defender=None, prev_attacker=None,
                           defender_collision=False, attacker_collision=False,
                           defender_captured=False, attacker_captured=False,
                           capture_progress_defender=0, capture_progress_attacker=0,
                           capture_required_steps=0, radar=None, initial_dist_def_att=None):
    """
    纯追逃奖励函数 - 用于训练 Chase 技能

    特点：
    - 纯追逃环境：忽略 target 的任何判定（attacker_captured 不触发终止）
    - 时间惩罚：每步-0.04（总共约-20）
    - defender捕获attacker: +10，episode结束
    - defender靠近attacker的微分奖励: 总计+10（按进度比例）
    - defender碰撞障碍物: -10，episode结束
    - attacker碰撞障碍物: 不结束episode
    - 超时算defender失败: 无额外奖励
    - 不使用GRU预测器
    """
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

    # 奖励配置
    capture_reward = 10.0  # 捕获奖励
    distance_total_reward = 10.0  # 微分距离奖励总计
    collision_penalty = 10.0  # 碰撞惩罚

    # 计算defender到attacker的距离
    dx_def_att = (defender['x'] + map_config.pixel_size * 0.5) - (attacker['x'] + map_config.pixel_size * 0.5)
    dy_def_att = (defender['y'] + map_config.pixel_size * 0.5) - (attacker['y'] + map_config.pixel_size * 0.5)
    curr_dist_def_att = math.hypot(dx_def_att, dy_def_att)

    agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
    capture_radius = agent_radius * 2  # 两个agent的半径之和

    # 微分距离奖励：按进度比例给奖励（类似protect1的导航奖励）
    if prev_defender is not None and prev_attacker is not None and initial_dist_def_att is not None and initial_dist_def_att > capture_radius:
        prev_dx_def_att = (prev_defender['x'] + map_config.pixel_size * 0.5) - (prev_attacker['x'] + map_config.pixel_size * 0.5)
        prev_dy_def_att = (prev_defender['y'] + map_config.pixel_size * 0.5) - (prev_attacker['y'] + map_config.pixel_size * 0.5)
        prev_dist_def_att = math.hypot(prev_dx_def_att, prev_dy_def_att)

        # 计算边界距离（到捕获半径的距离）
        prev_boundary_dist = max(0.0, prev_dist_def_att - capture_radius)
        curr_boundary_dist = max(0.0, curr_dist_def_att - capture_radius)
        initial_boundary_dist = max(0.0, initial_dist_def_att - capture_radius)

        if initial_boundary_dist > 0:
            distance_progress = (prev_boundary_dist - curr_boundary_dist) / initial_boundary_dist
            distance_reward = distance_progress * distance_total_reward
            reward += distance_reward

    # 终止奖励 - 纯追逃：只关心 defender 捕获 attacker，忽略 target
    if defender_captured:
        terminated = True
        info['reason'] = 'defender_caught_attacker'
        info['win'] = True
        reward += capture_reward
    elif defender_collision:
        terminated = True
        reward -= collision_penalty
        info['reason'] = 'defender_collision'
        info['win'] = False
    # 注意：attacker_captured (attacker到达target) 被忽略，不触发终止
    # attacker_collision 也不影响 episode 终止，仅记录在 info 中

    return float(reward), bool(terminated), False, info


# ============================================================================
# Academic-style rendering functions for TAD environment
# ============================================================================

def _draw_grid_academic(surface):
    """Draw subtle coordinate grid with tick marks."""
    if pygame is None or not getattr(map_config, 'draw_grid', True):
        return
    ss = getattr(map_config, 'ssaa', 1)
    step = int(map_config.grid_step * ss)
    w, h = surface.get_size()
    # Very light grid lines
    grid_col = (230, 230, 235)
    for x in range(0, w, step):
        pygame.draw.line(surface, grid_col, (x, 0), (x, h), 1)
    for y in range(0, h, step):
        pygame.draw.line(surface, grid_col, (0, y), (w, y), 1)


def _draw_obstacles_academic(surface, exclude_obstacle=None):
    """Draw obstacles with light fill + dark outline (academic style)."""
    if pygame is None:
        return
    ss = getattr(map_config, 'ssaa', 1)
    fill_color = (200, 200, 205, 255)      # light gray fill
    outline_color = (80, 80, 90, 255)       # dark outline
    outline_w = max(1, int(1.2 * ss))
    for obs in getattr(map_config, 'obstacles', []):
        if exclude_obstacle is not None and obs is exclude_obstacle:
            continue
        if obs['type'] == 'rect':
            rect = pygame.Rect(
                int(obs['x']*ss), int(obs['y']*ss),
                int(obs['w']*ss), int(obs['h']*ss))
            pygame.draw.rect(surface, fill_color, rect)
            pygame.draw.rect(surface, outline_color, rect, outline_w)
        elif obs['type'] == 'circle':
            cx, cy, r = int(obs['cx']*ss), int(obs['cy']*ss), int(obs['r']*ss)
            pygame.draw.circle(surface, fill_color, (cx, cy), r)
            pygame.draw.circle(surface, outline_color, (cx, cy), r, outline_w)
        elif obs.get('type') == 'segment':
            thick = max(1, int(float(obs.get('thick', 8.0)) * ss))
            x1, y1 = int(obs['x1']*ss), int(obs['y1']*ss)
            x2, y2 = int(obs['x2']*ss), int(obs['y2']*ss)
            pygame.draw.line(surface, fill_color, (x1, y1), (x2, y2), thick)
            pygame.draw.line(surface, outline_color, (x1, y1), (x2, y2), outline_w)


def _draw_trail_academic(surface, traj, color_rgb, width_px, time_markers=True):
    """Draw academic-style trajectory: steady line, markers, and arrows."""
    if pygame is None or len(traj) < 2:
        return
    ss = getattr(map_config, 'ssaa', 1)
    # 移除轨迹长度限制，绘制完整轨迹
    points = traj
    if len(points) < 2:
        return

    screen_pts = [(int(p[0] * ss), int(p[1] * ss)) for p in points]
    r, g, b = color_rgb[:3]
    w = max(int(width_px * ss), 1)
    n = len(screen_pts)

    # Main line (steady alpha to match PNG style)
    line_color = (r, g, b, 190)
    for i in range(n - 1):
        pygame.draw.line(surface, line_color, screen_pts[i], screen_pts[i + 1], w)

    # Time markers: small dots every ~10% of trajectory
    if time_markers and n > 10:
        marker_interval = max(1, n // 10)
        marker_r = max(2, int(2.5 * ss))
        for i in range(0, n, marker_interval):
            pygame.draw.circle(surface, (255, 255, 255, 200), screen_pts[i], marker_r + 1)
            pygame.draw.circle(surface, (r, g, b, 230), screen_pts[i], marker_r)

    # Direction arrows every ~20% of trajectory
    arrow_interval = max(1, n // 5)
    for i in range(arrow_interval, n - 1, arrow_interval):
        x1, y1 = screen_pts[i]
        x2, y2 = screen_pts[min(i + 1, n - 1)]
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) + abs(dy) < 1:
            continue
        # Short arrow segment
        ax2 = x1 + int(dx * 0.6)
        ay2 = y1 + int(dy * 0.6)
        pygame.draw.line(surface, (r, g, b, 220), (x1, y1), (ax2, ay2), max(1, w))
        # Arrow head (small V)
        ang = math.atan2(dy, dx)
        head_len = max(4, int(6 * ss))
        head_ang = math.radians(25)
        hx1 = int(ax2 - head_len * math.cos(ang - head_ang))
        hy1 = int(ay2 - head_len * math.sin(ang - head_ang))
        hx2 = int(ax2 - head_len * math.cos(ang + head_ang))
        hy2 = int(ay2 - head_len * math.sin(ang + head_ang))
        pygame.draw.line(surface, (r, g, b, 220), (ax2, ay2), (hx1, hy1), max(1, w))
        pygame.draw.line(surface, (r, g, b, 220), (ax2, ay2), (hx2, hy2), max(1, w))


def _draw_agent_academic(surface, agent, color_rgb, role=None, label=None):
    """Draw agent as a filled circle with heading arrow (academic style)."""
    if pygame is None:
        return

    ss = float(getattr(map_config, 'ssaa', 1))
    x_world = float(agent['x']) + float(map_config.pixel_size) * 0.5
    y_world = float(agent['y']) + float(map_config.pixel_size) * 0.5
    cx, cy = int(x_world * ss), int(y_world * ss)

    if role == 'target':
        r_world = getattr(map_config, 'target_radius', 16.0)
    else:
        r_world = getattr(map_config, 'agent_radius', 8.0)
    r_i = max(3, int(r_world * ss))

    r, g, b = color_rgb[:3]

    if role == 'target':
        # Target: double-circle (bullseye) style
        pygame.draw.circle(surface, (r, g, b, 60), (cx, cy), r_i)
        pygame.draw.circle(surface, (r, g, b, 255), (cx, cy), r_i, max(2, int(2.0 * ss)))
        inner_r = max(2, int(r_i * 0.35))
        pygame.draw.circle(surface, (r, g, b, 255), (cx, cy), inner_r)
        # Cross-hair lines
        line_len = int(r_i * 0.6)
        lw = max(1, int(1.0 * ss))
        pygame.draw.line(surface, (r, g, b, 180), (cx - line_len, cy), (cx + line_len, cy), lw)
        pygame.draw.line(surface, (r, g, b, 180), (cx, cy - line_len), (cx, cy + line_len), lw)
    else:
        # Agent: filled circle + heading triangle
        pygame.draw.circle(surface, (r, g, b, 50), (cx, cy), r_i)
        pygame.draw.circle(surface, (r, g, b, 255), (cx, cy), r_i, max(2, int(1.8 * ss)))

        # Heading arrow (chevron)
        theta_rad = math.radians(agent.get('theta', 0.0))
        cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
        arrow_len = r_i * 1.5
        arrow_w = r_i * 0.55

        tip = (cx + arrow_len * cos_t, cy + arrow_len * sin_t)
        left = (cx - arrow_w * sin_t, cy + arrow_w * cos_t)
        right = (cx + arrow_w * sin_t, cy - arrow_w * cos_t)

        pygame.draw.polygon(surface, (r, g, b, 220), [tip, left, right])


def _draw_fov_academic(surface, tracker, fov_points=None):
    """Draw FOV cone with subtle fill."""
    if pygame is None or not fov_points or len(fov_points) < 3:
        return
    fill_color = (100, 160, 255, 20)
    pygame.gfxdraw.filled_polygon(surface, fov_points, fill_color)
    # Edge lines
    center = fov_points[0]
    p_left = fov_points[1]
    p_right = fov_points[-1]
    edge_color = (100, 160, 255, 120)
    c_int = (int(center[0]), int(center[1]))
    pl_int = (int(p_left[0]), int(p_left[1]))
    pr_int = (int(p_right[0]), int(p_right[1]))
    pygame.draw.line(surface, edge_color, c_int, pl_int, 1)
    pygame.draw.line(surface, edge_color, c_int, pr_int, 1)


def _draw_capture_sector_academic(surface, tracker):
    """Draw capture sector with dashed arc style."""
    if pygame is None:
        return
    ss = getattr(map_config, 'ssaa', 1)
    cx_world = tracker['x'] + map_config.pixel_size * 0.5
    cy_world = tracker['y'] + map_config.pixel_size * 0.5
    cx, cy = cx_world * ss, cy_world * ss

    heading_rad = math.radians(tracker.get('theta', 0.0))
    half_sector = math.radians(getattr(map_config, 'capture_sector_angle_deg', 60.0)) * 0.5
    radius = float(getattr(map_config, 'capture_radius', 10.0))

    pts = [(cx, cy)]
    num_rays = 24
    for i in range(num_rays + 1):
        ang = heading_rad - half_sector + (2 * half_sector * i / num_rays)
        dist = ray_distance_grid(
            (cx_world, cy_world), ang, radius, padding=0.0
        )
        pts.append((cx + dist * ss * math.cos(ang), cy + dist * ss * math.sin(ang)))

    if len(pts) > 2:
        fill_color = (80, 200, 130, 25)
        pygame.gfxdraw.filled_polygon(surface, pts, fill_color)
        arc_color = (80, 200, 130, 140)
        # Draw arc outline with individual segments for clean look
        for i in range(1, len(pts) - 1):
            pygame.draw.line(surface, arc_color, 
                           (int(pts[i][0]), int(pts[i][1])), 
                           (int(pts[i+1][0]), int(pts[i+1][1])), 1)


def _draw_capture_radius_academic(surface, defender):
    """Draw capture arc aligned with heading (avoid full 360 circle)."""
    if pygame is None:
        return
    ss = getattr(map_config, 'ssaa', 1)
    cx_world = defender['x'] + map_config.pixel_size * 0.5
    cy_world = defender['y'] + map_config.pixel_size * 0.5
    cx, cy = int(cx_world * ss), int(cy_world * ss)
    radius = float(getattr(map_config, 'capture_radius', 10.0))
    r_i = max(2, int(radius * ss))
    heading_rad = math.radians(defender.get('theta', 0.0))
    half_sector = math.radians(getattr(map_config, 'capture_sector_angle_deg', 60.0)) * 0.5
    dash_color = (80, 200, 130, 120)
    num_dashes = 24
    start_ang = heading_rad - half_sector
    end_ang = heading_rad + half_sector
    for i in range(num_dashes):
        if i % 2 == 0:
            a1 = start_ang + (end_ang - start_ang) * (i / num_dashes)
            a2 = start_ang + (end_ang - start_ang) * ((i + 1) / num_dashes)
            p1 = (int(cx + r_i * math.cos(a1)), int(cy + r_i * math.sin(a1)))
            p2 = (int(cx + r_i * math.cos(a2)), int(cy + r_i * math.sin(a2)))
            pygame.draw.line(surface, dash_color, p1, p2, 1)


def _draw_start_marker(surface, pos, color_rgb, ss):
    """Draw start position marker (small hollow square)."""
    if pygame is None:
        return
    cx, cy = int(pos[0] * ss), int(pos[1] * ss)
    size = max(4, int(5 * ss))
    rect = pygame.Rect(cx - size, cy - size, size * 2, size * 2)
    pygame.draw.rect(surface, color_rgb[:3], rect, max(1, int(1.5 * ss)))


# Keep old functions as aliases for backward compatibility
_draw_grid = _draw_grid_academic
_draw_agent = _draw_agent_academic
_draw_trail = lambda surface, traj, rgba, width_px: _draw_trail_academic(surface, traj, rgba[:3], width_px)
_draw_obstacles = _draw_obstacles_academic
_draw_fov = _draw_fov_academic
_draw_capture_sector = _draw_capture_sector_academic
_draw_capture_radius = _draw_capture_radius_academic
_draw_target = lambda surface, target, color: _draw_agent_academic(surface, target, color[:3], role='target')


def get_canvas_tad(target, defender, attacker, defender_traj, attacker_traj, surface=None, fov_points=None, collision_info=None):
    """Render TAD scene (academic style) — used for GIF frames."""
    w, h = map_config.width, map_config.height
    ss = getattr(map_config, 'ssaa', 1)
    if pygame is None:
        return np.zeros((h, w, 3), dtype=np.uint8)

    if surface is None:
        surface = pygame.Surface((w * ss, h * ss), flags=pygame.SRCALPHA)
    # Clean white background
    surface.fill((252, 252, 255))

    _draw_grid_academic(surface)

    if collision_info and collision_info.get('collision'):
        _draw_obstacles_academic(surface, exclude_obstacle=collision_info.get('collided_obstacle'))
    else:
        _draw_obstacles_academic(surface)
    _draw_fov_academic(surface, defender, fov_points)
    _draw_capture_sector_academic(surface, defender)
    _draw_capture_radius_academic(surface, defender)
    # Defender trail (blue)
    _draw_trail_academic(surface, defender_traj, (50, 100, 220), map_config.trail_width)
    # Attacker trail (red)
    _draw_trail_academic(surface, attacker_traj, (220, 80, 60), map_config.trail_width)
    # Draw agents
    if collision_info and collision_info.get('collision'):
        _draw_agent_academic(surface, defender, (220, 50, 50), role='defender')
    else:
        _draw_agent_academic(surface, defender, (50, 100, 220), role='defender')
    if collision_info and collision_info.get('attacker_collision'):
        _draw_agent_academic(surface, attacker, (180, 50, 220), role='attacker')
    else:
        _draw_agent_academic(surface, attacker, (220, 80, 60), role='attacker')
    # Target
    _draw_agent_academic(surface, target, (50, 180, 80), role='target')
    # Start markers (draw AFTER trails so they don't get covered)
    if defender_traj:
        _draw_start_marker(surface, defender_traj[0], (50, 100, 220), ss)
    if attacker_traj:
        _draw_start_marker(surface, attacker_traj[0], (220, 80, 60), ss)

    canvas = pygame.transform.smoothscale(surface, (w, h)) if ss > 1 else surface
    return pygame.surfarray.array3d(canvas).swapaxes(0, 1)


# ═══════════════════════════════════════════════════════════════════════
# Matplotlib-based academic renderer (for GIF frames matching PNG style)
# ═══════════════════════════════════════════════════════════════════════

class _MplRenderer:
    """Matplotlib-based frame renderer for academic-style GIF frames.
    
    Produces frames visually identical to make_trajectory_plot() PNG output.
    Maintains a persistent figure object for performance (~30-60ms per frame).
    """

    def __init__(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.collections import PatchCollection

        self._plt = plt
        self._mpatches = mpatches
        self._PatchCollection = PatchCollection

        w = map_config.width
        h = map_config.height
        self.w = w
        self.h = h

        # Match PNG style
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 10,
            'axes.linewidth': 0.8,
        })

        dpi = 100
        fig_w, fig_h = 5.0, 5.0  # 500px — fast yet sharp enough for GIF
        self.fig, self.ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
        self.fig.set_facecolor('white')
        # Fixed margins — much faster than tight_layout() per frame
        self.fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.10)

    # ------------------------------------------------------------------
    def render_frame(self, target, defender, attacker,
                     defender_traj, attacker_traj,
                     fov_points=None, collision_info=None, **kwargs):
        """Render one frame -> HxWx3 uint8 numpy array."""
        ax = self.ax
        plt = self._plt
        mpatches = self._mpatches
        PatchCollection = self._PatchCollection

        ax.clear()

        w, h = self.w, self.h
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_xlabel('$x$ (pixels)', fontsize=11)
        ax.set_ylabel('$y$ (pixels)', fontsize=11)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(direction='in', length=3, width=0.6, labelsize=9)
        ax.set_facecolor('white')

        # Colors (identical to make_trajectory_plot)
        c_def = '#3264DC'
        c_atk = '#DC5038'
        c_tgt = '#32B450'
        c_cap = '#E86040'

        px = float(map_config.pixel_size)

        # ── Obstacles ──────────────────────────────────────────────────
        obs_patches = []
        for obs in getattr(map_config, 'obstacles', []):
            if obs['type'] == 'rect':
                obs_patches.append(
                    mpatches.Rectangle((obs['x'], obs['y']), obs['w'], obs['h']))
            elif obs['type'] == 'circle':
                obs_patches.append(
                    mpatches.Circle((obs['cx'], obs['cy']), obs['r']))
            elif obs['type'] == 'segment':
                thick = float(obs.get('thick', 8.0))
                x1, y1 = obs['x1'], obs['y1']
                x2, y2 = obs['x2'], obs['y2']
                dx, dy = x2 - x1, y2 - y1
                length = max(1e-6, (dx**2 + dy**2)**0.5)
                nx = -dy / length * thick / 2
                ny =  dx / length * thick / 2
                verts = [(x1+nx, y1+ny), (x2+nx, y2+ny),
                         (x2-nx, y2-ny), (x1-nx, y1-ny)]
                obs_patches.append(mpatches.Polygon(verts))
        if obs_patches:
            pc = PatchCollection(obs_patches, facecolor='#d0d0d4',
                                 edgecolor='#505058', linewidth=0.6, zorder=2)
            ax.add_collection(pc)

        # ── FOV cone ──────────────────────────────────────────────────
        if fov_points and len(fov_points) >= 3:
            ss = getattr(map_config, 'ssaa', 1)
            fov_world = [(p[0] / ss, p[1] / ss) for p in fov_points]
            fov_poly = mpatches.Polygon(
                fov_world, closed=True,
                facecolor=c_def, alpha=0.06, edgecolor='none', zorder=1)
            ax.add_patch(fov_poly)
            center = fov_world[0]
            # FOV边界线: 仅在非全向视野时绘制 (360°视野无边界)
            fov_angle = float(getattr(map_config, 'FOV_ANGLE',
                              getattr(getattr(map_config, 'EnvParameters', None), 'FOV_ANGLE', 360)))
            if fov_angle < 359.0:
                p_left = fov_world[1]
                p_right = fov_world[-1]
                ax.plot([center[0], p_left[0]], [center[1], p_left[1]],
                        color=c_def, alpha=0.3, linewidth=0.6, zorder=1)
                ax.plot([center[0], p_right[0]], [center[1], p_right[1]],
                        color=c_def, alpha=0.3, linewidth=0.6, zorder=1)

        # ── Capture sector (SECTOR with angle, NOT 360 circle!) ──────
        capture_radius = float(getattr(map_config, 'capture_radius', 20.0))
        capture_angle_deg = float(getattr(map_config, 'capture_sector_angle_deg', 30.0))
        def_cx = float(defender['x']) + px * 0.5
        def_cy = float(defender['y']) + px * 0.5
        def_theta = float(defender.get('theta', 0.0))

        # matplotlib Wedge: angles in degrees, counterclockwise in data coords.
        # With inverted y-axis, counterclockwise in data = clockwise on screen,
        # which matches the game's theta convention (clockwise from +x).
        wedge = mpatches.Wedge(
            (def_cx, def_cy), capture_radius,
            def_theta - capture_angle_deg / 2.0,
            def_theta + capture_angle_deg / 2.0,
            facecolor=c_cap, alpha=0.18,
            edgecolor=c_cap, linewidth=1.2, linestyle='--', zorder=2)
        ax.add_patch(wedge)

        # ── Trajectories ─────────────────────────────────────────────
        self._draw_traj(ax, defender_traj, c_def, 'Defender',
                        start_pos=kwargs.get('defender_start_pos'))
        self._draw_traj(ax, attacker_traj, c_atk, 'Attacker',
                        start_pos=kwargs.get('attacker_start_pos'))

        # ── Agent current positions ──────────────────────────────────
        self._draw_agent(ax, defender, c_def, 'Defender')
        self._draw_agent(ax, attacker, c_atk, 'Attacker')

        # ── Target ───────────────────────────────────────────────────
        tx = float(target['x']) + px * 0.5
        ty = float(target['y']) + px * 0.5
        tr = float(getattr(map_config, 'target_radius', 16.0))

        circle_outer = plt.Circle(
            (tx, ty), tr, fill=False,
            edgecolor=c_tgt, linewidth=1.8, linestyle='--', zorder=3)
        circle_inner = plt.Circle(
            (tx, ty), tr * 0.3, fill=True,
            facecolor=c_tgt, edgecolor='none', zorder=3)
        ax.add_patch(circle_outer)
        ax.add_patch(circle_inner)
        ch = tr * 0.5
        ax.plot([tx - ch, tx + ch], [ty, ty],
                color=c_tgt, linewidth=0.8, alpha=0.6, zorder=3)
        ax.plot([tx, tx], [ty - ch, ty + ch],
                color=c_tgt, linewidth=0.8, alpha=0.6, zorder=3)
        ax.annotate('Target', (tx, ty), textcoords='offset points',
                    xytext=(8, 8), fontsize=8, color=c_tgt,
                    fontstyle='italic', zorder=7)

        # ── Step counter ─────────────────────────────────────────────
        step_num = len(defender_traj)
        ax.set_title(f'Step {step_num}', fontsize=12, fontweight='bold', pad=10)

        # ── Legend ────────────────────────────────────────────────────
        handles = ax.get_legend_handles_labels()[0]
        if handles:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.85,
                      edgecolor='#cccccc', fancybox=False)

        # ── Render to numpy array ────────────────────────────────────
        self.fig.canvas.draw()
        buf = np.frombuffer(
            self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        ncols, nrows = self.fig.canvas.get_width_height()
        return buf.reshape((nrows, ncols, 3)).copy()

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_traj(ax, traj, color, label, start_pos=None):
        """Draw full trajectory with markers and arrows (matching PNG style).
        
        Args:
            start_pos: 真实出生点坐标 (x, y)。当轨迹被截断时，
                       traj[0] 不再是出生点，需用此参数绘制固定标记。
        """
        if len(traj) < 2:
            return
        pts = traj
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        n = len(xs)

        # Main line
        ax.plot(xs, ys, color=color, linewidth=1.4,
                alpha=0.75, zorder=4, label=label)

        # Direction arrows — fixed absolute spacing
        _ARROW_SPACING = 20
        for i in range(_ARROW_SPACING, n - 1, _ARROW_SPACING):
            dxx = xs[min(i + 1, n - 1)] - xs[i]
            dyy = ys[min(i + 1, n - 1)] - ys[i]
            if abs(dxx) + abs(dyy) > 0.1:
                ax.annotate('',
                    xy=(xs[i] + dxx * 0.5, ys[i] + dyy * 0.5),
                    xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.2, mutation_scale=8),
                    zorder=5)

        # Start marker (square) — 使用真实出生点
        sx, sy = (start_pos if start_pos else (xs[0], ys[0]))
        ax.plot(sx, sy, 's', color=color, markersize=7,
                markeredgecolor='white', markeredgewidth=1.0, zorder=6)

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_agent(ax, agent, color, role_name):
        """Draw agent circle with heading arrow (academic style)."""
        import matplotlib.patches as mp
        px = float(map_config.pixel_size)
        cx = float(agent['x']) + px * 0.5
        cy = float(agent['y']) + px * 0.5
        radius = float(getattr(map_config, 'agent_radius', 8.0))
        theta_rad = math.radians(float(agent.get('theta', 0.0)))

        circle = mp.Circle(
            (cx, cy), radius,
            facecolor=color, alpha=0.15,
            edgecolor=color, linewidth=1.8, zorder=6)
        ax.add_patch(circle)

        # Heading arrow
        arrow_len = radius * 1.8
        tip_x = cx + arrow_len * math.cos(theta_rad)
        tip_y = cy + arrow_len * math.sin(theta_rad)
        ax.annotate('', xy=(tip_x, tip_y), xytext=(cx, cy),
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.8, mutation_scale=12),
                    zorder=7)

        # Label
        ax.annotate(role_name, (cx, cy), textcoords='offset points',
                    xytext=(8, -10), fontsize=7, color=color,
                    fontstyle='italic', zorder=7)

    # ------------------------------------------------------------------
    def close(self):
        if self.fig is not None:
            self._plt.close(self.fig)
            self.fig = None
            self.ax = None


# Module-level singleton
_mpl_renderer = None


def get_canvas_tad_matplotlib(target, defender, attacker,
                               defender_traj, attacker_traj,
                               fov_points=None, collision_info=None, **kwargs):
    """Render TAD scene using matplotlib (academic style).
    
    Produces frames visually identical to the trajectory PNG/PDF plots.
    Falls back to Pygame renderer if matplotlib is not available.
    """
    global _mpl_renderer
    try:
        import matplotlib   # noqa
    except ImportError:
        return get_canvas_tad(target, defender, attacker,
                              defender_traj, attacker_traj,
                              fov_points=fov_points,
                              collision_info=collision_info)

    if _mpl_renderer is None:
        _mpl_renderer = _MplRenderer()

    return _mpl_renderer.render_frame(
        target, defender, attacker,
        defender_traj, attacker_traj,
        fov_points=fov_points,
        collision_info=collision_info, **kwargs)
