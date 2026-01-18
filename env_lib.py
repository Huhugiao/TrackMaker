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

def reward_calculate(tracker, target, prev_tracker=None, prev_target=None,
                     tracker_collision=False, target_collision=False,
                     sector_captured=False, capture_progress=0, capture_required_steps=0,
                     radar=None):
    info = {
        'capture_progress': int(capture_progress),
        'capture_required_steps': int(capture_required_steps),
        'tracker_collision': bool(tracker_collision),
        'target_collision': bool(target_collision)
    }

    reward = 0.0
    terminated = False

    dx = (tracker['x'] + map_config.pixel_size * 0.5) - (target['x'] + map_config.pixel_size * 0.5)
    dy = (tracker['y'] + map_config.pixel_size * 0.5) - (target['y'] + map_config.pixel_size * 0.5)
    curr_dist = math.hypot(dx, dy)

    if prev_tracker is not None and prev_target is not None:
        p_dx = (prev_tracker['x'] + map_config.pixel_size * 0.5) - (prev_target['x'] + map_config.pixel_size * 0.5)
        p_dy = (prev_tracker['y'] + map_config.pixel_size * 0.5) - (prev_target['y'] + map_config.pixel_size * 0.5)
        prev_dist = math.hypot(p_dx, p_dy)
    else:
        prev_dist = curr_dist

    alpha = 0.05
    reward += alpha * (prev_dist - curr_dist)
    reward -= 0.01

    # 雷达信息可用于未来扩展（如障碍物接近惩罚），当前仅记录最小边缘距离
    if radar is not None and len(radar) > 0:
        max_range = float(getattr(map_config, 'fov_range', 250.0))
        agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
        min_radar_normalized = float(min(radar))
        curr_center_dist = (min_radar_normalized + 1.0) * 0.5 * max_range
        curr_edge_dist = curr_center_dist - agent_radius
        info['min_edge_distance'] = float(curr_edge_dist)

    success_reward = float(getattr(map_config, 'success_reward', 20.0))

    if sector_captured:
        terminated = True
        info['reason'] = 'tracker_caught_target'
        reward += success_reward
    elif tracker_collision:
        terminated = True
        reward -= success_reward
        info['reason'] = 'tracker_collision'

    return float(reward), bool(terminated), False, info

def _draw_grid(surface):
    if pygame is None or not getattr(map_config, 'draw_grid', True):
        return
    ss = getattr(map_config, 'ssaa', 1)
    step = int(map_config.grid_step * ss)
    w, h = surface.get_size()
    for x in range(0, w, step):
        pygame.draw.line(surface, map_config.grid_color, (x, 0), (x, h), 1)
    for y in range(0, h, step):
        pygame.draw.line(surface, map_config.grid_color, (0, y), (w, y), 1)

def _draw_agent(surface, agent, color, role=None):
    if pygame is None:
        return

    ss = float(getattr(map_config, 'ssaa', 1))
    x_world = float(agent['x']) + float(map_config.pixel_size) * 0.5
    y_world = float(agent['y']) + float(map_config.pixel_size) * 0.5
    cx, cy = int(x_world * ss), int(y_world * ss)

    if role == 'target':
        r_world = getattr(map_config, 'target_radius', getattr(map_config, 'agent_radius', 8.0))
    elif role == 'tracker':
        r_world = getattr(map_config, 'tracker_radius', getattr(map_config, 'agent_radius', 8.0))
    else:
        r_world = getattr(map_config, 'agent_radius', 8.0)

    r_i = max(3, int(r_world * ss))

    thickness = max(1, int(1.5 * ss))
    pygame.draw.circle(surface, color[:3], (cx, cy), r_i, thickness)

    theta_deg = agent.get('theta', 0.0)
    theta_rad = math.radians(theta_deg)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    tip_len = r_i * 0.8
    base_len = r_i * 0.4
    wing_len = r_i * 0.5

    p1 = (cx + tip_len * cos_t, cy + tip_len * sin_t)
    p2 = (cx - base_len * cos_t - wing_len * sin_t, cy - base_len * sin_t + wing_len * cos_t)
    p_indent = (cx - (base_len * 0.5) * cos_t, cy - (base_len * 0.5) * sin_t)
    p3 = (cx - base_len * cos_t + wing_len * sin_t, cy - base_len * sin_t - wing_len * cos_t)

    pygame.draw.polygon(surface, color[:3], [p1, p2, p_indent, p3])

def _draw_trail(surface, traj, rgba, width_px):
    if pygame is None or len(traj) < 2:
        return
    ss = getattr(map_config, 'ssaa', 1)
    max_len = getattr(map_config, 'trail_max_len', 600)
    points = traj[-max_len:]
    if len(points) < 2:
        return

    screen_pts = [(int(p[0] * ss), int(p[1] * ss)) for p in points]
    r, g, b = rgba[:3]
    base_alpha = rgba[3] if len(rgba) > 3 else 200
    w = max(int(width_px * ss), 1)

    n = len(screen_pts)
    for i in range(n - 1):
        progress = i / max(1, n - 1)
        alpha = int(base_alpha * (progress ** 1.5))
        if alpha < 10: continue

        color = (r, g, b, alpha)
        start = screen_pts[i]
        end = screen_pts[i+1]
        pygame.draw.line(surface, color, start, end, w)

def _draw_obstacles(surface, exclude_obstacle=None):
    if pygame is None:
        return
    ss = getattr(map_config, 'ssaa', 1)
    for obs in getattr(map_config, 'obstacles', []):
        if exclude_obstacle is not None and obs is exclude_obstacle:
            continue
        color = obs.get('color', (80, 80, 80, 255))
        if obs['type'] == 'rect':
            pygame.draw.rect(
                surface, color,
                (int(obs['x']*ss), int(obs['y']*ss),
                 int(obs['w']*ss), int(obs['h']*ss))
            )
        elif obs['type'] == 'circle':
            pygame.draw.circle(
                surface, color,
                (int(obs['cx']*ss), int(obs['cy']*ss)),
                int(obs['r']*ss)
            )
        elif obs.get('type') == 'segment':
            pygame.draw.line(
                surface, color,
                (int(obs['x1']*ss), int(obs['y1']*ss)),
                (int(obs['x2']*ss), int(obs['y2']*ss)),
                max(1, int(float(obs.get('thick', 8.0))*ss))
            )

def _draw_fov(surface, tracker, fov_points=None):
    if pygame is None or not fov_points or len(fov_points) < 3:
        return

    fill_color = (80, 140, 255, 30)
    pygame.gfxdraw.filled_polygon(surface, fov_points, fill_color)

    outline_color = (80, 140, 255, 200)

    center = fov_points[0]
    p_left = fov_points[1]
    p_right = fov_points[-1]

    c_int = (int(center[0]), int(center[1]))
    pl_int = (int(p_left[0]), int(p_left[1]))
    pr_int = (int(p_right[0]), int(p_right[1]))

    pygame.draw.line(surface, outline_color, c_int, pl_int, 1)
    pygame.draw.line(surface, outline_color, c_int, pr_int, 1)

def _draw_capture_sector(surface, tracker):
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
    num_rays = 20
    for i in range(num_rays + 1):
        ang = heading_rad - half_sector + (2 * half_sector * i / num_rays)
        dist = ray_distance_grid(
            (cx_world, cy_world),
            ang,
            radius,
            padding=0.0
        )
        pts.append((cx + dist * ss * math.cos(ang), cy + dist * ss * math.sin(ang)))

    if len(pts) > 2:
        fill_color = getattr(map_config, 'CAPTURE_SECTOR_COLOR', (80, 200, 120, 40))
        pygame.gfxdraw.filled_polygon(surface, pts, fill_color)

        outline_color = (80, 200, 120, 200)
        pygame.gfxdraw.aapolygon(surface, pts, outline_color)
        pygame.draw.lines(surface, outline_color, True, pts, 1)

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
                         capture_required_steps=0, radar=None):
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


def reward_calculate_protect(defender, attacker, target, prev_defender=None, prev_attacker=None,
                             defender_collision=False, attacker_collision=False,
                             defender_captured=False, attacker_captured=False,
                             capture_progress_defender=0, capture_progress_attacker=0,
                             capture_required_steps=0, radar=None, initial_dist_def_tgt=None):
    """
    奖励函数用于训练保护target的skill

    特点：
    - 无时间惩罚
    - defender捕获attacker: +20
    - attacker到达target: -10（降低惩罚以增加探索）
    - defender碰撞障碍物: -20
    - attacker碰撞障碍物: 不结束episode
    - 超时算defender胜利: +20
    - 有defender-target距离奖励
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

    # 计算defender到target的距离
    dx_def_tgt = (defender['x'] + map_config.pixel_size * 0.5) - (target['x'] + map_config.pixel_size * 0.5)
    dy_def_tgt = (defender['y'] + map_config.pixel_size * 0.5) - (target['y'] + map_config.pixel_size * 0.5)
    curr_dist_def_tgt = math.hypot(dx_def_tgt, dy_def_tgt)

    success_reward = float(getattr(map_config, 'success_reward', 20.0))

    # 距离奖励：按进度比例给奖励，初始距离为基准
    if prev_defender is not None and initial_dist_def_tgt is not None and initial_dist_def_tgt > map_config.capture_radius:
        prev_dx_def_tgt = (prev_defender['x'] + map_config.pixel_size * 0.5) - (target['x'] + map_config.pixel_size * 0.5)
        prev_dy_def_tgt = (prev_defender['y'] + map_config.pixel_size * 0.5) - (target['y'] + map_config.pixel_size * 0.5)
        prev_dist_def_tgt = math.hypot(prev_dx_def_tgt, prev_dy_def_tgt)

        # 计算边界距离（实际距离 - 抓捕半径）
        capture_radius = float(getattr(map_config, 'capture_radius', 20.0))
        prev_boundary_dist = max(0.0, prev_dist_def_tgt - capture_radius)
        curr_boundary_dist = max(0.0, curr_dist_def_tgt - capture_radius)
        initial_boundary_dist = max(0.0, initial_dist_def_tgt - capture_radius)

        # 进度奖励：距离减小的比例 * success_reward
        if initial_boundary_dist > 0:
            distance_progress = (prev_boundary_dist - curr_boundary_dist) / initial_boundary_dist
            distance_reward = distance_progress * success_reward
            reward += distance_reward

    # 终止奖励
    if attacker_captured:
        terminated = True
        info['reason'] = 'attacker_caught_target'
        info['win'] = False
        reward -= 10.0  # 降低惩罚，让defender有更多探索空间
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
                           capture_required_steps=0, radar=None):
    """
    奖励函数用于训练追踪attacker的skill

    特点：
    - 有时间惩罚：每步-0.02（总共约-10）
    - defender捕获attacker: +20
    - attacker到达target: -20
    - defender碰撞障碍物: -20
    - attacker碰撞障碍物: 不结束episode
    - 超时算defender失败: 无额外奖励
    - 使用GRU预测器
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

    # 时间惩罚：每步-0.02
    reward -= 0.02

    success_reward = float(getattr(map_config, 'success_reward', 20.0))

    # 终止奖励
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

def _draw_target(surface, target, color):
    if pygame is None:
        return

    ss = float(getattr(map_config, 'ssaa', 1))
    target_radius = float(getattr(map_config, 'target_radius', 16))
    x_world = float(target['x']) + float(map_config.pixel_size) * 0.5
    y_world = float(target['y']) + float(map_config.pixel_size) * 0.5
    cx, cy = int(x_world * ss), int(y_world * ss)
    r_i = max(3, int(target_radius * ss))

    thickness = max(2, int(2.0 * ss))
    pygame.draw.circle(surface, color[:3], (cx, cy), r_i, thickness)

    pygame.draw.circle(surface, color[:3], (cx, cy), max(1, int(r_i * 0.3)))

def get_canvas_tad(target, defender, attacker, defender_traj, attacker_traj, surface=None, fov_points=None, collision_info=None):
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

    _draw_fov(surface, defender, fov_points)

    _draw_trail(surface, defender_traj, map_config.trail_color_defender, map_config.trail_width)
    _draw_trail(surface, attacker_traj, map_config.trail_color_attacker, map_config.trail_width)

    if collision_info and collision_info.get('collision'):
        collision_color = getattr(map_config, 'COLLISION_DEFENDER_COLOR', (255, 50, 50, 255))
        _draw_agent(surface, defender, collision_color, role='defender')
    else:
        _draw_agent(surface, defender, map_config.defender_color, role='defender')

    if collision_info and collision_info.get('attacker_collision'):
        collision_color = getattr(map_config, 'COLLISION_ATTACKER_COLOR', (180, 50, 255, 255))
        _draw_agent(surface, attacker, collision_color, role='attacker')
    else:
        _draw_agent(surface, attacker, map_config.attacker_color, role='attacker')

    _draw_target(surface, target, map_config.target_color)

    canvas = pygame.transform.smoothscale(surface, (w, h)) if ss > 1 else surface
    return pygame.surfarray.array3d(canvas).swapaxes(0, 1)
