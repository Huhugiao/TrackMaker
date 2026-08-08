"""
Attacker Global Pathfinding Policy

Strategy: 使用全局障碍物信息进行路径规划，直接导航到Target
保留两种策略模式:
- 'default': 默认模式，A*寻路 + 适度避开Defender
- 'evasive': 规避模式，最大化与Defender距离并避开其视野

Observation (72维):
- obs[0]: Attacker 全局 X 坐标（归一化）
- obs[1]: Attacker 全局 Y 坐标（归一化）
- obs[2]: Attacker 朝向（归一化）
- obs[3]: Defender 全局 X 坐标（归一化）
- obs[4]: Defender 全局 Y 坐标（归一化）
- obs[5]: Defender 朝向（归一化）
- obs[6]: Target 全局 X 坐标（归一化）
- obs[7]: Target 全局 Y 坐标（归一化）
- obs[8:72]: 雷达数据（64维）

Action: [angle_delta, speed_normalized]
"""

import numpy as np
import math
import heapq
from typing import Tuple, List, Optional
from configs import map_config
from envs import env_lib

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    njit = None
    _NUMBA_AVAILABLE = False

KP_TURN = 0.8
GRID_SIZE = 8.0
OBSTACLE_PADDING = 12.0
DEFENDER_AVOID_RADIUS = 40.0
DEFENDER_VIEW_DISTANCE = 250.0  # Defender视野距离
DEFENDER_VIEW_ANGLE = 360.0     # Defender视野角度（360°全向）

STRATEGY_CONFIGS = {
    'default':   {'response': 0.9, 'speed_mult': 1.0, 'avoid_view': False},
    'evasive':   {'response': 1.5, 'speed_mult': 0.75, 'avoid_view': True},
}

SUPPORTED_STRATEGIES = ['default', 'evasive']
TRAINING_STRATEGIES = SUPPORTED_STRATEGIES


# ---- Precomputed padded obstacle grid (class-level, built once) ----
_STATIC_GRID = None          # np.ndarray bool, shape (ny, nx)
_STATIC_GRID_CELL = None     # float
_STATIC_GRID_NX = 0
_STATIC_GRID_NY = 0
_STATIC_GRID_KEY = None
_PLANNER_TOKEN_MAX = np.iinfo(np.int32).max - 1
_NEIGHBOR_STEPS = tuple(
    (dx, dy, 1.414 if (dx != 0 and dy != 0) else 1.0)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    if not (dx == 0 and dy == 0)
)

if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _heap_less(f1, x1, y1, f2, x2, y2):
        if f1 < f2:
            return True
        if f1 > f2:
            return False
        if x1 < x2:
            return True
        if x1 > x2:
            return False
        return y1 < y2


    @njit(cache=True)
    def _heap_push(heap_f, heap_x, heap_y, heap_size, f_score, x, y):
        idx = heap_size
        heap_f[idx] = f_score
        heap_x[idx] = x
        heap_y[idx] = y
        heap_size += 1

        while idx > 0:
            parent = (idx - 1) // 2
            if not _heap_less(heap_f[idx], heap_x[idx], heap_y[idx], heap_f[parent], heap_x[parent], heap_y[parent]):
                break
            tmp_f = heap_f[parent]
            tmp_x = heap_x[parent]
            tmp_y = heap_y[parent]
            heap_f[parent] = heap_f[idx]
            heap_x[parent] = heap_x[idx]
            heap_y[parent] = heap_y[idx]
            heap_f[idx] = tmp_f
            heap_x[idx] = tmp_x
            heap_y[idx] = tmp_y
            idx = parent

        return heap_size


    @njit(cache=True)
    def _heap_pop(heap_f, heap_x, heap_y, heap_size):
        root_f = heap_f[0]
        root_x = heap_x[0]
        root_y = heap_y[0]
        heap_size -= 1

        if heap_size > 0:
            heap_f[0] = heap_f[heap_size]
            heap_x[0] = heap_x[heap_size]
            heap_y[0] = heap_y[heap_size]

            idx = 0
            while True:
                left = idx * 2 + 1
                if left >= heap_size:
                    break
                right = left + 1
                smallest = left
                if right < heap_size and _heap_less(
                    heap_f[right], heap_x[right], heap_y[right],
                    heap_f[left], heap_x[left], heap_y[left],
                ):
                    smallest = right
                if not _heap_less(
                    heap_f[smallest], heap_x[smallest], heap_y[smallest],
                    heap_f[idx], heap_x[idx], heap_y[idx],
                ):
                    break

                tmp_f = heap_f[idx]
                tmp_x = heap_x[idx]
                tmp_y = heap_y[idx]
                heap_f[idx] = heap_f[smallest]
                heap_x[idx] = heap_x[smallest]
                heap_y[idx] = heap_y[smallest]
                heap_f[smallest] = tmp_f
                heap_x[smallest] = tmp_x
                heap_y[smallest] = tmp_y
                idx = smallest

        return root_f, root_x, root_y, heap_size


    @njit(cache=True)
    def _plan_path_numba(
        grid,
        gs,
        gnx,
        gny,
        start_x,
        start_y,
        goal_x,
        goal_y,
        has_defender,
        def_gx,
        def_gy,
        avoid_r2,
        planner_marks,
        planner_parent_x,
        planner_parent_y,
        planner_g,
        heap_f,
        heap_x,
        heap_y,
        path_x,
        path_y,
        token,
        max_steps,
    ):
        planner_marks[start_y, start_x] = token
        planner_parent_x[start_y, start_x] = -1
        planner_parent_y[start_y, start_x] = -1
        planner_g[start_y, start_x] = 0.0

        heap_size = 0
        heap_size = _heap_push(heap_f, heap_x, heap_y, heap_size, 0.0, start_x, start_y)
        steps = 0
        half_gs = gs * 0.5

        while heap_size > 0:
            steps += 1
            if steps > max_steps:
                break

            _current_cost, cx0, cy0, heap_size = _heap_pop(heap_f, heap_x, heap_y, heap_size)

            if cx0 == goal_x and cy0 == goal_y:
                path_len = 0
                current_x = cx0
                current_y = cy0

                while planner_parent_x[current_y, current_x] != -1:
                    path_x[path_len] = current_x
                    path_y[path_len] = current_y
                    path_len += 1
                    parent_x = planner_parent_x[current_y, current_x]
                    parent_y = planner_parent_y[current_y, current_x]
                    current_x = parent_x
                    current_y = parent_y

                path_x[path_len] = start_x
                path_y[path_len] = start_y
                path_len += 1

                for idx in range(path_len // 2):
                    rev_idx = path_len - 1 - idx
                    tmp_x = path_x[idx]
                    tmp_y = path_y[idx]
                    path_x[idx] = path_x[rev_idx]
                    path_y[idx] = path_y[rev_idx]
                    path_x[rev_idx] = tmp_x
                    path_y[rev_idx] = tmp_y

                return 1, path_len

            current_g = planner_g[cy0, cx0]
            for dx in (-1, 0, 1):
                nx_ = cx0 + dx
                if nx_ < 0 or nx_ >= gnx:
                    continue
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny_ = cy0 + dy
                    if ny_ < 0 or ny_ >= gny:
                        continue

                    if grid[ny_, nx_]:
                        continue

                    if has_defender:
                        px = nx_ * gs + half_gs
                        py = ny_ * gs + half_gs
                        ddx = px - def_gx
                        ddy = py - def_gy
                        if ddx * ddx + ddy * ddy < avoid_r2:
                            continue

                    move_cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                    tentative_g = current_g + move_cost

                    if planner_marks[ny_, nx_] != token or tentative_g < planner_g[ny_, nx_]:
                        planner_marks[ny_, nx_] = token
                        planner_parent_x[ny_, nx_] = cx0
                        planner_parent_y[ny_, nx_] = cy0
                        planner_g[ny_, nx_] = tentative_g
                        f_score = tentative_g + math.hypot(nx_ - goal_x, ny_ - goal_y)
                        heap_size = _heap_push(heap_f, heap_x, heap_y, heap_size, f_score, nx_, ny_)

        return 0, 0

def _ensure_static_grid(grid_size: float, padding: float, env_width: float, env_height: float):
    """Build a padded grid and refresh it when the randomized map changes."""
    global _STATIC_GRID, _STATIC_GRID_CELL, _STATIC_GRID_NX, _STATIC_GRID_NY, _STATIC_GRID_KEY
    obstacle_signature = tuple(
        tuple(sorted((str(key), repr(value)) for key, value in dict(item).items()))
        for item in getattr(map_config, "obstacles", [])
    )
    cache_key = (
        float(grid_size),
        float(padding),
        float(env_width),
        float(env_height),
        obstacle_signature,
    )
    if _STATIC_GRID is not None and _STATIC_GRID_KEY == cache_key:
        return
    # Ensure obstacle arrays are compiled (build_occupancy must have been called)
    if not getattr(env_lib, '_OBS_COMPILED', False):
        env_lib.build_occupancy()
    nx = int(math.ceil(env_width / grid_size))
    ny = int(math.ceil(env_height / grid_size))
    grid = np.zeros((ny, nx), dtype=np.bool_)
    for iy in range(ny):
        cy = iy * grid_size + grid_size * 0.5
        for ix in range(nx):
            cx = ix * grid_size + grid_size * 0.5
            if env_lib.is_point_blocked(cx, cy, padding=padding):
                grid[iy, ix] = True
    _STATIC_GRID = grid
    _STATIC_GRID_CELL = grid_size
    _STATIC_GRID_NX = nx
    _STATIC_GRID_NY = ny
    _STATIC_GRID_KEY = cache_key


class AttackerGlobalPolicy:
    """
    Attacker全局路径规划策略
    """

    def __init__(
        self,
        env_width: float = 640,
        env_height: float = 640,
        attacker_speed: Optional[float] = None,
        attacker_max_turn: float = 12.0,
        kp_turn: float = KP_TURN,
        grid_size: float = GRID_SIZE,
        obstacle_padding: float = OBSTACLE_PADDING,
        defender_avoid_radius: float = DEFENDER_AVOID_RADIUS,
        defender_view_distance: float = DEFENDER_VIEW_DISTANCE,
        defender_view_angle: float = DEFENDER_VIEW_ANGLE,
        strategy: str = 'default',
        strategy_params: Optional[dict] = None,
    ):
        """
        初始化Attacker全局策略

        Args:
            env_width: 环境宽度
            env_height: 环境高度
            attacker_speed: Attacker最大速度（None时使用map_config.attacker_speed）
            attacker_max_turn: Attacker最大转向角速度（度/步）
            kp_turn: 转向比例系数
            grid_size: 路径规划网格大小
            obstacle_padding: 障碍物膨胀距离
            defender_avoid_radius: 规避defender的安全半径
            defender_view_distance: defender视野距离
            defender_view_angle: defender视野角度
            strategy: 策略名称
            strategy_params: 策略参数字典
        """
        self.env_width = env_width
        self.env_height = env_height
        if attacker_speed is None:
            attacker_speed = float(getattr(map_config, 'attacker_speed', 2.0))
        self.max_speed = float(attacker_speed)
        self.max_turn = attacker_max_turn

        # 策略参数
        self.kp_turn = kp_turn
        self.grid_size = grid_size
        self.obstacle_padding = obstacle_padding
        self.base_defender_avoid_radius = defender_avoid_radius
        self.defender_avoid_radius = defender_avoid_radius
        self.defender_view_distance = defender_view_distance
        self.defender_view_angle = defender_view_angle
        self.strategy_params = strategy_params or {}
        if strategy not in SUPPORTED_STRATEGIES:
            raise ValueError(f'Unsupported attacker strategy: {strategy}. Valid={SUPPORTED_STRATEGIES}')

        # strategy_mode is retained in diagnostics for API compatibility.
        self.strategy_mode = strategy
        self.strategy = strategy

        # 路径规划相关
        self.path = []
        self.current_path_index = 0
        self.last_planned_pos = None
        self.replan_interval = 20
        self.step_count = 0
        
        # Speed multiplier from config
        self._set_active_strategy(self.strategy, reset_state=False)

        # Build static obstacle grid once (shared across all instances)
        _ensure_static_grid(self.grid_size, self.obstacle_padding,
                            self.env_width, self.env_height)
        self._grid = _STATIC_GRID
        self._grid_nx = _STATIC_GRID_NX
        self._grid_ny = _STATIC_GRID_NY
        self._planner_marks = np.zeros((self._grid_ny, self._grid_nx), dtype=np.int32)
        self._planner_parent_x = np.full((self._grid_ny, self._grid_nx), -1, dtype=np.int32)
        self._planner_parent_y = np.full((self._grid_ny, self._grid_nx), -1, dtype=np.int32)
        self._planner_g = np.empty((self._grid_ny, self._grid_nx), dtype=np.float64)
        self._planner_token = 0
        heap_cap = max(16384, self._grid_nx * self._grid_ny * 2)
        self._planner_heap_f = np.empty((heap_cap,), dtype=np.float64)
        self._planner_heap_x = np.empty((heap_cap,), dtype=np.int32)
        self._planner_heap_y = np.empty((heap_cap,), dtype=np.int32)
        self._planner_path_x = np.empty((self._grid_nx * self._grid_ny,), dtype=np.int32)
        self._planner_path_y = np.empty((self._grid_nx * self._grid_ny,), dtype=np.int32)

    def _apply_strategy_config(self):
        """根据策略调整初始参数"""
        self.defender_avoid_radius = self.base_defender_avoid_radius * self.response_intensity
        self.replan_interval = 20

        if self.strategy == 'evasive':
            self.defender_avoid_radius = self.base_defender_avoid_radius * 2.0
            self.replan_interval = 10

    def _set_active_strategy(self, strategy_name: str, reset_state: bool = True):
        self.strategy = strategy_name
        cfg = STRATEGY_CONFIGS.get(self.strategy, STRATEGY_CONFIGS['default'])
        self.speed_mult = cfg.get('speed_mult', 1.0)
        self.response_intensity = cfg.get('response', 0.6)
        self.avoid_view = cfg.get('avoid_view', False)

        if reset_state:
            self.path = []
            self.current_path_index = 0
            self.last_planned_pos = None
            self.last_defender_pos = None
        self._apply_strategy_config()

    def reset(self):
        """重置策略状态"""
        _ensure_static_grid(
            self.grid_size,
            self.obstacle_padding,
            self.env_width,
            self.env_height,
        )
        self._grid = _STATIC_GRID
        self._grid_nx = _STATIC_GRID_NX
        self._grid_ny = _STATIC_GRID_NY
        self.path = []
        self.current_path_index = 0
        self.last_planned_pos = None
        self.last_defender_pos = None
        self.step_count = 0
        self._set_active_strategy(self.strategy, reset_state=False)

    def denormalize_pos(self, norm_x: float, norm_y: float) -> np.ndarray:
        """反归一化位置"""
        x = ((norm_x + 1.0) / 2.0) * self.env_width
        y = ((norm_y + 1.0) / 2.0) * self.env_height
        return np.array([x, y], dtype=np.float32)

    def denormalize_heading(self, norm_heading: float) -> float:
        """
        反归一化朝向（度）
        """
        return (norm_heading + 1.0) * 180.0

    def normalize_angle(self, angle: float) -> float:
        """
        将角度归一化到 [-180, 180]
        """
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def is_pos_blocked(self, x: float, y: float, defender_pos: np.ndarray = None) -> bool:
        """
        检查位置是否被障碍物阻挡
        """
        # 检查静态障碍物
        if env_lib.is_point_blocked(x, y, padding=self.obstacle_padding):
            return True

        # 检查是否太靠近defender
        if defender_pos is not None and self.defender_avoid_radius > 0:
            dist_to_defender = math.hypot(x - defender_pos[0], y - defender_pos[1])
            if dist_to_defender < self.defender_avoid_radius:
                return True

        return False

    def is_in_defender_view(self, pos: np.ndarray, defender_pos: np.ndarray, defender_heading: float = None) -> bool:
        """
        检查某位置是否在defender的视野范围内
        
        Args:
            pos: 待检查的位置
            defender_pos: defender位置
            defender_heading: defender朝向（度），360°视野时不需要
            
        Returns:
            bool: 是否在视野内
        """
        if defender_pos is None:
            return False
            
        dist_to_defender = np.linalg.norm(pos - defender_pos)
        
        # 检查距离
        if dist_to_defender > self.defender_view_distance:
            return False
            
        # 360°全向视野，只需检查距离
        if self.defender_view_angle >= 360:
            return True
            
        # 非全向视野时检查角度
        if defender_heading is not None:
            angle_to_pos = math.degrees(math.atan2(
                pos[1] - defender_pos[1],
                pos[0] - defender_pos[0]
            ))
            angle_diff = abs(self.normalize_angle(angle_to_pos - defender_heading))
            half_view = self.defender_view_angle / 2
            return angle_diff <= half_view
            
        return True

    def get_evasive_target(self, attacker_pos: np.ndarray, defender_pos: np.ndarray, 
                          target_pos: np.ndarray) -> np.ndarray:
        """
        获取规避视野的导航目标点
        优先选择远离defender视野的路径
        """
        dist_to_target = np.linalg.norm(attacker_pos - target_pos)
        if dist_to_target < 50:
            return target_pos
            
        # 如果当前位置在视野外，直接向目标移动
        if not self.is_in_defender_view(attacker_pos, defender_pos):
            return target_pos
            
        # 计算远离defender的方向
        away_dir = attacker_pos - defender_pos
        dist_to_defender = np.linalg.norm(away_dir)
        
        if dist_to_defender < 1e-6:
            # 如果重合，随机选择一个方向
            away_dir = np.array([1.0, 0.0])
        else:
            away_dir = away_dir / dist_to_defender
            
        # 向远离defender的方向移动，超出视野距离
        safe_distance = self.defender_view_distance + 30
        if dist_to_defender < safe_distance:
            evasive_point = defender_pos + away_dir * safe_distance
            evasive_point[0] = np.clip(evasive_point[0], 30, self.env_width - 30)
            evasive_point[1] = np.clip(evasive_point[1], 30, self.env_height - 30)
            return evasive_point
            
        return target_pos

    def plan_path(self, start_pos: np.ndarray, goal_pos: np.ndarray, defender_pos: np.ndarray = None) -> List[np.ndarray]:
        """
        使用A*算法规划从起点到终点的路径
        优化: 使用预计算的静态障碍物网格进行 O(1) 碰撞检测
        """
        gs = self.grid_size
        grid = self._grid
        gnx = self._grid_nx
        gny = self._grid_ny

        start_x = int(np.clip(int(start_pos[0] / gs), 0, gnx - 1))
        start_y = int(np.clip(int(start_pos[1] / gs), 0, gny - 1))
        goal_x = int(np.clip(int(goal_pos[0] / gs), 0, gnx - 1))
        goal_y = int(np.clip(int(goal_pos[1] / gs), 0, gny - 1))

        # Defender avoidance: precompute grid-coord range
        avoid_r2 = 0.0
        def_gx = 0.0
        def_gy = 0.0
        has_defender = defender_pos is not None and self.defender_avoid_radius > 0
        if has_defender:
            avoid_r2 = self.defender_avoid_radius * self.defender_avoid_radius
            def_gx = defender_pos[0]
            def_gy = defender_pos[1]

        self._planner_token += 1
        if self._planner_token > _PLANNER_TOKEN_MAX:
            self._planner_token = 1
            self._planner_marks.fill(0)
        token = self._planner_token

        planner_marks = self._planner_marks
        planner_parent_x = self._planner_parent_x
        planner_parent_y = self._planner_parent_y
        planner_g = self._planner_g
        max_steps = 2000
        half_gs = gs * 0.5

        if _NUMBA_AVAILABLE:
            reached_goal, path_len = _plan_path_numba(
                grid,
                gs,
                gnx,
                gny,
                start_x,
                start_y,
                goal_x,
                goal_y,
                has_defender,
                def_gx,
                def_gy,
                avoid_r2,
                planner_marks,
                planner_parent_x,
                planner_parent_y,
                planner_g,
                self._planner_heap_f,
                self._planner_heap_x,
                self._planner_heap_y,
                self._planner_path_x,
                self._planner_path_y,
                token,
                max_steps,
            )
            if reached_goal:
                path = []
                for idx in range(path_len):
                    path.append(np.array([
                        self._planner_path_x[idx] * gs + half_gs,
                        self._planner_path_y[idx] * gs + half_gs
                    ], dtype=np.float32))
                path[-1] = goal_pos.copy()
                return path

        open_set = [(0.0, start_x, start_y)]
        planner_marks[start_y, start_x] = token
        planner_parent_x[start_y, start_x] = -1
        planner_parent_y[start_y, start_x] = -1
        planner_g[start_y, start_x] = 0.0

        steps = 0

        while open_set:
            steps += 1
            if steps > max_steps:
                break

            _current_cost, cx0, cy0 = heapq.heappop(open_set)

            if cx0 == goal_x and cy0 == goal_y:
                path = []
                current_x = cx0
                current_y = cy0
                while planner_parent_x[current_y, current_x] != -1:
                    pos = np.array([
                        current_x * gs + half_gs,
                        current_y * gs + half_gs
                    ], dtype=np.float32)
                    path.append(pos)
                    parent_x = planner_parent_x[current_y, current_x]
                    parent_y = planner_parent_y[current_y, current_x]
                    current_x = parent_x
                    current_y = parent_y
                start_pos_continuous = np.array([
                    start_x * gs + half_gs,
                    start_y * gs + half_gs
                ], dtype=np.float32)
                path.append(start_pos_continuous)
                path.reverse()
                path[-1] = goal_pos.copy()
                return path

            current_g = planner_g[cy0, cx0]
            for dx, dy, move_cost in _NEIGHBOR_STEPS:
                nx_ = cx0 + dx
                if nx_ < 0 or nx_ >= gnx:
                    continue
                ny_ = cy0 + dy
                if ny_ < 0 or ny_ >= gny:
                    continue

                if grid[ny_, nx_]:
                    continue

                if has_defender:
                    px = nx_ * gs + half_gs
                    py = ny_ * gs + half_gs
                    ddx = px - def_gx
                    ddy = py - def_gy
                    if ddx * ddx + ddy * ddy < avoid_r2:
                        continue

                tentative_g = current_g + move_cost
                if planner_marks[ny_, nx_] != token or tentative_g < planner_g[ny_, nx_]:
                    planner_marks[ny_, nx_] = token
                    planner_parent_x[ny_, nx_] = cx0
                    planner_parent_y[ny_, nx_] = cy0
                    planner_g[ny_, nx_] = tentative_g
                    f_score = tentative_g + math.hypot(
                        nx_ - goal_x,
                        ny_ - goal_y
                    )
                    heapq.heappush(open_set, (f_score, nx_, ny_))

        return [start_pos.copy(), goal_pos.copy()]

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        根据观测计算动作

        Args:
            obs: Attacker观测 (71维)

        Returns:
            action: [angle_delta, speed_normalized]
        """
        # 解析观测
        attacker_pos = self.denormalize_pos(obs[0], obs[1])
        attacker_heading = self.denormalize_heading(obs[2])
        defender_pos = self.denormalize_pos(obs[3], obs[4])
        target_pos = self.denormalize_pos(obs[6], obs[7])

        self.step_count += 1
        
        # 1. 确定导航目标 Goal
        nav_goal = target_pos.copy()
        
        # 根据策略调整导航目标
        if self.strategy == 'evasive':
            nav_goal = self.get_evasive_target(attacker_pos, defender_pos, target_pos)

        # 2. 检查是否需要重新规划路径
        need_replan = False
        if len(self.path) == 0:
            need_replan = True
        elif self.last_planned_pos is None:
            need_replan = True
        elif np.linalg.norm(attacker_pos - self.last_planned_pos) > self.grid_size * 2:
            need_replan = True
        elif self.step_count % self.replan_interval == 0:
            # Only replan on timer if defender moved significantly since last plan
            if self.last_defender_pos is None:
                need_replan = True
            else:
                defender_moved = np.linalg.norm(defender_pos - self.last_defender_pos)
                if defender_moved > self.grid_size:
                    need_replan = True

        if need_replan:
            # 多级避让策略
            dist_defender_to_target = np.linalg.norm(defender_pos - target_pos)
            target_radius = float(getattr(map_config, 'target_radius', 16.0))
            agent_radius = float(getattr(map_config, 'agent_radius', 8.0))
            
            # 基础避让半径
            max_dist_for_full_avoid = 60.0
            if dist_defender_to_target >= max_dist_for_full_avoid:
                base_avoid_radius = self.defender_avoid_radius
            else:
                ratio = dist_defender_to_target / max_dist_for_full_avoid
                base_avoid_radius = agent_radius + ratio * (self.defender_avoid_radius - agent_radius)
            
            # 生成尝试列表
            check_radii = [base_avoid_radius, base_avoid_radius * 0.5, agent_radius, 0]

            original_radius = self.defender_avoid_radius
            self.path = None
            
            for try_radius in check_radii:
                self.defender_avoid_radius = try_radius
                test_path = self.plan_path(
                    attacker_pos, nav_goal, 
                    defender_pos if try_radius > 0 else None
                )
                if len(test_path) > 2:
                    self.path = test_path
                    break
            
            if self.path is None:
                self.path = test_path
            
            self.defender_avoid_radius = original_radius
            self.current_path_index = 0
            self.last_planned_pos = attacker_pos.copy()
            self.last_defender_pos = defender_pos.copy()

        # 3. 寻找当前路径点
        if len(self.path) > 0:
            while (self.current_path_index < len(self.path) and
                   np.linalg.norm(attacker_pos - self.path[self.current_path_index]) < self.grid_size):
                self.current_path_index += 1

            if self.current_path_index >= len(self.path):
                # 已到达终点
                current_goal = nav_goal
            else:
                current_goal = self.path[self.current_path_index]
        else:
            current_goal = nav_goal

        # 4. 执行控制
        # 计算到当前目标点的期望航向
        goal_angle = math.degrees(math.atan2(
            current_goal[1] - attacker_pos[1],
            current_goal[0] - attacker_pos[0]
        ))

        # 计算转向误差
        heading_error = self.normalize_angle(goal_angle - attacker_heading)

        # P控制转向
        turn_delta = heading_error * self.kp_turn

        # 速度：使用策略配置的倍率
        speed = self.max_speed * self.speed_mult
        
        # 限制转向量
        turn_delta = np.clip(turn_delta, -self.max_turn, self.max_turn)
        # 限制速度
        speed = np.clip(speed, 0, self.max_speed)

        # 归一化到动作空间
        angle_delta_norm = turn_delta / self.max_turn
        speed_norm = (speed / self.max_speed) * 2.0 - 1.0

        return np.array([angle_delta_norm, speed_norm], dtype=np.float32)

    def get_action_with_info(self, obs: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        获取动作并返回调试信息

        Args:
            obs: Attacker观测

        Returns:
            action: 动作
            info: 调试信息字典
        """
        
        attacker_pos = self.denormalize_pos(obs[0], obs[1])
        defender_pos = self.denormalize_pos(obs[3], obs[4])
        target_pos = self.denormalize_pos(obs[6], obs[7])
        
        action = self.get_action(obs)

        # 调试信息
        info = {
            'mode': f'global_{self.strategy_mode}',
            'attacker_pos': attacker_pos,
            'defender_pos': defender_pos,
            'target_pos': target_pos,
            'path_length': len(self.path),
            'strategy_mode': self.strategy_mode,
            'active_strategy': self.strategy,
            'strategy': self.strategy,
            'in_defender_view': self.is_in_defender_view(attacker_pos, defender_pos),
        }

        return action, info


def create_attacker_global_policy(**kwargs) -> AttackerGlobalPolicy:
    """
    创建Attacker全局策略的便捷函数

    Args:
        **kwargs: 传递给AttackerGlobalPolicy的参数

    Returns:
        AttackerGlobalPolicy实例
    """
    return AttackerGlobalPolicy(**kwargs)
