"""Path-based risk metrics for Chapter 2 HRL training.

The actor does not consume these metrics directly. They are used for training
labels, curriculum sampling, logging, and analysis.
"""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np

from configs import map_config
from envs import env_lib


Point = Tuple[float, float]
Cell = Tuple[int, int]

_OCCUPANCY_CACHE_KEY = None
_BLOCKED_GRID_CACHE: Dict[Tuple, np.ndarray] = {}
_ASTAR_CACHE: OrderedDict[Tuple, float] = OrderedDict()
_ASTAR_CACHE_MAX = 200_000


def clear_path_risk_caches() -> None:
    global _OCCUPANCY_CACHE_KEY
    _OCCUPANCY_CACHE_KEY = None
    _BLOCKED_GRID_CACHE.clear()
    _ASTAR_CACHE.clear()


def euclidean_distance(start: Sequence[float], goal: Sequence[float]) -> float:
    return float(math.hypot(float(goal[0]) - float(start[0]), float(goal[1]) - float(start[1])))


def _center_from_state(item: Mapping[str, float]) -> Point:
    if "center_x" in item and "center_y" in item:
        return float(item["center_x"]), float(item["center_y"])
    pixel = float(getattr(map_config, "pixel_size", 4.0))
    return float(item["x"]) + pixel * 0.5, float(item["y"]) + pixel * 0.5


def _grid_to_point(cell: Tuple[int, int], grid_size: float) -> Point:
    return (
        float(cell[0]) * float(grid_size) + float(grid_size) * 0.5,
        float(cell[1]) * float(grid_size) + float(grid_size) * 0.5,
    )


def _obstacle_signature(obstacles: Iterable[Mapping] | None) -> Tuple:
    if obstacles is None:
        return ()
    items = []
    for obs in obstacles:
        if not isinstance(obs, Mapping):
            items.append((repr(obs),))
            continue
        items.append(tuple(sorted((str(k), repr(v)) for k, v in obs.items())))
    return tuple(items)


def _ensure_occupancy(
    *,
    width: float,
    height: float,
    grid_size: float,
    obstacles: Iterable[Mapping] | None,
) -> Tuple:
    global _OCCUPANCY_CACHE_KEY
    obstacle_list = None if obstacles is None else list(obstacles)
    explicit_empty = obstacle_list is not None and not obstacle_list
    obstacle_key = ("default",) if obstacles is None else _obstacle_signature(obstacle_list)
    key = (float(width), float(height), float(grid_size), obstacle_key)
    if _OCCUPANCY_CACHE_KEY != key:
        if explicit_empty:
            env_lib._OCC_GRID = np.zeros(  # noqa: SLF001 - env_lib exposes no empty-obstacle builder.
                (max(1, int(math.ceil(height / grid_size))), max(1, int(math.ceil(width / grid_size)))),
                dtype=np.bool_,
            )
            env_lib._OCC_CELL = grid_size  # noqa: SLF001
            env_lib._OBS_COMPILED = True  # noqa: SLF001
            env_lib._RECT_OBS = np.empty((0, 4), dtype=np.float64)  # noqa: SLF001
            env_lib._CIRCLE_OBS = np.empty((0, 3), dtype=np.float64)  # noqa: SLF001
            env_lib._SEGMENT_OBS = np.empty((0, 5), dtype=np.float64)  # noqa: SLF001
        else:
            env_lib.build_occupancy(width=width, height=height, cell=grid_size, obstacles=obstacle_list)
        _OCCUPANCY_CACHE_KEY = key
        _BLOCKED_GRID_CACHE.clear()
        _ASTAR_CACHE.clear()
    return key


def _cell_from_point(point: Point, *, nx: int, ny: int, grid_size: float) -> Cell:
    return (
        int(np.clip(math.floor(float(point[0]) / grid_size), 0, nx - 1)),
        int(np.clip(math.floor(float(point[1]) / grid_size), 0, ny - 1)),
    )


def _is_blocked(pos: Point, obstacle_padding: float) -> bool:
    return bool(env_lib.is_point_blocked(float(pos[0]), float(pos[1]), padding=float(obstacle_padding)))


def _blocked_grid(
    *,
    width: float,
    height: float,
    grid_size: float,
    obstacle_padding: float,
    occupancy_key: Tuple,
) -> np.ndarray:
    key = (occupancy_key, float(obstacle_padding))
    cached = _BLOCKED_GRID_CACHE.get(key)
    if cached is not None:
        return cached
    nx = max(1, int(math.ceil(width / grid_size)))
    ny = max(1, int(math.ceil(height / grid_size)))
    grid = np.zeros((ny, nx), dtype=np.bool_)
    for y in range(ny):
        for x in range(nx):
            grid[y, x] = _is_blocked(_grid_to_point((x, y), grid_size), obstacle_padding)
    _BLOCKED_GRID_CACHE[key] = grid
    return grid


def _remember_astar(cache_key: Tuple, length: float) -> float:
    _ASTAR_CACHE[cache_key] = float(length)
    if len(_ASTAR_CACHE) > _ASTAR_CACHE_MAX:
        _ASTAR_CACHE.popitem(last=False)
    return float(length)


def astar_path_length(
    start: Sequence[float],
    goal: Sequence[float],
    *,
    width: float | None = None,
    height: float | None = None,
    grid_size: float | None = None,
    obstacle_padding: float = 12.0,
    obstacles: Iterable[Mapping] | None = None,
    rebuild_occupancy: bool = True,
) -> float:
    """Return A* shortest path length in pixels.

    If no grid path is found, this falls back to Euclidean distance so the risk
    metric remains finite during training.
    """
    width = float(width if width is not None else getattr(map_config, "width", 640.0))
    height = float(height if height is not None else getattr(map_config, "height", 640.0))
    grid_size = float(grid_size if grid_size is not None else getattr(map_config, "occ_cell", 4.0))
    start_pt = (float(start[0]), float(start[1]))
    goal_pt = (float(goal[0]), float(goal[1]))

    obstacle_list = None if obstacles is None else list(obstacles)
    occupancy_key = _ensure_occupancy(width=width, height=height, grid_size=grid_size, obstacles=obstacle_list)

    nx = max(1, int(math.ceil(width / grid_size)))
    ny = max(1, int(math.ceil(height / grid_size)))
    start_cell = _cell_from_point(start_pt, nx=nx, ny=ny, grid_size=grid_size)
    goal_cell = _cell_from_point(goal_pt, nx=nx, ny=ny, grid_size=grid_size)

    if start_cell == goal_cell:
        return euclidean_distance(start_pt, goal_pt)

    cache_key = (occupancy_key, float(obstacle_padding), start_cell, goal_cell)
    cached_length = _ASTAR_CACHE.get(cache_key)
    if cached_length is not None:
        _ASTAR_CACHE.move_to_end(cache_key)
        return float(cached_length)

    neighbors = (
        (1, 0, 1.0),
        (-1, 0, 1.0),
        (0, 1, 1.0),
        (0, -1, 1.0),
        (1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (-1, -1, math.sqrt(2.0)),
    )
    open_set = [(0.0, start_cell)]
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    g_score: Dict[Tuple[int, int], float] = {start_cell: 0.0}
    closed = set()
    blocked = _blocked_grid(
        width=width,
        height=height,
        grid_size=grid_size,
        obstacle_padding=obstacle_padding,
        occupancy_key=occupancy_key,
    )

    while open_set:
        _f, current = heapq.heappop(open_set)
        if current in closed:
            continue
        if current == goal_cell:
            cells = [current]
            while current in came_from:
                current = came_from[current]
                cells.append(current)
            cells.reverse()
            points = [start_pt]
            points.extend(_grid_to_point(cell, grid_size) for cell in cells[1:-1])
            points.append(goal_pt)
            return _remember_astar(cache_key, sum(euclidean_distance(a, b) for a, b in zip(points[:-1], points[1:])))

        closed.add(current)
        for dx, dy, move_cost in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor[0] < 0 or neighbor[0] >= nx or neighbor[1] < 0 or neighbor[1] >= ny:
                continue
            if neighbor != goal_cell and bool(blocked[neighbor[1], neighbor[0]]):
                continue
            tentative = g_score[current] + move_cost * grid_size
            if tentative >= g_score.get(neighbor, float("inf")):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            heuristic = euclidean_distance(_grid_to_point(neighbor, grid_size), goal_pt)
            heapq.heappush(open_set, (tentative + heuristic, neighbor))

    return _remember_astar(cache_key, euclidean_distance(start_pt, goal_pt))


def compute_path_risk_metrics(
    *,
    state: Mapping[str, Mapping[str, float]],
    defender_speed: float,
    attacker_speed: float,
    width: float | None = None,
    height: float | None = None,
    grid_size: float | None = None,
    obstacle_padding: float = 12.0,
    obstacles: Iterable[Mapping] | None = None,
    metric: str = "astar",
) -> Dict[str, float]:
    defender = _center_from_state(state["defender"])
    attacker = _center_from_state(state["attacker"])
    target = _center_from_state(state["target"])
    defender_speed = max(1e-6, float(defender_speed))
    attacker_speed = max(1e-6, float(attacker_speed))

    selected = str(metric).strip().lower()
    if selected not in {"astar", "euclidean"}:
        raise ValueError(f"metric must be 'astar' or 'euclidean', got {metric!r}")

    euclidean_da = euclidean_distance(defender, attacker)
    euclidean_at = euclidean_distance(attacker, target)
    obstacle_list = list(obstacles) if obstacles is not None else None
    if selected == "astar":
        astar_da = astar_path_length(
            defender,
            attacker,
            width=width,
            height=height,
            grid_size=grid_size,
            obstacle_padding=obstacle_padding,
            obstacles=obstacle_list,
            rebuild_occupancy=True,
        )
        astar_at = astar_path_length(
            attacker,
            target,
            width=width,
            height=height,
            grid_size=grid_size,
            obstacle_padding=obstacle_padding,
            obstacles=obstacle_list,
            rebuild_occupancy=False,
        )
    else:
        astar_da = euclidean_da
        astar_at = euclidean_at

    euclidean_margin = euclidean_at / attacker_speed - euclidean_da / defender_speed
    astar_margin = astar_at / attacker_speed - astar_da / defender_speed
    selected_da = astar_da if selected == "astar" else euclidean_da
    selected_at = astar_at if selected == "astar" else euclidean_at
    selected_margin = astar_margin if selected == "astar" else euclidean_margin

    return {
        "defender_attacker": float(selected_da),
        "attacker_target": float(selected_at),
        "margin": float(selected_margin),
        "astar_defender_attacker": float(astar_da),
        "astar_attacker_target": float(astar_at),
        "astar_margin": float(astar_margin),
        "euclidean_defender_attacker": float(euclidean_da),
        "euclidean_attacker_target": float(euclidean_at),
        "euclidean_margin": float(euclidean_margin),
        "detour_defender_attacker": float(astar_da / max(euclidean_da, 1e-6)),
        "detour_attacker_target": float(astar_at / max(euclidean_at, 1e-6)),
    }
