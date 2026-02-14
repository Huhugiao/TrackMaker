import os
import math

class EnvParameters:
    N_ACTIONS = 48
    EPISODE_LEN = 449
    NUM_TARGET_POLICIES = 4

    FOV_ANGLE = 360  # 360°全向视野
    FOV_RANGE = 300  # 视野距离300
    RADAR_RAYS = 64
    MAX_UNOBSERVED_STEPS = 80

class ObstacleDensity:
    NONE = "none"
    DENSE = "dense"
    ALL_LEVELS = [NONE, DENSE]

class MapLayout:
    DEFAULT = "default"
    ALL_LAYOUTS = [DEFAULT]

DEFAULT_OBSTACLE_DENSITY = ObstacleDensity.DENSE
current_obstacle_density = DEFAULT_OBSTACLE_DENSITY
DEFAULT_MAP_LAYOUT = MapLayout.DEFAULT
current_map_layout = DEFAULT_MAP_LAYOUT

_jitter_px = 0
_jitter_seed = 0

width = 640
height = 640
map_diagonal = math.hypot(width, height)

pixel_size = 4
defender_speed = 2.6
attacker_speed = 2.0

defender_max_acc = 0.5
attacker_max_acc = 0.6
defender_max_ang_acc = 2.0
attacker_max_ang_acc = 4.0

defender_max_angular_speed = 6.0
attacker_max_angular_speed = 12.0

capture_radius = 20
capture_sector_angle_deg = 30
capture_required_steps = 1
CAPTURE_SECTOR_COLOR = (90, 220, 140, 50)

FAST = os.getenv('SCRIMP_RENDER_MODE', 'fast').lower() == 'fast'
gif_max_side = 800

COLLISION_DEFENDER_COLOR = (255, 50, 50, 255)
COLLISION_ATTACKER_COLOR = (180, 50, 255, 255)
COLLISION_TARGET_COLOR = (255, 255, 0, 255)
COLLISION_POINT_COLOR = (255, 255, 0, 255)
COLLISION_FREEZE_FRAMES = 30

background_color = (255, 255, 255)
grid_color = (245, 245, 245)
grid_step = 64

trail_color_defender = (80, 130, 255, 190)
trail_color_attacker = (255, 120, 120, 190)
trail_color_target = (255, 200, 50, 190)
trail_max_len = 120
trail_width = 2

wall_thickness = 6
OBSTACLE_COLOR = (70, 70, 80, 255)

WALL_OBSTACLES = [
    {'type': 'rect', 'x': 0, 'y': 0, 'w': width, 'h': wall_thickness, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 0, 'y': height - wall_thickness, 'w': width, 'h': wall_thickness, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 0, 'y': 0, 'w': wall_thickness, 'h': height, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': width - wall_thickness, 'y': 0, 'w': wall_thickness, 'h': height, 'color': OBSTACLE_COLOR},
]

_DENSE_OBSTACLES = [
    {'type': 'rect', 'x': 50, 'y': 50, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 195, 'y1': 40, 'x2': 195, 'y2': 100, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 300, 'y': 50, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 445, 'y1': 40, 'x2': 445, 'y2': 100, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 550, 'y': 50, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 40, 'y1': 195, 'x2': 100, 'y2': 195, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 195, 'cy': 195, 'r': 20, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 320, 'y1': 165, 'x2': 320, 'y2': 225, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 445, 'cy': 195, 'r': 20, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 540, 'y1': 195, 'x2': 600, 'y2': 195, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 50, 'y': 300, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 165, 'y1': 320, 'x2': 225, 'y2': 320, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 300, 'y': 300, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 415, 'y1': 320, 'x2': 475, 'y2': 320, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 550, 'y': 300, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 40, 'y1': 445, 'x2': 100, 'y2': 445, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 195, 'cy': 445, 'r': 20, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 320, 'y1': 415, 'x2': 320, 'y2': 475, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'circle', 'cx': 445, 'cy': 445, 'r': 20, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 540, 'y1': 445, 'x2': 600, 'y2': 445, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 50, 'y': 550, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 195, 'y1': 540, 'x2': 195, 'y2': 600, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 300, 'y': 550, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
    {'type': 'segment', 'x1': 445, 'y1': 540, 'x2': 445, 'y2': 600, 'thick': 10, 'color': OBSTACLE_COLOR},
    {'type': 'rect', 'x': 550, 'y': 550, 'w': 40, 'h': 40, 'color': OBSTACLE_COLOR},
]

MAP_LAYOUT_PRESETS = {
    MapLayout.DEFAULT: _DENSE_OBSTACLES,
}

OBSTACLE_PRESETS = {
    ObstacleDensity.NONE: [],
    ObstacleDensity.DENSE: MAP_LAYOUT_PRESETS.get(DEFAULT_MAP_LAYOUT, _DENSE_OBSTACLES),
}

extra_obstacles = []

obstacles = list(WALL_OBSTACLES)
obstacles.extend(OBSTACLE_PRESETS.get(DEFAULT_OBSTACLE_DENSITY, []))

occ_cell = 4.0

def regenerate_obstacles(count=None, seed=None, density_level=None, target_pos=None, map_layout=None):
    global current_obstacle_density, current_map_layout, obstacles
    if density_level is not None:
        current_obstacle_density = density_level
    if map_layout is not None:
        if map_layout not in MapLayout.ALL_LAYOUTS:
            raise ValueError(f"Invalid map layout: {map_layout}")
        current_map_layout = map_layout
    obstacles[:] = list(WALL_OBSTACLES)
    if current_obstacle_density == ObstacleDensity.NONE:
        interior_obstacles = []
    else:
        interior_obstacles = MAP_LAYOUT_PRESETS.get(current_map_layout, MAP_LAYOUT_PRESETS[MapLayout.DEFAULT])
    obstacles.extend(interior_obstacles)
    obstacles.extend(extra_obstacles)

    if target_pos is not None:
        tx, ty = target_pos['x'], target_pos['y']
        target_radius = target_pos.get('r', agent_radius * 2)
        obstacles[:] = [obs for obs in obstacles if not _obstacle_intersects_target(obs, tx, ty, target_radius)]

    return obstacles

def _obstacle_intersects_target(obs, tx, ty, tr):
    if obs['type'] == 'rect':
        closest_x = max(obs['x'], min(tx, obs['x'] + obs['w']))
        closest_y = max(obs['y'], min(ty, obs['y'] + obs['h']))
        distance_x = tx - closest_x
        distance_y = ty - closest_y
        return (distance_x ** 2 + distance_y ** 2) < (tr ** 2)
    elif obs['type'] == 'circle':
        ox, oy = obs['cx'], obs['cy']
        dist = math.hypot(tx - ox, ty - oy)
        return dist < (obs['r'] + tr)
    elif obs['type'] == 'segment':
        x1, y1, x2, y2 = obs['x1'], obs['y1'], obs['x2'], obs['y2']
        px, py = x2 - x1, y2 - y1
        norm = px * px + py * py
        if norm == 0:
            return False
        u = max(0, min(1, ((tx - x1) * px + (ty - y1) * py) / norm))
        closest_x = x1 + u * px
        closest_y = y1 + u * py
        dist = math.hypot(tx - closest_x, ty - closest_y)
        return dist < (obs.get('thick', 8.0) * 0.5 + tr)
    return False

def set_obstacle_density(density_level):
    global current_obstacle_density
    if density_level not in ObstacleDensity.ALL_LEVELS:
        raise ValueError(f"Invalid density level: {density_level}")
    current_obstacle_density = density_level
    regenerate_obstacles(density_level=density_level)


def set_extra_obstacles(obstacle_list):
    global extra_obstacles
    if obstacle_list is None:
        extra_obstacles = []
    else:
        extra_obstacles = [dict(obs) for obs in obstacle_list]

def get_obstacle_density():
    return current_obstacle_density

def set_map_layout(layout_name):
    global current_map_layout
    if layout_name not in MapLayout.ALL_LAYOUTS:
        raise ValueError(f"Invalid map layout: {layout_name}")
    current_map_layout = layout_name
    regenerate_obstacles(map_layout=layout_name)

def get_map_layout():
    return current_map_layout

def set_obstacle_jitter(jitter_px=0, seed=0):
    global _jitter_px, _jitter_seed
    _jitter_px = int(max(0, jitter_px))
    _jitter_seed = int(seed)

base_color_inner = (46, 160, 67)
base_color_outer = (22, 122, 39)
defender_color = (90, 110, 255)
attacker_color = (255, 120, 80)
target_color = (255, 200, 50)
agent_radius = 8
target_radius = 16
base_radius_draw = 12
ssaa = 1
enable_aa = True
draw_grid = False
test_flag = False
mask_flag = False
success_reward = 20
max_loss_step = 50
total_steps = 500
agent_spawn_min_gap = 150.0

def set_render_quality(mode: str):
    global FAST, ssaa, enable_aa, draw_grid, trail_max_len, trail_width
    FAST = (mode.lower() == 'fast')
    if FAST:
        ssaa = 1
        enable_aa = False
        draw_grid = False
        trail_max_len = 80
        trail_width = 1
    else:
        ssaa = 2
        enable_aa = True
        draw_grid = False
        trail_max_len = 400
        trail_width = 2

def set_speeds(defender_spd: float = None, attacker_spd: float = None):
    global defender_speed, attacker_speed
    if defender_spd is not None:
        defender_speed = float(defender_spd)
    if attacker_spd is not None:
        attacker_speed = float(attacker_spd)

def set_capture_params(radius: float = None, sector_angle_deg: float = None, required_steps: int = None):
    global capture_radius, capture_sector_angle_deg, capture_required_steps
    if radius is not None:
        capture_radius = float(radius)
    if sector_angle_deg is not None:
        capture_sector_angle_deg = float(sector_angle_deg)
    if required_steps is not None:
        capture_required_steps = int(required_steps)
