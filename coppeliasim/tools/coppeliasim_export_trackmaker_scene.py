#!/usr/bin/env python3
"""Export the TrackMaker map as a CoppeliaSim .ttt scene via an internal add-on."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs import map_config


DEFAULT_COPPELIASIM_ROOT = Path.home() / "opt/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/coppeliasim/trackmaker_scene.ttt"
DEFAULT_TURTLEBOT4_XACRO = Path("/opt/ros/humble/share/turtlebot4_description/urdf/standard/turtlebot4.urdf.xacro")
DEFAULT_TURTLEBOT4_URDF = PROJECT_ROOT / "coppeliasim/urdf/turtlebot4_standard.urdf"
DEFAULT_ROS_SHARE = Path("/opt/ros/humble/share")
DEFAULT_ROS_STUB_PREFIX = PROJECT_ROOT / "outputs/ros_stub"
DEFAULT_SIM_STEP_DT = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coppeliasim-root", type=Path, default=Path(os.getenv("COPPELIASIM_ROOT", DEFAULT_COPPELIASIM_ROOT)))
    parser.add_argument("--save", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scale", type=float, default=0.01, help="CoppeliaSim meters per TrackMaker pixel.")
    parser.add_argument("--sim-step-dt", type=float, default=DEFAULT_SIM_STEP_DT, help="CoppeliaSim seconds represented by one Gym step.")
    parser.add_argument("--height", type=float, default=0.18, help="Obstacle height in meters.")
    parser.add_argument("--density", choices=map_config.ObstacleDensity.ALL_LEVELS, default=map_config.ObstacleDensity.DENSE)
    parser.add_argument("--obstacle-seed", type=int, default=20260325)
    parser.add_argument("--spawn-seed", type=int, default=20260326)
    parser.add_argument("--jitter-px", type=int, default=15, help="Obstacle jitter in TrackMaker pixels.")
    parser.add_argument(
        "--robot-radius-px",
        type=float,
        default=float(getattr(map_config, "agent_radius", 8.0)),
        help="Gym agent footprint radius in TrackMaker pixels.",
    )
    parser.add_argument(
        "--target-radius-px",
        type=float,
        default=float(getattr(map_config, "target_radius", 16.0)),
        help="Gym target radius in TrackMaker pixels.",
    )
    parser.add_argument(
        "--robot-source",
        choices=("official", "geometry"),
        default="official",
        help="Use official TurtleBot4 URDF meshes or the lightweight generated geometry fallback.",
    )
    parser.add_argument("--turtlebot4-xacro", type=Path, default=DEFAULT_TURTLEBOT4_XACRO)
    parser.add_argument("--turtlebot4-urdf", type=Path, default=DEFAULT_TURTLEBOT4_URDF)
    parser.add_argument("--ros-share", type=Path, default=DEFAULT_ROS_SHARE, help="ROS share directory used for package:// mesh replacement.")
    parser.add_argument("--ros-stub-prefix", type=Path, default=DEFAULT_ROS_STUB_PREFIX)
    parser.add_argument("--skip-urdf-regenerate", action="store_true", help="Reuse --turtlebot4-urdf instead of regenerating it from xacro.")
    parser.add_argument("--official-robot-scale", type=float, default=None, help="Override official TurtleBot4 mesh scale.")
    parser.add_argument("--preserve-official-robot-size", action="store_true", help="Keep the imported TurtleBot4 at its original URDF size.")
    parser.add_argument("--write-manifest-only", action="store_true", help="Write the scene JSON manifest without launching CoppeliaSim.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--show-gui", action="store_true", help="Do not pass CoppeliaSim's emulated-headless flag.")
    return parser.parse_args()


def lua_literal(value: Any) -> str:
    if value is None:
        return "nil"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(float(value)) if isinstance(value, float) else str(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, dict):
        parts = []
        for key in sorted(value):
            if key == "color":
                continue
            parts.append(f"[{lua_literal(str(key))}] = {lua_literal(value[key])}")
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, (list, tuple)):
        return "{" + ", ".join(lua_literal(item) for item in value) + "}"
    raise TypeError(f"Unsupported Lua literal type: {type(value)!r}")


def _write_text_if_changed(path: Path, text: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def prepare_turtlebot4_urdf(args: argparse.Namespace) -> Path | None:
    if args.robot_source != "official":
        return None

    urdf_path = args.turtlebot4_urdf.resolve()
    if args.skip_urdf_regenerate and urdf_path.is_file():
        return write_coppeliasim_urdf(urdf_path, args.ros_share.resolve())

    xacro_path = args.turtlebot4_xacro.resolve()
    if not xacro_path.is_file():
        raise FileNotFoundError(f"TurtleBot4 xacro not found: {xacro_path}")
    if shutil.which("xacro") is None:
        raise RuntimeError("xacro is not installed in the active environment; run `pip install xacro` in lnenv.")

    stub_prefix = args.ros_stub_prefix.resolve()
    _write_text_if_changed(
        stub_prefix / "share/ament_index/resource_index/packages/irobot_create_control",
        str(stub_prefix) + "\n",
    )
    _write_text_if_changed(
        stub_prefix / "share/irobot_create_control/config/control.yaml",
        "# Placeholder used only to expand TurtleBot4 xacro for CoppeliaSim import.\n"
        "controller_manager:\n"
        "  ros__parameters: {}\n",
    )

    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    ros_setup = Path("/opt/ros/humble/setup.bash")
    env = os.environ.copy()
    if ros_setup.is_file():
        command = (
            f"source {shlex_quote(str(ros_setup))} && "
            f"export AMENT_PREFIX_PATH={shlex_quote(str(stub_prefix))}:$AMENT_PREFIX_PATH && "
            f"xacro {shlex_quote(str(xacro_path))}"
        )
        result = subprocess.run(
            ["bash", "-lc", command],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    else:
        env["AMENT_PREFIX_PATH"] = f"{stub_prefix}:{env.get('AMENT_PREFIX_PATH', '')}"
        result = subprocess.run(
            ["xacro", str(xacro_path)],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate TurtleBot4 URDF from xacro:\n{result.stderr.strip()}")
    if "xacro:" in result.stdout or "$(" in result.stdout or "${" in result.stdout:
        raise RuntimeError("Generated TurtleBot4 URDF still contains xacro macros or substitutions.")

    urdf_path.write_text(result.stdout, encoding="utf-8")
    return write_coppeliasim_urdf(urdf_path, args.ros_share.resolve())


def shlex_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _xml_tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _resolve_package_uri(uri: str, ros_share: Path) -> str:
    if not uri.startswith("package://"):
        return uri
    return str((ros_share / uri.removeprefix("package://")).resolve())


def _rewrite_package_mesh_filenames(root: ET.Element, ros_share: Path) -> None:
    for elem in root.iter():
        if _xml_tag_name(elem.tag) != "mesh":
            continue
        filename = elem.attrib.get("filename")
        if filename:
            elem.set("filename", _resolve_package_uri(filename, ros_share))


def write_coppeliasim_urdf(raw_urdf_path: Path, ros_share: Path = DEFAULT_ROS_SHARE) -> Path:
    tree = ET.parse(raw_urdf_path)
    root = tree.getroot()
    remove_top_level = {"gazebo", "ros2_control", "transmission"}
    for child in list(root):
        if _xml_tag_name(child.tag) in remove_top_level:
            root.remove(child)
    _rewrite_package_mesh_filenames(root, ros_share.resolve())

    clean_urdf_path = raw_urdf_path.with_suffix(".coppeliasim.urdf")
    ET.indent(tree, space="  ")
    tree.write(clean_urdf_path, encoding="utf-8", xml_declaration=True)
    return clean_urdf_path


def _distance_point_to_segment(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / denom))
    cx = x1 + t * dx
    cy = y1 + t * dy
    return math.hypot(px - cx, py - cy)


def _point_hits_obstacle(px: float, py: float, radius: float, obs: dict[str, Any]) -> bool:
    kind = obs.get("type")
    if kind == "rect":
        x1 = float(obs["x"])
        y1 = float(obs["y"])
        x2 = x1 + float(obs["w"])
        y2 = y1 + float(obs["h"])
        cx = max(x1, min(px, x2))
        cy = max(y1, min(py, y2))
        return math.hypot(px - cx, py - cy) < radius
    if kind == "circle":
        return math.hypot(px - float(obs["cx"]), py - float(obs["cy"])) < (radius + float(obs["r"]))
    if kind == "segment":
        distance = _distance_point_to_segment(
            px,
            py,
            float(obs["x1"]),
            float(obs["y1"]),
            float(obs["x2"]),
            float(obs["y2"]),
        )
        return distance < (radius + float(obs.get("thick", 8.0)) * 0.5)
    return False


def _is_clear(px: float, py: float, radius: float, obstacles: list[dict[str, Any]]) -> bool:
    if px < radius or px > float(map_config.width) - radius:
        return False
    if py < radius or py > float(map_config.height) - radius:
        return False
    return not any(_point_hits_obstacle(px, py, radius, obs) for obs in obstacles)


def _sample_point(
    rng: random.Random,
    obstacles: list[dict[str, Any]],
    radius: float,
    margin: float,
    attempts: int = 5000,
) -> tuple[float, float]:
    lo_x = max(margin, radius)
    hi_x = float(map_config.width) - max(margin, radius)
    lo_y = max(margin, radius)
    hi_y = float(map_config.height) - max(margin, radius)
    for _ in range(attempts):
        px = rng.uniform(lo_x, hi_x)
        py = rng.uniform(lo_y, hi_y)
        if _is_clear(px, py, radius, obstacles):
            return px, py
    raise RuntimeError(f"Failed to sample clear point with radius={radius}")


def _sample_scene_entities(
    obstacles: list[dict[str, Any]],
    spawn_seed: int,
    robot_radius_px: float,
    target_radius_px: float,
) -> dict[str, dict[str, float]]:
    rng = random.Random(int(spawn_seed))
    target_radius = float(target_radius_px)
    min_agent_gap = float(getattr(map_config, "agent_spawn_min_gap", 150.0))
    min_target_gap = 80.0
    defender_speed = float(getattr(map_config, "defender_speed", 2.6))
    attacker_speed = float(getattr(map_config, "attacker_speed", 2.0))

    defender = None
    attacker = None
    target = None
    for _ in range(300):
        dcx, dcy = _sample_point(rng, obstacles, robot_radius_px, margin=40.0)
        defender = {"cx": dcx, "cy": dcy, "theta": rng.uniform(0.0, 360.0)}

        for _ in range(1000):
            acx, acy = _sample_point(rng, obstacles, robot_radius_px, margin=40.0)
            if math.hypot(acx - dcx, acy - dcy) >= min_agent_gap:
                attacker = {"cx": acx, "cy": acy, "theta": rng.uniform(0.0, 360.0)}
                break
        if attacker is None:
            continue

        for _ in range(1500):
            tcx, tcy = _sample_point(rng, obstacles, target_radius, margin=55.0)
            d_dt = math.hypot(tcx - dcx, tcy - dcy)
            d_at = math.hypot(tcx - attacker["cx"], tcy - attacker["cy"])
            if d_dt < min_target_gap or d_at < min_target_gap:
                continue
            if d_at * defender_speed <= d_dt * attacker_speed:
                continue
            target = {"cx": tcx, "cy": tcy, "theta": 0.0}
            break

        if target is not None:
            return {"defender": defender, "attacker": attacker, "target": target}

    raise RuntimeError("Failed to sample non-overlapping defender/attacker/target positions.")


def prepare_scene_data(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], dict[str, Any]]:
    map_config.set_obstacle_randomization(enabled=True, jitter_px=args.jitter_px, seed=args.obstacle_seed)
    map_config.regenerate_obstacles(density_level=args.density, seed=args.obstacle_seed, target_pos=None)
    obstacles = [dict(obs) for obs in getattr(map_config, "obstacles", [])]
    entities = _sample_scene_entities(obstacles, args.spawn_seed, args.robot_radius_px, args.target_radius_px)

    target_radius = float(args.target_radius_px)
    map_config.regenerate_obstacles(
        density_level=args.density,
        seed=args.obstacle_seed,
        target_pos={"x": entities["target"]["cx"], "y": entities["target"]["cy"], "r": target_radius},
    )
    obstacles = [dict(obs) for obs in getattr(map_config, "obstacles", [])]

    for name, radius in (("defender", args.robot_radius_px), ("attacker", args.robot_radius_px), ("target", target_radius)):
        entity = entities[name]
        if not _is_clear(entity["cx"], entity["cy"], float(radius), obstacles):
            raise RuntimeError(f"Sampled {name} overlaps an obstacle after regeneration: {entity}")

    meta = {
        "density": args.density,
        "obstacle_seed": int(args.obstacle_seed),
        "spawn_seed": int(args.spawn_seed),
        "jitter_px": int(args.jitter_px),
        "robot_radius_px": float(args.robot_radius_px),
        "target_radius_px": float(args.target_radius_px),
        "pixel_size_px": float(getattr(map_config, "pixel_size", 4.0)),
        "capture_radius_px": float(getattr(map_config, "capture_radius", 20.0)),
        "capture_sector_angle_deg": float(getattr(map_config, "capture_sector_angle_deg", 30.0)),
        "defender_speed_px_per_step": float(getattr(map_config, "defender_speed", 2.6)),
        "attacker_speed_px_per_step": float(getattr(map_config, "attacker_speed", 2.0)),
        "defender_max_turn_deg_per_step": float(getattr(map_config, "defender_max_angular_speed", 6.0)),
        "attacker_max_turn_deg_per_step": float(getattr(map_config, "attacker_max_angular_speed", 12.0)),
        "attacker_max_acc_px_per_step": float(getattr(map_config, "attacker_max_acc", 0.6)),
        "scale_m_per_px": float(args.scale),
        "sim_step_dt": float(args.sim_step_dt),
        "defender_speed_m_per_s": float(getattr(map_config, "defender_speed", 2.6)) * float(args.scale) / float(args.sim_step_dt),
        "attacker_speed_m_per_s": float(getattr(map_config, "attacker_speed", 2.0)) * float(args.scale) / float(args.sim_step_dt),
        "defender_max_turn_rad_per_s": math.radians(float(getattr(map_config, "defender_max_angular_speed", 6.0))) / float(args.sim_step_dt),
        "attacker_max_turn_rad_per_s": math.radians(float(getattr(map_config, "attacker_max_angular_speed", 12.0))) / float(args.sim_step_dt),
        "robot_source": args.robot_source,
    }
    return obstacles, entities, meta


def build_scene_manifest(
    scale: float,
    height: float,
    obstacles: list[dict[str, Any]],
    entities: dict[str, dict[str, float]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    return {
        "map": {
            "width_px": int(map_config.width),
            "height_px": int(map_config.height),
            "scale_m_per_px": float(scale),
            "obstacle_height_m": float(height),
        },
        "meta": dict(meta),
        "entities": {
            name: {
                "center_px": [float(entity["cx"]), float(entity["cy"])],
                "theta_deg": float(entity.get("theta", 0.0)),
            }
            for name, entity in entities.items()
        },
        "objects": {
            "defender": {"alias": "TrackMaker_defender_turtlebot4"},
            "attacker": {"alias": "TrackMaker_attacker_turtlebot4"},
            "target": {"alias": "TrackMaker_target"},
            "defender_footprint": {"alias": "TrackMaker_defender_turtlebot4_gym_footprint"},
            "attacker_footprint": {"alias": "TrackMaker_attacker_turtlebot4_gym_footprint"},
            "defender_capture_radius": {"alias": "TrackMaker_defender_turtlebot4_capture_radius"},
            "attacker_capture_radius": {"alias": "TrackMaker_attacker_turtlebot4_capture_radius"},
        },
        "obstacles": [dict(obs) for obs in obstacles],
    }


def write_scene_manifest(
    path: Path,
    scale: float,
    height: float,
    obstacles: list[dict[str, Any]],
    entities: dict[str, dict[str, float]],
    meta: dict[str, Any],
) -> None:
    manifest = build_scene_manifest(scale, height, obstacles, entities, meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def build_addon_script(
    output: Path,
    status: Path,
    scale: float,
    height: float,
    sim_step_dt: float,
    official_robot_scale: float | None,
    preserve_official_robot_size: bool,
    robot_source: str,
    turtlebot4_urdf: Path | None,
    ros_share: Path,
    obstacles: list[dict[str, Any]],
    entities: dict[str, dict[str, float]],
    meta: dict[str, Any],
) -> str:
    return f"""sim = require 'sim'

function sysCall_info()
    return {{autoStart = true, menu = 'TrackMaker\\nBuild scene'}}
end

local outputPath = {lua_literal(str(output))}
local statusPath = {lua_literal(str(status))}
local mapWidth = {float(map_config.width)}
local mapHeight = {float(map_config.height)}
local scale = {float(scale)}
local obstacleHeight = {float(height)}
local simStepDt = {float(sim_step_dt)}
local officialRobotScaleOverride = {lua_literal(official_robot_scale)}
local preserveOfficialRobotSize = {lua_literal(bool(preserve_official_robot_size))}
local robotSource = {lua_literal(robot_source)}
local turtlebot4Urdf = {lua_literal(str(turtlebot4_urdf) if turtlebot4_urdf else "")}
local rosShareReplacement = {lua_literal(str(ros_share) + "/")}
local obstacles = {lua_literal(obstacles)}
local entities = {lua_literal(entities)}
local exportMeta = {lua_literal(meta)}

local function log(message)
    sim.addLog(sim.verbosity_scriptinfos, '[TrackMaker] ' .. tostring(message))
end

local function writeStatus(message)
    local f = io.open(statusPath, 'w')
    if f then
        f:write(tostring(message), '\\n')
        f:close()
    end
end

local function pxToWorld(px, py)
    return {{
        (px - mapWidth * 0.5) * scale,
        (mapHeight * 0.5 - py) * scale,
        0.0
    }}
end

local function setAlias(handle, alias)
    pcall(sim.setObjectAlias, handle, alias)
end

local function setColor(handle, color)
    pcall(sim.setShapeColor, handle, nil, sim.colorcomponent_ambient_diffuse, color)
end

local function setObjectTransparency(handle, alpha)
    pcall(sim.setShapeColor, handle, nil, sim.colorcomponent_transparency, {{1.0 - (alpha or 0.35)}})
    pcall(sim.setObjectInt32Param, handle, sim.objintparam_visibility_layer, 1)
end

local function makeCuboid(alias, centerPx, sizePx, zHeight, color)
    local sx = math.max(sizePx[1] * scale, 0.001)
    local sy = math.max(sizePx[2] * scale, 0.001)
    local handle = sim.createPrimitiveShape(sim.primitiveshape_cuboid, {{sx, sy, zHeight}}, 0)
    local pos = pxToWorld(centerPx[1], centerPx[2])
    sim.setObjectPosition(handle, {{pos[1], pos[2], zHeight * 0.5}})
    setAlias(handle, alias)
    setColor(handle, color)
    return handle
end

local function makeCylinder(alias, centerPx, radiusPx, zHeight, color)
    local diameter = math.max(radiusPx * 2.0 * scale, 0.001)
    local handle = sim.createPrimitiveShape(sim.primitiveshape_cylinder, {{diameter, diameter, zHeight}}, 0)
    local pos = pxToWorld(centerPx[1], centerPx[2])
    sim.setObjectPosition(handle, {{pos[1], pos[2], zHeight * 0.5}})
    setAlias(handle, alias)
    setColor(handle, color)
    return handle
end

local function makeDisc(alias, centerPx, radiusPx, zHeight, color, alpha)
    local handle = makeCylinder(alias, centerPx, radiusPx, zHeight, color)
    pcall(sim.setObjectInt32Param, handle, sim.shapeintparam_static, 1)
    pcall(sim.setObjectInt32Param, handle, sim.shapeintparam_respondable, 0)
    setObjectTransparency(handle, alpha or 0.35)
    return handle
end

local function makePart(alias, primitive, size, localPos, localOri, color, parent)
    local handle = sim.createPrimitiveShape(primitive, size, 0)
    sim.setObjectPosition(handle, localPos, parent or -1)
    if localOri then
        sim.setObjectOrientation(handle, localOri, parent or -1)
    end
    setAlias(handle, alias)
    setColor(handle, color)
    if parent then
        sim.setObjectParent(handle, parent, true)
    end
    return handle
end

local function setModelStatic(root)
    local shapes = sim.getObjectsInTree(root, sim.sceneobject_shape, 0)
    for _, h in ipairs(shapes) do
        pcall(sim.setObjectInt32Param, h, sim.shapeintparam_static, 1)
        pcall(sim.setObjectInt32Param, h, sim.shapeintparam_respondable, 0)
    end
end

local function getTreeWorldAabb(root)
    local shapes = sim.getObjectsInTree(root, sim.sceneobject_shape, 0)
    local minX, minY, minZ = math.huge, math.huge, math.huge
    local maxX, maxY, maxZ = -math.huge, -math.huge, -math.huge
    local count = 0
    for _, h in ipairs(shapes) do
        local ok, bx1 = pcall(sim.getObjectFloatParam, h, sim.objfloatparam_objbbox_min_x)
        if ok then
            local by1 = sim.getObjectFloatParam(h, sim.objfloatparam_objbbox_min_y)
            local bz1 = sim.getObjectFloatParam(h, sim.objfloatparam_objbbox_min_z)
            local bx2 = sim.getObjectFloatParam(h, sim.objfloatparam_objbbox_max_x)
            local by2 = sim.getObjectFloatParam(h, sim.objfloatparam_objbbox_max_y)
            local bz2 = sim.getObjectFloatParam(h, sim.objfloatparam_objbbox_max_z)
            local matrix = sim.getObjectMatrix(h, -1)
            for _, x in ipairs({{bx1, bx2}}) do
                for _, y in ipairs({{by1, by2}}) do
                    for _, z in ipairs({{bz1, bz2}}) do
                        local p = sim.multiplyVector(matrix, {{x, y, z}})
                        minX = math.min(minX, p[1])
                        minY = math.min(minY, p[2])
                        minZ = math.min(minZ, p[3])
                        maxX = math.max(maxX, p[1])
                        maxY = math.max(maxY, p[2])
                        maxZ = math.max(maxZ, p[3])
                    end
                end
            end
            count = count + 1
        end
    end
    return {{minX, minY, minZ}}, {{maxX, maxY, maxZ}}, count
end

local function getObjectsWorldAabb(handles)
    local minX, minY, minZ = math.huge, math.huge, math.huge
    local maxX, maxY, maxZ = -math.huge, -math.huge, -math.huge
    local count = 0
    for _, root in ipairs(handles) do
        local minB, maxB, c = getTreeWorldAabb(root)
        if c > 0 then
            minX = math.min(minX, minB[1])
            minY = math.min(minY, minB[2])
            minZ = math.min(minZ, minB[3])
            maxX = math.max(maxX, maxB[1])
            maxY = math.max(maxY, maxB[2])
            maxZ = math.max(maxZ, maxB[3])
            count = count + c
        end
    end
    return {{minX, minY, minZ}}, {{maxX, maxY, maxZ}}, count
end

local function alignTreeBottom(root)
    local minB, maxB, count = getTreeWorldAabb(root)
    if count > 0 and minB[3] < 0.0 then
        local p = sim.getObjectPosition(root, -1)
        sim.setObjectPosition(root, {{p[1], p[2], p[3] - minB[3]}})
    end
end

local function scaleOfficialRobotToGymFootprint(handles)
    if preserveOfficialRobotSize then
        return 1.0
    end
    local factor = officialRobotScaleOverride
    if factor == nil then
        local minB, maxB, count = getObjectsWorldAabb(handles)
        if count <= 0 then
            return 1.0
        end
        local widthX = math.max(0.001, maxB[1] - minB[1])
        local widthY = math.max(0.001, maxB[2] - minB[2])
        local currentDiameter = math.max(widthX, widthY)
        local desiredDiameter = math.max(0.001, exportMeta.robot_radius_px * 2.0 * scale)
        factor = desiredDiameter / currentDiameter
    end
    sim.scaleObjects(handles, factor, true)
    return factor
end

local function addRobotReferenceGeometry(alias, entity)
    local footprint = makeDisc(
        alias .. '_gym_footprint',
        {{entity.cx, entity.cy}},
        exportMeta.robot_radius_px,
        0.012,
        {{0.1, 0.45, 1.0}},
        0.18
    )
    local capture = makeDisc(
        alias .. '_capture_radius',
        {{entity.cx, entity.cy}},
        exportMeta.capture_radius_px,
        0.006,
        {{0.1, 0.85, 0.3}},
        0.12
    )
    pcall(sim.setObjectInt32Param, footprint, sim.objintparam_visibility_layer, 1)
    pcall(sim.setObjectInt32Param, capture, sim.objintparam_visibility_layer, 1)
end

local function createGeometryTurtleBot4(alias, entity, color)
    local pos = pxToWorld(entity.cx, entity.cy)
    local yaw = -math.rad(entity.theta or 0.0)

    local root = sim.createDummy(0.06)
    sim.setObjectPosition(root, {{pos[1], pos[2], 0.02}})
    sim.setObjectOrientation(root, {{0.0, 0.0, yaw}})
    setAlias(root, alias)

    local body = makePart(alias .. '_create3_base', sim.primitiveshape_cylinder, {{0.34, 0.34, 0.08}}, {{0.0, 0.0, 0.05}}, nil, color, root)
    local shell = makePart(alias .. '_shell', sim.primitiveshape_cylinder, {{0.28, 0.28, 0.07}}, {{0.0, 0.0, 0.12}}, nil, {{0.06, 0.07, 0.08}}, root)
    local tower = makePart(alias .. '_tower', sim.primitiveshape_cuboid, {{0.16, 0.10, 0.16}}, {{-0.03, 0.0, 0.23}}, nil, {{0.12, 0.12, 0.12}}, root)
    local lidar = makePart(alias .. '_rplidar', sim.primitiveshape_cylinder, {{0.10, 0.10, 0.035}}, {{-0.04, 0.0, 0.33}}, nil, {{0.02, 0.02, 0.02}}, root)
    local camera = makePart(alias .. '_oakd_camera', sim.primitiveshape_cuboid, {{0.06, 0.12, 0.045}}, {{0.15, 0.0, 0.22}}, nil, {{0.02, 0.02, 0.025}}, root)
    local leftWheel = makePart(alias .. '_left_wheel', sim.primitiveshape_cylinder, {{0.07, 0.07, 0.035}}, {{0.02, 0.18, 0.055}}, {{math.pi * 0.5, 0.0, 0.0}}, {{0.015, 0.015, 0.015}}, root)
    local rightWheel = makePart(alias .. '_right_wheel', sim.primitiveshape_cylinder, {{0.07, 0.07, 0.035}}, {{0.02, -0.18, 0.055}}, {{math.pi * 0.5, 0.0, 0.0}}, {{0.015, 0.015, 0.015}}, root)
    local caster = makePart(alias .. '_front_caster', sim.primitiveshape_spheroid, {{0.05, 0.05, 0.035}}, {{0.15, 0.0, 0.035}}, nil, {{0.02, 0.02, 0.02}}, root)
    local heading = makePart(alias .. '_heading_marker', sim.primitiveshape_cuboid, {{0.13, 0.035, 0.025}}, {{0.18, 0.0, 0.115}}, nil, {{0.98, 0.98, 0.98}}, root)

    local parts = {{body, shell, tower, lidar, camera, leftWheel, rightWheel, caster, heading}}
    for _, h in ipairs(parts) do
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        sim.setObjectInt32Param(h, sim.shapeintparam_respondable, 0)
    end
    return root
end

local function collectSceneObjects()
    local objects = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    local set = {{}}
    for _, h in ipairs(objects) do
        set[h] = true
    end
    return set
end

local function importOfficialTurtleBot4(alias, entity)
    if turtlebot4Urdf == '' then
        error('official TurtleBot4 URDF path is empty')
    end

    local simURDF = require('simURDF')
    local before = collectSceneObjects()
    local urdfOptions = 2 + 8 + 256
    local robotName = simURDF.import(turtlebot4Urdf, urdfOptions, rosShareReplacement)
    log('imported official TurtleBot4 URDF for ' .. alias .. ' as ' .. tostring(robotName))

    local imported = {{}}
    local topLevel = {{}}
    local objects = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    for _, h in ipairs(objects) do
        if not before[h] then
            imported[h] = true
        end
    end
    for h, _ in pairs(imported) do
        local parent = sim.getObjectParent(h)
        if parent == -1 or not imported[parent] then
            topLevel[#topLevel + 1] = h
        end
    end
    if #topLevel == 0 then
        error('URDF import created no top-level objects for ' .. alias)
    end

    local appliedScale = scaleOfficialRobotToGymFootprint(topLevel)

    local root = sim.createDummy(0.06)
    setAlias(root, alias)
    for _, h in ipairs(topLevel) do
        sim.setObjectParent(h, root, true)
    end

    setModelStatic(root)
    local pos = pxToWorld(entity.cx, entity.cy)
    sim.setObjectPosition(root, {{pos[1], pos[2], 0.0}})
    sim.setObjectOrientation(root, {{0.0, 0.0, -math.rad(entity.theta or 0.0)}})
    alignTreeBottom(root)
    log('official TurtleBot4 scale=' .. tostring(appliedScale) .. ' gym_radius_px=' .. tostring(exportMeta.robot_radius_px))
    return root
end

local function createTurtleBot4(alias, entity, color)
    addRobotReferenceGeometry(alias, entity)
    if robotSource == 'official' then
        return importOfficialTurtleBot4(alias, entity)
    end
    return createGeometryTurtleBot4(alias, entity, color)
end

local function addObstacle(obs, index)
    local alias = string.format('TrackMaker_obstacle_%03d', index)
    local color = {{0.22, 0.24, 0.29}}
    if obs.type == 'rect' then
        makeCuboid(
            alias,
            {{obs.x + obs.w * 0.5, obs.y + obs.h * 0.5}},
            {{obs.w, obs.h}},
            obstacleHeight,
            color
        )
    elseif obs.type == 'circle' then
        makeCylinder(alias, {{obs.cx, obs.cy}}, obs.r, obstacleHeight, color)
    elseif obs.type == 'segment' then
        local dx = obs.x2 - obs.x1
        local dy = obs.y2 - obs.y1
        local length = math.sqrt(dx * dx + dy * dy)
        local thick = obs.thick or 8.0
        local handle = makeCuboid(
            alias,
            {{(obs.x1 + obs.x2) * 0.5, (obs.y1 + obs.y2) * 0.5}},
            {{length, thick}},
            obstacleHeight,
            color
        )
        sim.setObjectOrientation(handle, {{0.0, 0.0, -math.atan(dy, dx)}})
    end
end

function sysCall_init()
    local ok, err = pcall(function()
        log('building TrackMaker scene')
        log('can_save=' .. tostring(sim.getBoolParam(sim.boolparam_cansave)))
        log('density=' .. tostring(exportMeta.density) .. ' obstacle_seed=' .. tostring(exportMeta.obstacle_seed) .. ' spawn_seed=' .. tostring(exportMeta.spawn_seed) .. ' jitter_px=' .. tostring(exportMeta.jitter_px) .. ' robot_source=' .. tostring(exportMeta.robot_source))
        log('gym_scale=' .. tostring(exportMeta.scale_m_per_px) .. 'm/px sim_step_dt=' .. tostring(exportMeta.sim_step_dt) .. ' defender_speed=' .. tostring(exportMeta.defender_speed_m_per_s) .. 'm/s attacker_speed=' .. tostring(exportMeta.attacker_speed_m_per_s) .. 'm/s')

        makeCuboid(
            'TrackMaker_floor',
            {{mapWidth * 0.5, mapHeight * 0.5}},
            {{mapWidth, mapHeight}},
            0.02,
            {{0.88, 0.88, 0.86}}
        )

        for index, obs in ipairs(obstacles) do
            addObstacle(obs, index)
        end

        createTurtleBot4('TrackMaker_defender_turtlebot4', entities.defender, {{0.1, 0.35, 1.0}})
        createTurtleBot4('TrackMaker_attacker_turtlebot4', entities.attacker, {{1.0, 0.2, 0.15}})
        makeDisc('TrackMaker_target', {{entities.target.cx, entities.target.cy}}, exportMeta.target_radius_px, 0.08, {{1.0, 0.78, 0.1}}, 0.65)

        sim.saveScene(outputPath)
        writeStatus('saved ' .. outputPath)
        log('saved ' .. outputPath)
    end)

    if not ok then
        writeStatus('error ' .. tostring(err))
        log('error ' .. tostring(err))
    end

    sim.quitSimulator()
end
"""


def main() -> int:
    args = parse_args()
    coppeliasim_root = args.coppeliasim_root.resolve()
    launcher = coppeliasim_root / "coppeliaSim.sh"
    if not launcher.is_file():
        print(f"CoppeliaSim launcher not found: {launcher}", file=sys.stderr)
        return 2

    output = args.save.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    status = output.with_suffix(".status.txt")
    addon = output.with_name("build_trackmaker_scene.lua")
    log_path = output.with_suffix(".export.log")

    obstacles, entities, meta = prepare_scene_data(args)
    manifest_path = output.with_suffix(".json")
    write_scene_manifest(manifest_path, args.scale, args.height, obstacles, entities, meta)
    if args.write_manifest_only:
        print(f"wrote {manifest_path}")
        return 0

    for path in (output, status):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    try:
        turtlebot4_urdf = prepare_turtlebot4_urdf(args)
    except Exception as exc:
        print(f"failed to prepare official TurtleBot4 URDF: {exc}", file=sys.stderr)
        return 1

    addon.write_text(
        build_addon_script(
            output,
            status,
            args.scale,
            args.height,
            args.sim_step_dt,
            args.official_robot_scale,
            args.preserve_official_robot_size,
            args.robot_source,
            turtlebot4_urdf,
            args.ros_share.resolve(),
            obstacles,
            entities,
            meta,
        ),
        encoding="utf-8",
    )
    print(
        "scene "
        f"density={meta['density']} obstacle_seed={meta['obstacle_seed']} "
        f"spawn_seed={meta['spawn_seed']} jitter_px={meta['jitter_px']} "
        f"robot_source={meta['robot_source']} obstacles={len(obstacles)}"
    )
    for name in ("defender", "attacker", "target"):
        entity = entities[name]
        print(f"{name} center_px=({entity['cx']:.1f}, {entity['cy']:.1f}) theta={entity['theta']:.1f}")

    cmd = [str(launcher)]
    if not args.show_gui:
        cmd.append("-h")
    python_bin = Path(os.getenv("PYTHON_BIN", sys.executable)).resolve()
    cmd.extend([f"-a{addon}", f"-Gpython={python_bin}"])

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{coppeliasim_root}:{env.get('LD_LIBRARY_PATH', '')}"
    proc = subprocess.run(
        cmd,
        cwd=str(coppeliasim_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.timeout,
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")

    if output.is_file() and output.stat().st_size > 0:
        print(f"saved {output}")
        print(f"log {log_path}")
        return 0

    status_text = status.read_text(encoding="utf-8").strip() if status.is_file() else "no status file"
    print(f"failed to save {output}: {status_text}", file=sys.stderr)
    print(f"log {log_path}", file=sys.stderr)
    return proc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
