#!/usr/bin/env python3
"""Build an official TurtleBot4 TrackMaker scene for CoppeliaSim training."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coppelia_env.turtlebot4 import TurtleBot4Spec, load_turtlebot4_spec
from configs import map_config
from coppeliasim.tools.coppeliasim_export_trackmaker_scene import (
    DEFAULT_COPPELIASIM_ROOT,
    DEFAULT_ROS_SHARE,
    DEFAULT_ROS_STUB_PREFIX,
    DEFAULT_TURTLEBOT4_XACRO,
    DEFAULT_TURTLEBOT4_URDF,
    prepare_scene_data,
    prepare_turtlebot4_urdf,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "coppeliasim/scenes/trackmaker_turtlebot4_scene.ttt"
DEFAULT_SIM_STEP_DT = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coppeliasim-root", type=Path, default=Path(os.getenv("COPPELIASIM_ROOT", DEFAULT_COPPELIASIM_ROOT)))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument("--sim-step-dt", type=float, default=DEFAULT_SIM_STEP_DT)
    parser.add_argument("--density", default="dense", choices=map_config.ObstacleDensity.ALL_LEVELS)
    parser.add_argument("--obstacle-seed", type=int, default=20260325)
    parser.add_argument("--spawn-seed", type=int, default=20260326)
    parser.add_argument("--jitter-px", type=int, default=15)
    parser.add_argument("--robot-radius-px", type=float, default=float(getattr(map_config, "agent_radius", 8.0)))
    parser.add_argument("--target-radius-px", type=float, default=float(getattr(map_config, "target_radius", 16.0)))
    parser.add_argument("--obstacle-height", type=float, default=0.5)
    parser.add_argument("--turtlebot4-xacro", type=Path, default=DEFAULT_TURTLEBOT4_XACRO)
    parser.add_argument("--turtlebot4-urdf", type=Path, default=DEFAULT_TURTLEBOT4_URDF)
    parser.add_argument("--ros-share", type=Path, default=DEFAULT_ROS_SHARE)
    parser.add_argument("--ros-stub-prefix", type=Path, default=DEFAULT_ROS_STUB_PREFIX)
    parser.add_argument("--skip-urdf-regenerate", action="store_true")
    parser.add_argument("--official-robot-scale", type=float, default=None)
    parser.add_argument("--preserve-official-robot-size", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--write-manifest-only", action="store_true")
    parser.add_argument("--robot-source", default="official", help=argparse.SUPPRESS)
    parser.add_argument("--height", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--save", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _entity_manifest(entity: dict[str, float]) -> dict[str, Any]:
    return {
        "center_px": [float(entity["cx"]), float(entity["cy"])],
        "theta_deg": float(entity.get("theta", 0.0)),
    }


def _robot_alias(role: str) -> str:
    return f"TrackMaker_{role}"


def _robot_spec_alias(role: str, name: str) -> str:
    return f"{_robot_alias(role)}_{name}"


def build_turtlebot4_manifest(
    width_px: int,
    height_px: int,
    scale_m_per_px: float,
    obstacles: list[dict[str, Any]],
    entities: dict[str, dict[str, float]],
    meta: dict[str, Any],
    spec: TurtleBot4Spec,
    obstacle_aliases: list[str],
) -> dict[str, Any]:
    meta_out = dict(meta)
    collision_proxy_radius_m = float(meta.get("collision_proxy_radius_m", 0.18))
    collision_proxy_height_m = float(meta.get("collision_proxy_height_m", 0.11))
    meta_out.update(
        {
            "robot_source": "official_turtlebot4",
            "robot_model": "official_turtlebot4_urdf",
            "motion_model": "official_turtlebot4_wheel_velocity",
            "scale_m_per_px": float(scale_m_per_px),
            "sim_step_dt": float(meta.get("sim_step_dt", DEFAULT_SIM_STEP_DT)),
            "turtlebot4": spec.to_manifest(),
            "collision_proxy_enabled": bool(meta.get("collision_proxy_enabled", True)),
            "collision_proxy_radius_m": collision_proxy_radius_m,
            "collision_proxy_height_m": collision_proxy_height_m,
        }
    )
    robots: dict[str, dict[str, Any]] = {}
    objects: dict[str, dict[str, str]] = {
        "floor": {"alias": "TrackMaker_turtlebot4_floor"},
        "target": {"alias": "TrackMaker_target"},
    }
    for idx, alias in enumerate(obstacle_aliases, start=1):
        objects[f"obstacle_{idx:03d}"] = {"alias": alias}
    for role in ("defender", "attacker"):
        root_alias = _robot_alias(role)
        robots[role] = {
            "root": root_alias,
            "base_link": _robot_spec_alias(role, spec.base_link),
            "collision_proxy": _robot_spec_alias(role, "collision_proxy"),
            "left_wheel_joint": _robot_spec_alias(role, spec.left_wheel_joint),
            "right_wheel_joint": _robot_spec_alias(role, spec.right_wheel_joint),
            "lidar_link": _robot_spec_alias(role, spec.lidar_link),
            "rgb_camera_frame": _robot_spec_alias(role, spec.rgb_camera_frame),
            "lidar_sensor": _robot_spec_alias(role, "rplidar_sensor"),
            "vision_sensor": _robot_spec_alias(role, "oakd_rgb_sensor"),
            "wheel_radius_m": float(spec.wheel_radius_m),
            "wheel_separation_m": float(spec.wheel_separation_m),
        }
        objects[role] = {"alias": root_alias}
        objects[f"{role}_base"] = {"alias": robots[role]["base_link"]}
        objects[f"{role}_collision_proxy"] = {"alias": robots[role]["collision_proxy"]}
        objects[f"{role}_lidar"] = {"alias": robots[role]["lidar_sensor"]}
        objects[f"{role}_vision"] = {"alias": robots[role]["vision_sensor"]}

    return {
        "map": {
            "width_px": int(width_px),
            "height_px": int(height_px),
            "scale_m_per_px": float(scale_m_per_px),
            "obstacle_height_m": float(meta.get("obstacle_height_m", 0.5)),
        },
        "meta": meta_out,
        "obstacles": [dict(obs) for obs in obstacles],
        "entities": {
            key: _entity_manifest(value)
            for key, value in entities.items()
        },
        "objects": objects,
        "robots": robots,
    }


def _lua_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "nil"
    if isinstance(value, (int, float)):
        return repr(float(value))
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (list, tuple)):
        return "{" + ", ".join(_lua_literal(v) for v in value) + "}"
    if isinstance(value, dict):
        parts = []
        for key, val in value.items():
            parts.append(f"[{_lua_literal(str(key))}] = {_lua_literal(val)}")
        return "{" + ", ".join(parts) + "}"
    return json.dumps(value)


def _scene_lua(
    output_path: Path,
    status_path: Path,
    manifest: dict[str, Any],
    turtlebot4_urdf: Path,
    ros_share: Path,
    official_robot_scale: float | None,
    preserve_official_robot_size: bool,
) -> str:
    obstacles = manifest["obstacles"]
    entities = manifest["entities"]
    meta = manifest["meta"]
    scale = float(manifest["map"]["scale_m_per_px"])
    width = int(manifest["map"]["width_px"])
    height = int(manifest["map"]["height_px"])
    obstacle_height = float(manifest["map"]["obstacle_height_m"])
    return f"""
local outputPath = {_lua_literal(str(output_path))}
local statusPath = {_lua_literal(str(status_path))}
local turtlebot4Urdf = {_lua_literal(str(turtlebot4_urdf))}
local rosShareReplacement = nil
local officialRobotScaleOverride = {_lua_literal(official_robot_scale)}
local preserveOfficialRobotSize = {_lua_literal(bool(preserve_official_robot_size))}
local urdfImportOptions = 8 + 128 + 256
local scale = {_lua_literal(scale)}
local mapWidth = {_lua_literal(width)}
local mapHeight = {_lua_literal(height)}
local obstacleHeight = {_lua_literal(obstacle_height)}
local obstacles = {_lua_literal(obstacles)}
local entities = {_lua_literal(entities)}
local exportMeta = {_lua_literal(meta)}

local function log(msg)
    sim.addLog(sim.verbosity_scriptinfos, '[TrackMakerTurtleBot4] ' .. tostring(msg))
end

local function writeStatus(msg)
    local f = io.open(statusPath, 'w')
    if f then f:write(tostring(msg)); f:close() end
end

local function step(name, fn)
    log('begin ' .. tostring(name))
    local ok, result, extra = pcall(fn)
    if not ok then
        error('while ' .. tostring(name) .. ': ' .. tostring(result))
    end
    log('end ' .. tostring(name))
    return result, extra
end

local function setAlias(handle, alias)
    pcall(sim.setObjectAlias, handle, alias)
end

local function pxToWorld(cx, cy)
    return {{(cx - mapWidth * 0.5) * scale, (mapHeight * 0.5 - cy) * scale, 0.0}}
end

local function setShapeFlags(handle, static, respondable)
    pcall(sim.setObjectInt32Param, handle, sim.shapeintparam_static, static and 1 or 0)
    pcall(sim.setObjectInt32Param, handle, sim.shapeintparam_respondable, respondable and 1 or 0)
end

local function color(handle, rgb)
    pcall(sim.setShapeColor, handle, nil, sim.colorcomponent_ambient_diffuse, rgb)
end

local function makeCuboid(alias, center, size, height, rgb, static, respondable)
    local h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, {{math.max(size[1] * scale, 0.001), math.max(size[2] * scale, 0.001), height}}, 0)
    local p = pxToWorld(center[1], center[2])
    sim.setObjectPosition(h, -1, {{p[1], p[2], height * 0.5}})
    setAlias(h, alias)
    color(h, rgb)
    setShapeFlags(h, static, respondable)
    return h
end

local function makeCylinder(alias, center, radiusPx, height, rgb, static, respondable)
    local d = math.max(radiusPx * 2.0 * scale, 0.001)
    local h = sim.createPrimitiveShape(sim.primitiveshape_cylinder, {{d, d, height}}, 0)
    local p = pxToWorld(center[1], center[2])
    sim.setObjectPosition(h, -1, {{p[1], p[2], height * 0.5}})
    setAlias(h, alias)
    color(h, rgb)
    setShapeFlags(h, static, respondable)
    return h
end

local function addObstacle(obs, index)
    local alias = string.format('TrackMaker_turtlebot4_obstacle_%03d', index)
    if obs.type == 'rect' then
        makeCuboid(alias, {{obs.x + obs.w * 0.5, obs.y + obs.h * 0.5}}, {{obs.w, obs.h}}, obstacleHeight, {{0.22, 0.24, 0.29}}, true, true)
    elseif obs.type == 'circle' then
        makeCylinder(alias, {{obs.cx, obs.cy}}, obs.r, obstacleHeight, {{0.22, 0.24, 0.29}}, true, true)
    elseif obs.type == 'segment' then
        local dx = obs.x2 - obs.x1
        local dy = obs.y2 - obs.y1
        local h = makeCuboid(alias, {{(obs.x1 + obs.x2) * 0.5, (obs.y1 + obs.y2) * 0.5}}, {{math.sqrt(dx * dx + dy * dy), obs.thick or 8.0}}, obstacleHeight, {{0.22, 0.24, 0.29}}, true, true)
        sim.setObjectOrientation(h, -1, {{0.0, 0.0, -math.atan(dy, dx)}})
        makeCylinder(alias .. '_cap_1', {{obs.x1, obs.y1}}, (obs.thick or 8.0) * 0.5, obstacleHeight, {{0.22, 0.24, 0.29}}, true, true)
        makeCylinder(alias .. '_cap_2', {{obs.x2, obs.y2}}, (obs.thick or 8.0) * 0.5, obstacleHeight, {{0.22, 0.24, 0.29}}, true, true)
    end
end

local function collectSceneObjects()
    local objects = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    local set = {{}}
    for _, h in ipairs(objects) do set[h] = true end
    return set
end

local function collectImported(before)
    local out = {{}}
    local objects = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    for _, h in ipairs(objects) do
        if not before[h] then out[#out + 1] = h end
    end
    return out
end

local function copyHandleArray(handles)
    local out = {{}}
    if type(handles) ~= 'table' then return out end
    for _, h in ipairs(handles) do
        if type(h) == 'number' then out[#out + 1] = h end
    end
    return out
end

local function topLevelImported(imported)
    local set = {{}}
    local top = {{}}
    for _, h in ipairs(imported) do set[h] = true end
    for _, h in ipairs(imported) do
        local ok, parent = pcall(sim.getObjectParent, h)
        if ok and (parent == -1 or not set[parent]) then
            top[#top + 1] = h
        end
    end
    return top
end

local function scaleObjects(handles)
    if preserveOfficialRobotSize then return 1.0 end
    local factor = officialRobotScaleOverride or 1.0
    sim.scaleObjects(handles, factor, true)
    return factor
end

local function findChildByAlias(root, wanted)
    local all = sim.getObjectsInTree(root, sim.handle_all, 0)
    for _, h in ipairs(all) do
        local ok, alias = pcall(sim.getObjectAlias, h, 1)
        if ok and alias == wanted then return h end
        ok, alias = pcall(sim.getObjectAlias, h, 5)
        if ok and alias == wanted then return h end
    end
    return -1
end

local function aliasMatches(alias, wanted)
    if alias == wanted then return true end
    if type(alias) ~= 'string' then return false end
    local cleanWanted = string.gsub(wanted, '[^%w_]', '_')
    if alias == cleanWanted then return true end
    if string.sub(alias, -#wanted) == wanted then return true end
    if string.sub(alias, -#cleanWanted) == cleanWanted then return true end
    if string.find(alias, wanted, 1, true) ~= nil then return true end
    if string.find(alias, cleanWanted, 1, true) ~= nil then return true end
    return false
end

local function findDescendantByAliasSuffix(root, wanted, objectType)
    local all = sim.getObjectsInTree(root, sim.handle_all, 0)
    for _, h in ipairs(all) do
        if objectType == nil or sim.getObjectType(h) == objectType then
            for _, opt in ipairs({{0, 1, 3, 5}}) do
                local ok, alias = pcall(sim.getObjectAlias, h, opt)
                if ok and aliasMatches(alias, wanted) then return h end
            end
        end
    end
    return -1
end

local function configureImportedOfficialShapes(root)
    local all = sim.getObjectsInTree(root, sim.handle_all, 0)
    for _, h in ipairs(all) do
        if sim.getObjectType(h) == sim.sceneobject_shape then
            pcall(sim.setObjectInt32Param, h, sim.shapeintparam_static, 1)
            pcall(sim.setObjectInt32Param, h, sim.shapeintparam_respondable, 0)
        end
    end
end

local function objectAlias(handle)
    for _, opt in ipairs({{1, 5, 0}}) do
        local ok, alias = pcall(sim.getObjectAlias, handle, opt)
        if ok and type(alias) == 'string' then return alias end
    end
    return ''
end

local function roleShellColor(role)
    if role == 'attacker' then return {{0.86, 0.20, 0.14}} end
    return {{0.08, 0.38, 0.86}}
end

local function turtlebotVisualColor(alias, role)
    if string.find(alias, 'shell', 1, true) ~= nil then return roleShellColor(role) end
    if string.find(alias, 'tower_sensor_plate', 1, true) ~= nil then return roleShellColor(role) end
    if string.find(alias, 'base_link', 1, true) ~= nil then return {{0.18, 0.19, 0.21}} end
    if string.find(alias, 'bumper', 1, true) ~= nil then return {{0.10, 0.11, 0.12}} end
    if string.find(alias, 'rplidar', 1, true) ~= nil then return {{0.04, 0.04, 0.05}} end
    if string.find(alias, 'oakd', 1, true) ~= nil then return {{0.04, 0.04, 0.05}} end
    return {{0.82, 0.84, 0.80}}
end

local function tintOfficialVisualMesh(handle, role)
    local alias = objectAlias(handle)
    if string.find(alias, '_visual', 1, true) == nil then return end
    pcall(sim.setObjectInt32Param, handle, sim.objintparam_visibility_layer, 1)
    color(handle, turtlebotVisualColor(alias, role))
end

local function hideRobotSceneIcons(root)
    local all = sim.getObjectsInTree(root, sim.handle_all, 0)
    for _, h in ipairs(all) do
        if sim.getObjectType(h) ~= sim.sceneobject_shape then
            pcall(sim.setObjectInt32Param, h, sim.objintparam_visibility_layer, 0)
        end
    end
end

local function configureVisibleTurtlebotMesh(root, role)
    local all = sim.getObjectsInTree(root, sim.handle_all, 0)
    for _, h in ipairs(all) do
        if sim.getObjectType(h) == sim.sceneobject_shape then
            local alias = objectAlias(h)
            if string.find(alias, '_visual', 1, true) ~= nil then
                tintOfficialVisualMesh(h, role)
            else
                pcall(sim.setObjectInt32Param, h, sim.objintparam_visibility_layer, 0)
            end
        end
    end
    hideRobotSceneIcons(root)
end

local function attachCollisionProxy(root, role)
    if exportMeta.collision_proxy_enabled == false then return -1 end
    local radius = exportMeta.collision_proxy_radius_m or 0.18
    local height = exportMeta.collision_proxy_height_m or 0.11
    local proxy = sim.createPrimitiveShape(sim.primitiveshape_cylinder, {{math.max(radius * 2.0, 0.001), math.max(radius * 2.0, 0.001), math.max(height, 0.001)}}, 0)
    setAlias(proxy, 'TrackMaker_' .. role .. '_collision_proxy')
    color(proxy, {{0.05, 0.55, 0.95}})
    setShapeFlags(proxy, false, true)
    pcall(sim.setObjectInt32Param, proxy, sim.objintparam_visibility_layer, 0)
    pcall(sim.setShapeMass, proxy, 0.05)
    pcall(sim.setObjectFloatParam, proxy, sim.shapefloatparam_shading_angle, 0.0)
    local proxyParent = findChildByAlias(root, 'TrackMaker_' .. role .. '_base_link')
    if proxyParent < 0 then proxyParent = root end
    sim.setObjectParent(proxy, proxyParent, false)
    sim.setObjectPosition(proxy, proxyParent, {{0.0, 0.0, height * 0.5}})
    sim.setObjectOrientation(proxy, proxyParent, {{0.0, 0.0, 0.0}})
    return proxy
end

local function aliasTrainingHandles(root, role)
    setAlias(root, 'TrackMaker_' .. role)
    local base = findDescendantByAliasSuffix(root, 'base_link_respondable', sim.sceneobject_shape)
    if base < 0 then base = findDescendantByAliasSuffix(root, 'base_link', sim.sceneobject_shape) end
    if base >= 0 then setAlias(base, 'TrackMaker_' .. role .. '_base_link') end
    local leftWheel = findDescendantByAliasSuffix(root, 'left_wheel_joint', sim.sceneobject_joint)
    if leftWheel >= 0 then setAlias(leftWheel, 'TrackMaker_' .. role .. '_left_wheel_joint') end
    local rightWheel = findDescendantByAliasSuffix(root, 'right_wheel_joint', sim.sceneobject_joint)
    if rightWheel >= 0 then setAlias(rightWheel, 'TrackMaker_' .. role .. '_right_wheel_joint') end
    local lidar = findDescendantByAliasSuffix(root, 'rplidar_link', sim.sceneobject_shape)
    if lidar >= 0 then setAlias(lidar, 'TrackMaker_' .. role .. '_rplidar_link') end
    local camera = findDescendantByAliasSuffix(root, 'oakd_rgb_camera_frame', sim.sceneobject_shape)
    if camera >= 0 then setAlias(camera, 'TrackMaker_' .. role .. '_oakd_rgb_camera_frame') end
end

local function attachSensors(root, role)
    local lidarParent = findChildByAlias(root, 'TrackMaker_' .. role .. '_rplidar_link')
    if lidarParent < 0 then lidarParent = root end
    local lidar = sim.createProximitySensor(
        sim.proximitysensor_ray, 16, 1 + 4,
        {{0, 0, 0, 0, 0, 0, 0, 0}},
        {{0.0, exportMeta.sensor_range_m or 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0}}
    )
    setAlias(lidar, 'TrackMaker_' .. role .. '_rplidar_sensor')
    sim.setObjectParent(lidar, lidarParent, false)
    sim.setObjectPosition(lidar, lidarParent, {{0.0, 0.0, 0.0}})
    sim.setObjectOrientation(lidar, lidarParent, {{0.0, 0.0, 0.0}})

    local cameraParent = findChildByAlias(root, 'TrackMaker_' .. role .. '_oakd_rgb_camera_frame')
    if cameraParent < 0 then cameraParent = root end
    local vision = sim.createVisionSensor(
        1 + 4,
        {{64, 48, 0, 0}},
        {{0.01, exportMeta.sensor_range_m or 3.0, math.rad(70.0), 0.01, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}
    )
    setAlias(vision, 'TrackMaker_' .. role .. '_oakd_rgb_sensor')
    sim.setObjectParent(vision, cameraParent, false)
    sim.setObjectPosition(vision, cameraParent, {{0.0, 0.0, 0.0}})
    sim.setObjectOrientation(vision, cameraParent, {{0.0, 0.0, 0.0}})
end

local function importRobot(role, entity)
    local simURDF = step(role .. ': require simURDF', function()
        return require('simURDF')
    end)
    local before = step(role .. ': collect pre-import scene', function()
        return collectSceneObjects()
    end)
    local robotName, modelHandles = step(role .. ': import urdf', function()
        return simURDF.import(turtlebot4Urdf, urdfImportOptions, rosShareReplacement)
    end)
    log('imported ' .. role .. ' as ' .. tostring(robotName))
    local imported = copyHandleArray(modelHandles)
    if #imported == 0 then
        imported = step(role .. ': collect imported diff', function()
            return collectImported(before)
        end)
    end
    if #imported == 0 then error('URDF import created no objects for ' .. role) end
    log(role .. ' imported handles=' .. tostring(#imported))
    step(role .. ': scale imported objects', function()
        return scaleObjects(imported)
    end)
    local top = step(role .. ': collect top-level objects', function()
        return topLevelImported(imported)
    end)
    if #top == 0 then error('URDF import created no top-level objects for ' .. role) end
    local root = sim.createDummy(0.06)
    setAlias(root, 'TrackMaker_' .. role)
    step(role .. ': parent top-level objects', function()
        for _, h in ipairs(top) do
            sim.setObjectParent(h, root, true)
        end
    end)
    step(role .. ': alias training handles', function()
        aliasTrainingHandles(root, role)
    end)
    -- Preserve the URDF importer's dynamic/respondable flags. Making every
    -- imported shape static turns the TurtleBot4 into a visual-only model and
    -- makes wheel velocity commands physically inert.
    local p = pxToWorld(entity.center_px[1], entity.center_px[2])
    sim.setObjectPosition(root, -1, {{p[1], p[2], 0.0}})
    sim.setObjectOrientation(root, -1, {{0.0, 0.0, -math.rad(entity.theta_deg or 0.0)}})
    step(role .. ': attach collision proxy', function()
        attachCollisionProxy(root, role)
    end)
    step(role .. ': attach sensors', function()
        attachSensors(root, role)
    end)
    step(role .. ': configure visible turtlebot mesh', function()
        configureVisibleTurtlebotMesh(root, role)
    end)
    return root
end

function sysCall_init()
    local ok, err = pcall(function()
        log('building official TurtleBot4 training scene')
        if sim.getSimulationState() ~= sim.simulation_stopped then sim.stopSimulation() end
        pcall(sim.setFloatParam, sim.floatparam_simulation_time_step, exportMeta.sim_step_dt or 0.05)
        makeCuboid('TrackMaker_turtlebot4_floor', {{mapWidth * 0.5, mapHeight * 0.5}}, {{mapWidth, mapHeight}}, 0.02, {{0.88, 0.88, 0.86}}, true, true)
        for index, obs in ipairs(obstacles) do addObstacle(obs, index) end
        importRobot('defender', entities.defender)
        importRobot('attacker', entities.attacker)
        makeCylinder('TrackMaker_target', entities.target.center_px, exportMeta.target_radius_px or 16.0, 0.08, {{1.0, 0.78, 0.1}}, true, false)
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
    output = (args.save or args.output).resolve()
    args.output = output
    args.save = output
    args.height = float(args.height if args.height is not None else args.obstacle_height)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        turtlebot4_urdf = prepare_turtlebot4_urdf(args)
        spec = load_turtlebot4_spec(turtlebot4_urdf)
    except Exception as exc:
        print(f"failed to prepare TurtleBot4 URDF: {exc}", file=sys.stderr)
        return 1

    obstacles, entities, meta = prepare_scene_data(args)
    meta = dict(meta)
    meta.update(
        {
            "sim_step_dt": float(args.sim_step_dt),
            "obstacle_height_m": float(args.obstacle_height),
            "sensor_range_m": float(min(getattr(map_config.EnvParameters, "FOV_RANGE", 300), math.hypot(map_config.width, map_config.height)) * args.scale),
            "defender_speed_m_per_s": float(meta["defender_speed_px_per_step"]) * float(args.scale) / float(args.sim_step_dt),
            "attacker_speed_m_per_s": float(meta["attacker_speed_px_per_step"]) * float(args.scale) / float(args.sim_step_dt),
            "defender_max_turn_rad_per_s": math.radians(float(meta["defender_max_turn_deg_per_step"])) / float(args.sim_step_dt),
            "attacker_max_turn_rad_per_s": math.radians(float(meta["attacker_max_turn_deg_per_step"])) / float(args.sim_step_dt),
            "episode_len": int(map_config.EnvParameters.EPISODE_LEN) if hasattr(map_config, "EnvParameters") else 449,
            "collision_proxy_enabled": False,
        }
    )
    obstacle_aliases = [f"TrackMaker_turtlebot4_obstacle_{idx:03d}" for idx, _obs in enumerate(obstacles, start=1)]
    manifest = build_turtlebot4_manifest(
        width_px=int(map_config.width),
        height_px=int(map_config.height),
        scale_m_per_px=float(args.scale),
        obstacles=obstacles,
        entities=entities,
        meta=meta,
        spec=spec,
        obstacle_aliases=obstacle_aliases,
    )
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.write_manifest_only:
        print(f"wrote {manifest_path}")
        return 0

    coppeliasim_root = args.coppeliasim_root.resolve()
    launcher = coppeliasim_root / "coppeliaSim.sh"
    if not launcher.is_file():
        print(f"CoppeliaSim launcher not found: {launcher}", file=sys.stderr)
        return 2
    status_path = output.with_suffix(".status.txt")
    addon_path = output.with_name("build_turtlebot4_scene.lua")
    log_path = output.with_suffix(".build.log")
    for path in (output, status_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    addon_path.write_text(
        _scene_lua(
            output,
            status_path,
            manifest,
            turtlebot4_urdf,
            args.ros_share.resolve(),
            args.official_robot_scale,
            args.preserve_official_robot_size,
        ),
        encoding="utf-8",
    )
    cmd = [str(launcher)]
    if not args.show_gui:
        cmd.append("-h")
    python_bin = Path(os.getenv("PYTHON_BIN", sys.executable)).resolve()
    cmd.extend([f"-a{addon_path}", f"-Gpython={python_bin}"])
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{coppeliasim_root}:{env.get('LD_LIBRARY_PATH', '')}"
    env["QT_PLUGIN_PATH"] = str(coppeliasim_root)
    env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(coppeliasim_root / "platforms")
    env.pop("QT_DEBUG_PLUGINS", None)
    proc = subprocess.run(
        cmd,
        cwd=str(coppeliasim_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=float(args.timeout),
        check=False,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    if output.is_file() and output.stat().st_size > 0:
        print(f"saved {output}")
        print(f"wrote {manifest_path}")
        print(f"log {log_path}")
        return 0
    status = status_path.read_text(encoding="utf-8").strip() if status_path.is_file() else "no status file"
    print(f"failed to save {output}: {status}", file=sys.stderr)
    print(f"log {log_path}", file=sys.stderr)
    return proc.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
