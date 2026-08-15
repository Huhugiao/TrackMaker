#!/usr/bin/env python3
"""Build the profile-driven TurtleBot4 digital-twin V2.1 scene."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coppelia_env.digital_twin import load_profile, profile_metadata  # noqa: E402
from configs import map_config  # noqa: E402
from coppeliasim.tools.coppeliasim_build_turtlebot4_scene import _lua_literal  # noqa: E402
from coppeliasim.tools.coppeliasim_export_trackmaker_scene import (  # noqa: E402
    DEFAULT_COPPELIASIM_ROOT,
    DEFAULT_ROS_SHARE,
    DEFAULT_ROS_STUB_PREFIX,
    DEFAULT_TURTLEBOT4_URDF,
    DEFAULT_TURTLEBOT4_XACRO,
    prepare_scene_data,
    prepare_turtlebot4_urdf,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "coppeliasim/scenes/trackmaker_turtlebot4_v2_1_scene.ttt"
DEFAULT_PROFILE = PROJECT_ROOT / "coppeliasim/profiles/trackmaker_turtlebot4_v2_1_prior.json"
DEFAULT_VISUAL_SOURCE_SCENE = PROJECT_ROOT / "coppeliasim/scenes/trackmaker_turtlebot4_scene.ttt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coppeliasim-root", type=Path, default=Path(os.getenv("COPPELIASIM_ROOT", DEFAULT_COPPELIASIM_ROOT)))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--visual-source-scene", type=Path, default=DEFAULT_VISUAL_SOURCE_SCENE)
    parser.add_argument("--scale", type=float, default=0.01)
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
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--show-gui", action="store_true")
    parser.add_argument("--write-manifest-only", action="store_true")
    parser.add_argument("--replace", action="store_true", help="replace an existing generated V2 scene")
    parser.add_argument("--robot-source", default="official", help=argparse.SUPPRESS)
    parser.add_argument("--height", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--save", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _entity(entity: dict[str, float]) -> dict[str, Any]:
    return {
        "center_px": [float(entity["cx"]), float(entity["cy"])],
        "theta_deg": float(entity.get("theta", 0.0)),
    }


def build_manifest(
    profile: dict[str, Any],
    obstacles: list[dict[str, Any]],
    entities: dict[str, dict[str, float]],
    meta: dict[str, Any],
    *,
    scale: float,
    profile_path: Path,
) -> dict[str, Any]:
    obstacle_aliases = [f"TrackMaker_turtlebot4_obstacle_{index:03d}" for index in range(1, len(obstacles) + 1)]
    objects: dict[str, dict[str, str]] = {
        "floor": {"alias": "TrackMaker_turtlebot4_floor"},
        "target": {"alias": "TrackMaker_target"},
    }
    for index, alias in enumerate(obstacle_aliases, start=1):
        objects[f"obstacle_{index:03d}"] = {"alias": alias}
    robots: dict[str, Any] = {}
    for role, physical in profile["robots"].items():
        prefix = f"TrackMaker_{role}"
        robots[role] = {
            "root": prefix,
            "base_link": f"{prefix}_base_link",
            "chassis": f"{prefix}_chassis",
            "caster": f"{prefix}_front_caster",
            "left_wheel_joint": f"{prefix}_left_wheel_joint",
            "right_wheel_joint": f"{prefix}_right_wheel_joint",
            "left_wheel": f"{prefix}_left_wheel",
            "right_wheel": f"{prefix}_right_wheel",
            "lidar_link": f"{prefix}_rplidar_link",
            "lidar_sensor": f"{prefix}_rplidar_sensor",
            "wheel_radius_m": physical["wheel_radius_m"],
            "wheel_separation_m": physical["wheel_separation_m"],
            "profile": physical,
        }
        objects[role] = {"alias": prefix}
        objects[f"{role}_base"] = {"alias": robots[role]["base_link"]}
        objects[f"{role}_chassis"] = {"alias": robots[role]["chassis"]}
        objects[f"{role}_caster"] = {"alias": robots[role]["caster"]}
        objects[f"{role}_left_wheel"] = {"alias": robots[role]["left_wheel"]}
        objects[f"{role}_right_wheel"] = {"alias": robots[role]["right_wheel"]}
        objects[f"{role}_lidar"] = {"alias": robots[role]["lidar_sensor"]}
    metadata = dict(meta)
    metadata.update(
        {
            "digital_twin_version": "2.1",
            "profile": profile_metadata(profile),
            "profile_path": str(profile_path.relative_to(PROJECT_ROOT)),
            "robot_source": "official_turtlebot4_visual_plus_equivalent_dynamics",
            "robot_model": "calibratable_turtlebot4_v2_1",
            "motion_model": "profiled_turtlebot4_wheel_velocity",
            "scale_m_per_px": float(scale),
            "sim_step_dt": profile["simulation"]["outer_step_s"],
            "physics_step_dt": profile["simulation"]["physics_step_s"],
            "collision_proxy_enabled": False,
            "runtime_teleport_enabled": False,
            "wheel_drop_enabled": False,
            "create3_reflex_simulated": False,
            "visual_mesh_respondable": False,
            "controller_env_obstacle_mask": False,
            "action_shield": False,
        }
    )
    return {
        "map": {
            "width_px": int(map_config.width),
            "height_px": int(map_config.height),
            "scale_m_per_px": float(scale),
            "obstacle_height_m": float(metadata["obstacle_height_m"]),
        },
        "meta": metadata,
        "profile": profile,
        "obstacles": [dict(obstacle) for obstacle in obstacles],
        "entities": {name: _entity(entity) for name, entity in entities.items()},
        "objects": objects,
        "robots": robots,
    }


def _scene_lua(
    output_path: Path,
    status_path: Path,
    manifest: dict[str, Any],
    turtlebot4_urdf: Path,
    visual_source_scene: Path,
) -> str:
    profile = manifest["profile"]
    simulation = profile["simulation"]
    map_data = manifest["map"]
    return f"""
local outputPath = {_lua_literal(str(output_path))}
local statusPath = {_lua_literal(str(status_path))}
local turtlebot4Urdf = {_lua_literal(str(turtlebot4_urdf))}
local visualSourceScene = {_lua_literal(str(visual_source_scene))}
local profile = {_lua_literal(profile)}
local obstacles = {_lua_literal(manifest["obstacles"])}
local entities = {_lua_literal(manifest["entities"])}
local mapWidth = {_lua_literal(map_data["width_px"])}
local mapHeight = {_lua_literal(map_data["height_px"])}
local scale = {_lua_literal(map_data["scale_m_per_px"])}
local obstacleHeight = {_lua_literal(map_data["obstacle_height_m"])}
local urdfImportOptions = 8 + 128 + 256

local function log(message)
    sim.addLog(sim.verbosity_scriptinfos, '[TrackMakerV2Builder] ' .. tostring(message))
end

local function writeStatus(message)
    local file = io.open(statusPath, 'w')
    if file then file:write(tostring(message)); file:close() end
end

local function requireClose(name, actual, expected, tolerance)
    if math.abs(actual - expected) > (tolerance or 1e-9) then
        error(name .. ' mismatch: expected ' .. tostring(expected) .. ', got ' .. tostring(actual))
    end
end

local function configurePhysics()
    sim.setIntArrayProperty(sim.handle_scene, 'dynamicsEngine', {{math.floor(profile.simulation.engine_index), math.floor(profile.simulation.engine_version)}})
    sim.setBoolProperty(sim.handle_scene, 'dynamicsEnabled', true)
    sim.setFloatParam(sim.floatparam_simulation_time_step, profile.simulation.outer_step_s)
    sim.setFloatProperty(sim.handle_scene, 'dynamicsStepSize', profile.simulation.physics_step_s)
    sim.setIntProperty(sim.handle_scene, 'bullet.iterations', math.floor(profile.simulation.solver_iterations))
    sim.setVector3Property(sim.handle_scene, 'gravity', profile.simulation.gravity_mps2)
    local engine = sim.getIntArrayProperty(sim.handle_scene, 'dynamicsEngine')
    if engine[1] ~= profile.simulation.engine_index then error('failed to select Bullet engine') end
    requireClose('outer step', sim.getFloatParam(sim.floatparam_simulation_time_step), profile.simulation.outer_step_s)
    requireClose('physics step', sim.getFloatProperty(sim.handle_scene, 'dynamicsStepSize'), profile.simulation.physics_step_s)
    if sim.getIntProperty(sim.handle_scene, 'bullet.iterations') ~= profile.simulation.solver_iterations then error('solver iteration mismatch') end
    local gravity = sim.getVector3Property(sim.handle_scene, 'gravity')
    for index = 1, 3 do requireClose('gravity[' .. index .. ']', gravity[index], profile.simulation.gravity_mps2[index]) end
end

local function setAlias(handle, name)
    sim.setObjectAlias(handle, name)
end

local function tagPhysical(handle, role, kind)
    sim.setStringProperty(handle, 'customData.trackmakerPhysicalRole', role)
    sim.setStringProperty(handle, 'customData.trackmakerPhysicalKind', kind)
end

local objectAlias

local function findExactAlias(wanted)
    for _, handle in ipairs(sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)) do
        if objectAlias(handle) == wanted then return handle end
    end
    return -1
end

local function countExactAlias(wanted)
    local count = 0
    for _, handle in ipairs(sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)) do
        if objectAlias(handle) == wanted then count = count + 1 end
    end
    return count
end

local function setFlags(handle, static, respondable, mask)
    sim.setObjectInt32Param(handle, sim.shapeintparam_static, static and 1 or 0)
    sim.setObjectInt32Param(handle, sim.shapeintparam_respondable, respondable and 1 or 0)
    if mask ~= nil then pcall(sim.setObjectInt32Param, handle, sim.shapeintparam_respondable_mask, mask) end
end

local function setColor(handle, color)
    pcall(sim.setShapeColor, handle, nil, sim.colorcomponent_ambient_diffuse, color)
end

local function setMaterial(handle, name)
    local material = profile.materials[name]
    sim.setFloatProperty(handle, 'bullet.friction', material.friction)
    sim.setFloatProperty(handle, 'bullet.restitution', material.restitution)
    sim.setFloatProperty(handle, 'bullet.linearDamping', material.linear_damping)
    sim.setFloatProperty(handle, 'bullet.angularDamping', material.angular_damping)
    sim.setStringProperty(handle, 'customData.trackmakerMaterial', name)
end

local function setMassInertia(handle, mass, inertia, center)
    sim.setShapeMass(handle, mass)
    sim.setShapeInertia(handle, inertia, {{1.0, 0.0, 0.0, center[1], 0.0, 1.0, 0.0, center[2], 0.0, 0.0, 1.0, center[3]}})
end

local function shapeMeshWorldMinZ(handle)
    local minimum = math.huge
    local matrix = sim.getObjectMatrix(handle, sim.handle_world)
    local index = 0
    while true do
        local ok, data = pcall(sim.getShapeViz, handle, index)
        if not ok or data == nil or data.vertices == nil then break end
        for offset = 1, #data.vertices, 3 do
            local z = matrix[9] * data.vertices[offset] + matrix[10] * data.vertices[offset + 1]
                + matrix[11] * data.vertices[offset + 2] + matrix[12]
            minimum = math.min(minimum, z)
        end
        index = index + 1
    end
    return minimum
end

local function pxToWorld(cx, cy)
    return {{(cx - mapWidth * 0.5) * scale, (mapHeight * 0.5 - cy) * scale}}
end

local function cuboid(name, center, size, color, static, respondable)
    local handle = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 0)
    sim.setObjectPosition(handle, sim.handle_world, center)
    setAlias(handle, name)
    setColor(handle, color)
    setFlags(handle, static, respondable, 0xffff)
    return handle
end

local function cylinder(name, center, radius, height, color, static, respondable)
    local handle = sim.createPrimitiveShape(sim.primitiveshape_cylinder, {{radius * 2.0, radius * 2.0, height}}, 0)
    sim.setObjectPosition(handle, sim.handle_world, center)
    setAlias(handle, name)
    setColor(handle, color)
    setFlags(handle, static, respondable, 0xffff)
    return handle
end

local function sphere(name, center, radius, color, static, respondable)
    local handle = sim.createPrimitiveShape(sim.primitiveshape_spheroid, {{radius * 2.0, radius * 2.0, radius * 2.0}}, 0)
    sim.setObjectPosition(handle, sim.handle_world, center)
    setAlias(handle, name)
    setColor(handle, color)
    setFlags(handle, static, respondable, 0xffff)
    return handle
end

local function addObstacle(obstacle, index)
    local name = string.format('TrackMaker_turtlebot4_obstacle_%03d', index)
    local handle
    if obstacle.type == 'rect' then
        local p = pxToWorld(obstacle.x + obstacle.w * 0.5, obstacle.y + obstacle.h * 0.5)
        handle = cuboid(name, {{p[1], p[2], obstacleHeight * 0.5}}, {{math.max(obstacle.w * scale, 0.001), math.max(obstacle.h * scale, 0.001), obstacleHeight}}, {{0.22, 0.24, 0.29}}, true, true)
    elseif obstacle.type == 'circle' then
        local p = pxToWorld(obstacle.cx, obstacle.cy)
        handle = cylinder(name, {{p[1], p[2], obstacleHeight * 0.5}}, obstacle.r * scale, obstacleHeight, {{0.22, 0.24, 0.29}}, true, true)
    elseif obstacle.type == 'segment' then
        local p = pxToWorld((obstacle.x1 + obstacle.x2) * 0.5, (obstacle.y1 + obstacle.y2) * 0.5)
        local dx, dy = obstacle.x2 - obstacle.x1, obstacle.y2 - obstacle.y1
        handle = cuboid(name, {{p[1], p[2], obstacleHeight * 0.5}}, {{math.sqrt(dx * dx + dy * dy) * scale, (obstacle.thick or 8.0) * scale, obstacleHeight}}, {{0.22, 0.24, 0.29}}, true, true)
        sim.setObjectOrientation(handle, sim.handle_world, {{0.0, 0.0, -math.atan(dy, dx)}})
        local p1, p2 = pxToWorld(obstacle.x1, obstacle.y1), pxToWorld(obstacle.x2, obstacle.y2)
        local cap1 = cylinder(name .. '_cap_1', {{p1[1], p1[2], obstacleHeight * 0.5}}, (obstacle.thick or 8.0) * 0.5 * scale, obstacleHeight, {{0.22, 0.24, 0.29}}, true, true)
        local cap2 = cylinder(name .. '_cap_2', {{p2[1], p2[2], obstacleHeight * 0.5}}, (obstacle.thick or 8.0) * 0.5 * scale, obstacleHeight, {{0.22, 0.24, 0.29}}, true, true)
        setMaterial(cap1, 'obstacle'); setMaterial(cap2, 'obstacle')
    end
    if handle then setMaterial(handle, 'obstacle') end
end

local function collectSceneObjects()
    local set = {{}}
    for _, handle in ipairs(sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)) do set[handle] = true end
    return set
end

local function collectImported(before, modelHandles)
    local imported = {{}}
    if type(modelHandles) == 'table' then
        for _, handle in ipairs(modelHandles) do if type(handle) == 'number' then imported[#imported + 1] = handle end end
    end
    if #imported == 0 then
        for _, handle in ipairs(sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)) do
            if not before[handle] then imported[#imported + 1] = handle end
        end
    end
    return imported
end

local function topLevel(handles)
    local set, result = {{}}, {{}}
    for _, handle in ipairs(handles) do set[handle] = true end
    for _, handle in ipairs(handles) do
        local parent = sim.getObjectParent(handle)
        if parent == -1 or not set[parent] then result[#result + 1] = handle end
    end
    return result
end

objectAlias = function(handle)
    local ok, value = pcall(sim.getObjectAlias, handle, -1)
    if ok and type(value) == 'string' then return value end
    return ''
end

local function findSuffix(root, wanted, objectType)
    for _, handle in ipairs(sim.getObjectsInTree(root, sim.handle_all, 0)) do
        if objectType == nil or sim.getObjectType(handle) == objectType then
            local name = objectAlias(handle)
            if name == wanted or string.find(name, wanted, 1, true) ~= nil then return handle end
        end
    end
    return -1
end

local function adoptVisual(role, chassis, visualRoot)
    if visualRoot < 0 then error('visual source robot missing for ' .. role) end
    local lidarVisual = findSuffix(visualRoot, 'TrackMaker_' .. role .. '_rplidar_link', sim.sceneobject_shape)
    local lidarPose = lidarVisual >= 0 and sim.getObjectPose(lidarVisual, sim.handle_world) or nil
    local original = sim.getObjectsInTree(visualRoot, sim.handle_all, 0)
    local visualShapes = {{}}
    for _, handle in ipairs(original) do
        if sim.getObjectType(handle) == sim.sceneobject_shape then
            local name = objectAlias(handle)
            if string.find(name, '_visual', 1, true) ~= nil then visualShapes[#visualShapes + 1] = handle end
        end
    end
    if #visualShapes == 0 then error('visual source contains no _visual meshes for ' .. role) end
    local visualAnchor = sim.createDummy(0.01)
    setAlias(visualAnchor, 'TrackMaker_' .. role .. '_visual_root')
    sim.setObjectPose(visualAnchor, sim.handle_world, sim.getObjectPose(chassis, sim.handle_world))
    sim.setObjectParent(visualAnchor, chassis, true)
    local kept = {{}}
    for index, handle in ipairs(visualShapes) do
        kept[handle] = true
        local name = objectAlias(handle)
        setFlags(handle, true, false, 0)
        sim.setBoolProperty(handle, 'collidable', false)
        sim.setBoolProperty(handle, 'measurable', false)
        sim.setBoolProperty(handle, 'detectable', false)
        sim.setBoolProperty(handle, 'hideFromModelBB', true)
        sim.setObjectInt32Param(handle, sim.objintparam_visibility_layer, 1)
        if string.find(name, 'shell', 1, true) then
            setColor(handle, role == 'attacker' and {{0.86, 0.20, 0.14}} or {{0.08, 0.38, 0.86}})
        elseif string.find(name, 'base_link', 1, true) or string.find(name, 'bumper', 1, true) then
            setColor(handle, {{0.12, 0.13, 0.15}})
        end
        setAlias(handle, string.format('TrackMaker_%s_visual_%04d', role, index))
        sim.setObjectParent(handle, visualAnchor, true)
    end
    local remove = {{}}
    for _, handle in ipairs(original) do if not kept[handle] then remove[#remove + 1] = handle end end
    if #remove > 0 then sim.removeObjects(remove) end
    local lidarFrame = sim.createDummy(0.01)
    setAlias(lidarFrame, 'TrackMaker_' .. role .. '_rplidar_link')
    tagPhysical(lidarFrame, role, 'lidar_link')
    if lidarPose ~= nil then
        sim.setObjectPose(lidarFrame, sim.handle_world, lidarPose)
    else
        sim.setObjectPosition(lidarFrame, sim.handle_world, {{0.0, 0.0, 0.35}})
    end
    sim.setObjectParent(lidarFrame, chassis, true)
    return lidarFrame
end

local function configureJoint(handle, force)
    sim.setJointMode(handle, sim.jointmode_dynamic)
    sim.setObjectInt32Param(handle, sim.jointintparam_dynctrlmode, sim.jointdynctrl_velocity)
    sim.setJointTargetForce(handle, force)
    sim.setJointTargetVelocity(handle, 0.0)
end

local function createRobot(role)
    local robot = profile.robots[role]
    local geometry = robot.geometry
    -- The V1 visual source contains a full URDF dynamics tree rooted at
    -- ``TrackMaker_<role>_base_link`` plus a tracking dummy.  Selecting the
    -- dummy leaves the legacy joints alive and creates duplicate wheel aliases;
    -- the runtime bridge can then command the wrong joint.  Adopt visuals from
    -- the model root so every legacy physical object is removed atomically.
    local sourceVisualRoot = findExactAlias('TrackMaker_' .. role .. '_base_link')
    if sourceVisualRoot < 0 then error('visual source robot missing for ' .. role) end
    local roleColor = role == 'attacker' and {{0.86, 0.20, 0.14}} or {{0.08, 0.38, 0.86}}
    local casterGeometry = geometry.caster
    local base = cylinder('TrackMaker_' .. role .. '_base_component', geometry.base.center_m, geometry.base.radius_m, geometry.base.height_m, roleColor, false, true)
    local shell = cylinder('TrackMaker_' .. role .. '_shell_component', geometry.shell.center_m, geometry.shell.radius_m, geometry.shell.height_m, roleColor, false, true)
    local bumper = cuboid('TrackMaker_' .. role .. '_bumper_component', geometry.bumper.center_m, geometry.bumper.size_m, {{0.10, 0.11, 0.12}}, false, true)
    local baseCenter = geometry.base.center_m
    local chassis = sim.groupShapes({{shell, bumper, base}}, false)
    if not sim.relocateShapeFrame(chassis, {{baseCenter[1], baseCenter[2], baseCenter[3], 0.0, 0.0, 0.0, 1.0}}) then
        error('failed to relocate compound chassis frame for ' .. role)
    end
    requireClose(
        role .. '.chassis_mesh_world_min_z',
        shapeMeshWorldMinZ(chassis),
        geometry.base.center_m[3] - geometry.base.height_m * 0.5,
        1e-6
    )
    setAlias(chassis, 'TrackMaker_' .. role .. '_chassis')
    tagPhysical(chassis, role, 'chassis')
    sim.setBoolProperty(chassis, 'modelBase', true)
    sim.setBoolProperty(chassis, 'model.notDynamic', false)
    sim.setBoolProperty(chassis, 'model.notRespondable', false)
    sim.setBoolProperty(chassis, 'model.notCollidable', false)
    setFlags(chassis, false, true, 0xff00)
    setMaterial(chassis, 'chassis')
    local chassisCom = {{robot.chassis.center_of_mass_m[1] - baseCenter[1], robot.chassis.center_of_mass_m[2] - baseCenter[2], robot.chassis.center_of_mass_m[3] - baseCenter[3]}}
    setMassInertia(chassis, robot.chassis.mass_kg, robot.chassis.inertia_kg_m2, chassisCom)

    local root = sim.createDummy(0.02)
    setAlias(root, 'TrackMaker_' .. role .. '_physical_root_pending')
    tagPhysical(root, role, 'root')
    sim.setObjectParent(root, chassis, false)
    sim.setObjectPosition(root, chassis, {{-baseCenter[1], -baseCenter[2], -baseCenter[3]}})
    sim.setObjectOrientation(root, chassis, {{0.0, 0.0, 0.0}})
    local baseFrame = sim.createDummy(0.015)
    setAlias(baseFrame, 'TrackMaker_' .. role .. '_base_link')
    tagPhysical(baseFrame, role, 'base_link')
    sim.setObjectParent(baseFrame, chassis, false)
    sim.setObjectPosition(baseFrame, chassis, {{-baseCenter[1], -baseCenter[2], -baseCenter[3]}})
    sim.setObjectOrientation(baseFrame, chassis, {{0.0, 0.0, 0.0}})

    local caster = sphere('TrackMaker_' .. role .. '_front_caster', casterGeometry.center_m, casterGeometry.radius_m, {{0.08, 0.08, 0.09}}, false, true)
    tagPhysical(caster, role, 'caster')
    setFlags(caster, false, true, 0xff00)
    setMaterial(caster, 'caster')
    setMassInertia(caster, robot.caster.mass_kg, robot.caster.inertia_kg_m2, robot.caster.center_of_mass_m)
    local casterJoint = sim.createJoint(sim.joint_spherical_subtype, sim.jointmode_dynamic, 0)
    sim.setJointMode(casterJoint, sim.jointmode_dynamic)
    sim.setObjectInt32Param(casterJoint, sim.jointintparam_dynctrlmode, sim.jointdynctrl_free)
    setAlias(casterJoint, 'TrackMaker_' .. role .. '_caster_spherical_joint')
    tagPhysical(casterJoint, role, 'caster_spherical_joint')
    sim.setObjectParent(casterJoint, chassis, false)
    sim.setObjectPosition(casterJoint, chassis, {{casterGeometry.center_m[1] - baseCenter[1], casterGeometry.center_m[2] - baseCenter[2], casterGeometry.center_m[3] - baseCenter[3]}})
    sim.setObjectOrientation(casterJoint, chassis, {{0.0, 0.0, 0.0}})
    sim.setObjectParent(caster, casterJoint, false)
    sim.setObjectPosition(caster, casterJoint, {{0.0, 0.0, 0.0}})
    sim.setObjectOrientation(caster, casterJoint, {{0.0, 0.0, 0.0}})

    local wheelHandles, jointHandles = {{}}, {{}}
    for _, side in ipairs({{'left', 'right'}}) do
        local wheelGeometry = geometry[side .. '_wheel']
        local joint = sim.createJoint(sim.joint_revolute_subtype, sim.jointmode_dynamic, 0)
        setAlias(joint, 'TrackMaker_' .. role .. '_' .. side .. '_wheel_joint')
        tagPhysical(joint, role, side .. '_wheel_joint')
        sim.setObjectParent(joint, chassis, false)
        sim.setObjectPosition(joint, chassis, {{wheelGeometry.center_m[1] - baseCenter[1], wheelGeometry.center_m[2] - baseCenter[2], wheelGeometry.center_m[3] - baseCenter[3]}})
        sim.setObjectOrientation(joint, chassis, {{-math.pi * 0.5, 0.0, 0.0}})
        configureJoint(joint, robot.actuator.target_force_nm)
        local wheel = cylinder('TrackMaker_' .. role .. '_' .. side .. '_wheel', wheelGeometry.center_m, wheelGeometry.radius_m, wheelGeometry.width_m, {{0.06, 0.06, 0.07}}, false, true)
        tagPhysical(wheel, role, side .. '_wheel')
        sim.setObjectOrientation(wheel, sim.handle_world, {{-math.pi * 0.5, 0.0, 0.0}})
        setFlags(wheel, false, true, 0xff00)
        setMaterial(wheel, 'wheel')
        setMassInertia(wheel, robot.wheel.mass_kg_each, robot.wheel.inertia_kg_m2_each, {{0.0, 0.0, 0.0}})
        sim.setObjectParent(wheel, joint, false)
        sim.setObjectPosition(wheel, joint, {{0.0, 0.0, 0.0}})
        sim.setObjectOrientation(wheel, joint, {{0.0, 0.0, 0.0}})
        wheelHandles[side] = wheel
        jointHandles[side] = joint
    end
    for _, side in ipairs({{'left', 'right'}}) do
        local actual = sim.getObjectPosition(jointHandles[side], baseFrame)
        local expected = geometry[side .. '_wheel'].center_m
        for index = 1, 3 do requireClose(role .. '.' .. side .. '_wheel_axis[' .. index .. ']', actual[index], expected[index], 1e-9) end
    end

    sim.setStringProperty(chassis, 'customData.trackmakerProfileChecksum', profile.checksum)
    sim.setFloatProperty(chassis, 'customData.trackmakerWheelRadius', robot.wheel_radius_m)
    sim.setFloatProperty(chassis, 'customData.trackmakerWheelSeparation', robot.wheel_separation_m)
    sim.setFloatProperty(chassis, 'customData.trackmakerTargetForce', robot.actuator.target_force_nm)
    sim.setFloatProperty(chassis, 'customData.trackmakerCasterMass', robot.caster.mass_kg)
    sim.setStringProperty(chassis, 'customData.trackmakerRole', role)

    local entity = entities[role]
    local world = pxToWorld(entity.center_px[1], entity.center_px[2])
    sim.setObjectPosition(chassis, sim.handle_world, {{world[1], world[2], geometry.nominal_base_z_m + baseCenter[3]}})
    sim.setObjectOrientation(chassis, sim.handle_world, {{0.0, 0.0, -math.rad(entity.theta_deg or 0.0)}})
    local lidarFrame = adoptVisual(role, chassis, sourceVisualRoot)
    setAlias(root, 'TrackMaker_' .. role)
    local lidar = sim.createProximitySensor(sim.proximitysensor_ray, 16, 1 + 4, {{0, 0, 0, 0, 0, 0, 0, 0}}, {{0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0}})
    setAlias(lidar, 'TrackMaker_' .. role .. '_rplidar_sensor')
    tagPhysical(lidar, role, 'lidar_sensor')
    sim.setObjectParent(lidar, lidarFrame, false)
    sim.setObjectPosition(lidar, lidarFrame, {{0.0, 0.0, 0.0}})
end

function sysCall_init()
    local ok, message = pcall(function()
        if sim.getSimulationState() ~= sim.simulation_stopped then sim.stopSimulation() end
        sim.loadScene(visualSourceScene)
        configurePhysics()
        sim.setBufferProperty(sim.handle_scene, 'customData.trackmakerProfile', sim.packTable(profile))
        sim.setStringProperty(sim.handle_scene, 'customData.trackmakerProfileChecksum', profile.checksum)
        local floor = findExactAlias('TrackMaker_turtlebot4_floor')
        if floor < 0 then error('visual source floor missing') end
        setMaterial(floor, 'floor')
        local oldObstacles = {{}}
        for _, handle in ipairs(sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)) do
            if string.sub(objectAlias(handle), 1, 31) == 'TrackMaker_turtlebot4_obstacle_' then
                oldObstacles[#oldObstacles + 1] = handle
            end
        end
        if #oldObstacles > 0 then sim.removeObjects(oldObstacles) end
        for index, obstacle in ipairs(obstacles) do
            addObstacle(obstacle, index)
        end
        createRobot('defender')
        createRobot('attacker')
        for _, role in ipairs({{'defender', 'attacker'}}) do
            for _, suffix in ipairs({{'chassis', 'base_link', 'left_wheel_joint', 'right_wheel_joint', 'left_wheel', 'right_wheel'}}) do
                local wanted = 'TrackMaker_' .. role .. '_' .. suffix
                local count = countExactAlias(wanted)
                if count ~= 1 then error('physical alias must be unique: ' .. wanted .. ' count=' .. count) end
            end
        end
        local target = findExactAlias('TrackMaker_target')
        if target < 0 then error('visual source target missing') end
        setFlags(target, true, false, 0)
        sim.saveScene(outputPath)
        writeStatus('saved ' .. outputPath)
        log('saved profile-driven V2.1 scene ' .. outputPath)
    end)
    if not ok then
        writeStatus('error ' .. tostring(message))
        log('error ' .. tostring(message))
    end
    sim.quitSimulator()
end
"""


def main() -> int:
    args = parse_args()
    output = (args.save or args.output).resolve()
    profile_path = args.profile.resolve()
    visual_source_scene = args.visual_source_scene.resolve()
    if output.exists() and not args.replace:
        print(f"refusing to overwrite existing scene without --replace: {output}", file=sys.stderr)
        return 2
    args.output = output
    args.save = output
    args.height = float(args.height if args.height is not None else args.obstacle_height)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        profile = load_profile(profile_path)
        args.sim_step_dt = float(profile["simulation"]["outer_step_s"])
        turtlebot4_urdf = prepare_turtlebot4_urdf(args).resolve()
    except Exception as exc:
        print(f"failed to prepare V2.1 inputs: {exc}", file=sys.stderr)
        return 1
    actual_urdf_checksum = hashlib.sha256(turtlebot4_urdf.read_bytes()).hexdigest()
    if profile["source"].get("urdf_sha256") != actual_urdf_checksum:
        print("profile URDF checksum does not match the prepared URDF", file=sys.stderr)
        return 1
    if not visual_source_scene.is_file():
        print(f"visual source scene not found: {visual_source_scene}", file=sys.stderr)
        return 1
    obstacles, entities, metadata = prepare_scene_data(args)
    metadata = dict(metadata)
    defender_actuator = profile["robots"]["defender"]["actuator"]
    attacker_actuator = profile["robots"]["attacker"]["actuator"]
    metadata.update(
        {
            "obstacle_height_m": float(args.obstacle_height),
            "sensor_range_m": 3.0,
            "episode_len": int(getattr(map_config.EnvParameters, "EPISODE_LEN", 449)),
            "defender_speed_m_per_s": defender_actuator["max_linear_mps"],
            "attacker_speed_m_per_s": attacker_actuator["max_linear_mps"],
            "defender_max_turn_rad_per_s": defender_actuator["max_angular_radps"],
            "attacker_max_turn_rad_per_s": attacker_actuator["max_angular_radps"],
        }
    )
    manifest = build_manifest(
        profile,
        obstacles,
        entities,
        metadata,
        scale=float(args.scale),
        profile_path=profile_path,
    )
    manifest_path = output.with_suffix(".json")
    if args.write_manifest_only:
        temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
        temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary_manifest, manifest_path)
        print(f"wrote {manifest_path}")
        return 0

    coppeliasim_root = args.coppeliasim_root.resolve()
    launcher = coppeliasim_root / "coppeliaSim.sh"
    if not launcher.is_file():
        print(f"CoppeliaSim launcher not found: {launcher}", file=sys.stderr)
        return 2
    log_path = output.with_suffix(".build.log")
    with tempfile.TemporaryDirectory(prefix=f".{output.stem}.build-", dir=output.parent) as temporary:
        temporary_dir = Path(temporary)
        build_output = temporary_dir / output.name
        temporary_manifest = temporary_dir / manifest_path.name
        status_path = temporary_dir / "status.txt"
        addon_path = temporary_dir / "builder.lua"
        temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        status_path.write_text("pending\n", encoding="utf-8")
        addon_path.write_text(
            _scene_lua(build_output, status_path, manifest, turtlebot4_urdf, visual_source_scene),
            encoding="utf-8",
        )
        command = [str(launcher)]
        if not args.show_gui:
            command.append("-h")
        command.extend([f"-a{addon_path}", f"-Gpython={Path(sys.executable).resolve()}"])
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = f"{coppeliasim_root}:{environment.get('LD_LIBRARY_PATH', '')}"
        environment["QT_PLUGIN_PATH"] = str(coppeliasim_root)
        environment["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(coppeliasim_root / "platforms")
        if not args.show_gui:
            environment["QT_QPA_PLATFORM"] = "offscreen"
        environment.pop("QT_DEBUG_PLUGINS", None)
        try:
            completed = subprocess.run(
                command,
                cwd=coppeliasim_root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=float(args.timeout),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            log_path.write_text(str(exc.stdout or ""), encoding="utf-8")
            print(f"timed out while building {output}; existing assets were not replaced", file=sys.stderr)
            print(f"log {log_path}", file=sys.stderr)
            return 1
        log_path.write_text(completed.stdout, encoding="utf-8")
        status = status_path.read_text(encoding="utf-8").strip() if status_path.is_file() else "no status file"
        expected_status = f"saved {build_output}"
        if completed.returncode == 0 and status == expected_status and build_output.is_file() and build_output.stat().st_size > 0:
            os.replace(build_output, output)
            os.replace(temporary_manifest, manifest_path)
            print(f"saved {output}")
            print(f"wrote {manifest_path}")
            print(f"log {log_path}")
            return 0
        print(f"failed to save {output}: {status}", file=sys.stderr)
        print(f"log {log_path}", file=sys.stderr)
        return completed.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
