-- One-shot scene migration: restore URDF-imported differential-drive physics.

function sysCall_info()
    return {autoStart = true}
end

local sim = require('sim')
local migrated = false
local obstacleHeight = 0.5
local segmentIndices = {6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28}

local function alias(handle)
    return sim.getObjectAlias(handle, -1)
end

local function findByAlias(wanted)
    for _, handle in ipairs(sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)) do
        if alias(handle) == wanted then return handle end
    end
    return -1
end

local function configureRobot(role)
    local root = findByAlias('TrackMaker_' .. role)
    if root < 0 then error('missing robot root: ' .. role) end
    local baseName = 'TrackMaker_' .. role .. '_base_link'
    local leftJointName = 'TrackMaker_' .. role .. '_left_wheel_joint'
    local rightJointName = 'TrackMaker_' .. role .. '_right_wheel_joint'
    local physicalShapes = {
        [baseName] = true,
        left_wheel_respondable = true,
        right_wheel_respondable = true,
    }
    for _, handle in ipairs(sim.getObjectsInTree(root, sim.handle_all, 0)) do
        local name = alias(handle)
        if sim.getObjectType(handle) == sim.sceneobject_shape then
            local physical = physicalShapes[name] == true
            sim.setObjectInt32Param(handle, sim.shapeintparam_static, physical and 0 or 1)
            sim.setObjectInt32Param(handle, sim.shapeintparam_respondable, physical and 1 or 0)
        elseif sim.getObjectType(handle) == sim.sceneobject_joint
            and (name == leftJointName or name == rightJointName) then
                sim.setObjectInt32Param(handle, sim.jointintparam_dynctrlmode, sim.jointdynctrl_velocity)
                sim.setJointMode(handle, sim.jointmode_dynamic)
                sim.setJointTargetForce(handle, 12.0)
                sim.setJointTargetVelocity(handle, 0.0)
        end
    end
    local base = findByAlias(baseName)
    local leftJoint = findByAlias(leftJointName)
    local rightJoint = findByAlias(rightJointName)
    if base < 0 or leftJoint < 0 or rightJoint < 0 then error('missing drive train for ' .. role) end
    -- Remove the imported zero-travel wheel-drop links from the dynamic chain.
    sim.setObjectParent(leftJoint, base, true)
    sim.setObjectParent(rightJoint, base, true)
    -- The imported respondable base shape moves relative to its link dummy.
    -- Detach it first, then put the remaining visual/sensor tree under that
    -- physical shape so camera rendering, TF and dynamics share one pose.
    sim.setObjectParent(base, -1, true)
    sim.setObjectParent(root, base, true)
end

local function addSegmentCaps(index)
    local obstacleName = string.format('TrackMaker_turtlebot4_obstacle_%03d', index)
    local obstacle = findByAlias(obstacleName)
    if obstacle < 0 then error('missing segment obstacle: ' .. obstacleName) end
    local minX = sim.getObjectFloatParam(obstacle, sim.objfloatparam_objbbox_min_x)
    local maxX = sim.getObjectFloatParam(obstacle, sim.objfloatparam_objbbox_max_x)
    local minY = sim.getObjectFloatParam(obstacle, sim.objfloatparam_objbbox_min_y)
    local maxY = sim.getObjectFloatParam(obstacle, sim.objfloatparam_objbbox_max_y)
    local radius = 0.5 * (maxY - minY)
    local matrix = sim.getObjectMatrix(obstacle, sim.handle_world)
    for capIndex, localX in ipairs({minX, maxX}) do
        local capName = obstacleName .. '_cap_' .. tostring(capIndex)
        if findByAlias(capName) < 0 then
            local cap = sim.createPrimitiveShape(
                sim.primitiveshape_cylinder,
                {radius * 2.0, radius * 2.0, obstacleHeight},
                0
            )
            sim.setObjectAlias(cap, capName)
            sim.setObjectPosition(cap, sim.handle_world, {
                matrix[1] * localX + matrix[4],
                matrix[5] * localX + matrix[8],
                obstacleHeight * 0.5,
            })
            sim.setObjectInt32Param(cap, sim.shapeintparam_static, 1)
            sim.setObjectInt32Param(cap, sim.shapeintparam_respondable, 1)
            sim.setShapeColor(cap, nil, sim.colorcomponent_ambient_diffuse, {0.22, 0.24, 0.29})
        end
    end
end

function sysCall_beforeSimulation()
    if migrated then return end
    configureRobot('defender')
    configureRobot('attacker')
    for _, handle in ipairs(sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)) do
        if string.find(alias(handle), 'TrackMaker_turtlebot4_obstacle_', 1, true) == 1 then
            local currentHeight = sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_max_z)
                - sim.getObjectFloatParam(handle, sim.objfloatparam_objbbox_min_z)
            sim.scaleObject(handle, 1.0, 1.0, obstacleHeight / currentHeight, 0)
            local p = sim.getObjectPosition(handle, sim.handle_world)
            sim.setObjectPosition(handle, sim.handle_world, {p[1], p[2], obstacleHeight * 0.5})
        end
    end
    for _, index in ipairs(segmentIndices) do addSegmentCaps(index) end
    local scenePath = sim.getStringProperty(sim.handle_scene, 'scenePath')
    if type(scenePath) ~= 'string' or scenePath == '' then error('scenePath is empty') end
    sim.saveScene(scenePath)
    migrated = true
    sim.addLog(sim.verbosity_scriptinfos, '[TrackMaker] wheel physics scene migration saved: ' .. scenePath)
    sim.quitSimulator()
end
