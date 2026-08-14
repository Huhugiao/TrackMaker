-- TrackMaker CoppeliaSim 4.10 <-> ROS 2 Humble bridge.
-- The bridge deliberately drives wheel joints only; it never moves robot poses.

function sysCall_info()
    return {autoStart = true, menu = 'TrackMaker\nROS 2 bridge'}
end

local sim = require('sim')
local simROS2 = require('simROS2')

local commandTimeout = 0.5
local wheelRadius = 0.03575
local wheelSeparation = 0.233
local lidarRange = 3.0
local lidarRays = 64
local robots = {}
local publishers = {}
local subscriptions = {}
local obstacles = {}
local obstacleCollection = -1
local target = -1
local camera = -1
local cameraEnabled = true
local sensingCount = 0
local ready = false

local function log(message)
    sim.addLog(sim.verbosity_scriptinfos, '[TrackMakerROS2] ' .. tostring(message))
end

local function finite(value)
    return type(value) == 'number' and value == value and value ~= math.huge and value ~= -math.huge
end

local function clamp(value, limit)
    if not finite(value) then return 0.0 end
    return math.max(-limit, math.min(limit, value))
end

local function stamp(seconds)
    local sec = math.floor(seconds)
    local nanosec = math.floor((seconds - sec) * 1000000000.0 + 0.5)
    if nanosec >= 1000000000 then sec = sec + 1; nanosec = 0 end
    return {sec = sec, nanosec = nanosec}
end

local function alias(handle)
    local ok, value = pcall(sim.getObjectAlias, handle, -1)
    if ok then return value end
    return ''
end

local function findByAlias(wanted, objectType)
    for _, handle in ipairs(sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)) do
        if (objectType == nil or sim.getObjectType(handle) == objectType) and alias(handle) == wanted then
            return handle
        end
    end
    error('required scene object not found: ' .. wanted)
end

local function findOptionalByAlias(wanted)
    for _, handle in ipairs(sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)) do
        if alias(handle) == wanted then return handle end
    end
    return -1
end

local function quaternionWithYawOffset(q, offset)
    local s = math.sin(offset * 0.5)
    local c = math.cos(offset * 0.5)
    return {
        q[4] * 0.0 + q[1] * c + q[2] * s,
        q[4] * 0.0 - q[1] * s + q[2] * c,
        q[4] * s + q[3] * c,
        q[4] * c - q[3] * s,
    }
end

local function poseMessage(handle, frame, now, yawOffset)
    local p = sim.getObjectPosition(handle, sim.handle_world)
    local q = quaternionWithYawOffset(sim.getObjectQuaternion(handle, sim.handle_world), yawOffset or 0.0)
    return {
        header = {stamp = now, frame_id = 'map'},
        pose = {
            position = {x = p[1], y = p[2], z = p[3]},
            orientation = {x = q[1], y = q[2], z = q[3], w = q[4]},
        },
    }
end

local function transformMessage(handle, child, now, yawOffset)
    local p = sim.getObjectPosition(handle, sim.handle_world)
    local q = quaternionWithYawOffset(sim.getObjectQuaternion(handle, sim.handle_world), yawOffset or 0.0)
    return {
        header = {stamp = now, frame_id = 'map'},
        child_frame_id = child,
        transform = {
            translation = {x = p[1], y = p[2], z = p[3]},
            rotation = {x = q[1], y = q[2], z = q[3], w = q[4]},
        },
    }
end

local function rayMatrix(origin, angle)
    local dx = math.cos(angle)
    local dy = math.sin(angle)
    -- A Coppelia ray sensor detects along its local +Z axis.
    return {
        -dy, 0.0, dx, origin[1],
         dx, 0.0, dy, origin[2],
        0.0, 1.0, 0.0, origin[3],
    }
end

local function laserMessage(robot, now)
    -- The ray sensor is only a movable query object.  In older generated
    -- scenes it was parented to the model root at floor height, so using its
    -- own pose made scans depend on millimetre-scale suspension settling.
    -- Always cast from the real RPLIDAR link carried by the robot instead.
    local p = sim.getObjectPosition(robot.lidarOrigin, sim.handle_world)
    local o = sim.getObjectOrientation(robot.base, sim.handle_world)
    local origin = {p[1], p[2], p[3]}
    local angleMin = -math.pi
    local increment = 2.0 * math.pi / lidarRays
    local ranges = {}
    for index = 0, lidarRays - 1 do
        local relativeAngle = angleMin + index * increment
        sim.setObjectMatrix(robot.lidar, rayMatrix(origin, o[3] + robot.yawOffset + relativeAngle), sim.handle_world)
        local detected, distance = sim.handleProximitySensor(robot.lidar)
        if detected > 0 and finite(distance) then
            ranges[index + 1] = math.max(0.05, math.min(lidarRange, distance))
        else
            ranges[index + 1] = lidarRange
        end
    end
    return {
        header = {stamp = now, frame_id = robot.role .. '/laser'},
        angle_min = angleMin,
        angle_max = math.pi - increment,
        angle_increment = increment,
        time_increment = 0.0,
        scan_time = 0.1,
        range_min = 0.05,
        range_max = lidarRange,
        ranges = ranges,
        intensities = {},
    }
end

local function hasObstacleCollision(robot)
    for _, obstacle in ipairs(obstacles) do
        local result = sim.checkCollision(robot.collision, obstacle)
        if result > 0 then return true end
    end
    return false
end

local function setWheelSpeeds(robot, linear, angular)
    local v = clamp(linear, robot.maxLinear)
    local w = clamp(angular, robot.maxAngular)
    local left = (v - 0.5 * wheelSeparation * w) / wheelRadius
    local right = (v + 0.5 * wheelSeparation * w) / wheelRadius
    sim.setJointTargetVelocity(robot.leftWheel, left)
    sim.setJointTargetVelocity(robot.rightWheel, right)
end

local function configureVelocityJoint(handle)
    pcall(sim.setJointMode, handle, sim.jointmode_dynamic)
    pcall(sim.setObjectInt32Param, handle, sim.jointintparam_dynctrlmode, sim.jointdynctrl_velocity)
    pcall(sim.setJointTargetForce, handle, 12.0)
    pcall(sim.setJointTargetVelocity, handle, 0.0)
end

local function receiveCommand(role, message)
    local robot = robots[role]
    if robot == nil then return end
    local linear = message.linear and message.linear.x or 0.0
    local angular = message.angular and message.angular.z or 0.0
    if not finite(linear) or not finite(angular) then
        robot.command = {linear = 0.0, angular = 0.0}
        log(role .. ' rejected non-finite cmd_vel')
    else
        robot.command = {
            linear = clamp(linear, robot.maxLinear),
            angular = clamp(angular, robot.maxAngular),
        }
    end
    robot.lastCommandTime = sim.getSimulationTime()
end

function trackmakerDefenderCmdVel(message)
    receiveCommand('defender', message)
end

function trackmakerAttackerCmdVel(message)
    receiveCommand('attacker', message)
end

local function addPublisher(topic, messageType, key)
    publishers[key] = simROS2.createPublisher(topic, messageType)
end

local function configureRobot(role, maxLinear, maxAngular, callback)
    local robot = {
        role = role,
        base = findByAlias('TrackMaker_' .. role .. '_base_link'),
        -- Use the same physical base shape that the dynamics engine drives.
        -- Legacy scenes also contain a visual collision proxy, but after the
        -- wheel-physics migration it is not part of the dynamic base tree.
        collision = -1,
        leftWheel = findByAlias('TrackMaker_' .. role .. '_left_wheel_joint', sim.sceneobject_joint),
        rightWheel = findByAlias('TrackMaker_' .. role .. '_right_wheel_joint', sim.sceneobject_joint),
        lidar = findByAlias('TrackMaker_' .. role .. '_rplidar_sensor', sim.sceneobject_proximitysensor),
        lidarOrigin = findByAlias('TrackMaker_' .. role .. '_rplidar_link', sim.sceneobject_shape),
        maxLinear = maxLinear,
        maxAngular = maxAngular,
        -- The imported base collision cylinder carries the URDF geometry's
        -- +90 degree visual origin. ROS base_link itself has no such offset.
        yawOffset = -math.pi * 0.5,
        command = {linear = 0.0, angular = 0.0},
        lastCommandTime = -math.huge,
    }
    robot.collision = robot.base
    robots[role] = robot
    configureVelocityJoint(robot.leftWheel)
    configureVelocityJoint(robot.rightWheel)
    sim.setObjectInt32Param(robot.lidar, sim.proxintparam_entity_to_detect, obstacleCollection)
    setWheelSpeeds(robot, 0.0, 0.0)
    addPublisher('/tracking/' .. role .. '/pose', 'geometry_msgs/msg/PoseStamped', role .. 'Pose')
    addPublisher('/' .. role .. '/scan', 'sensor_msgs/msg/LaserScan', role .. 'Scan')
    addPublisher('/' .. role .. '/collision', 'std_msgs/msg/Bool', role .. 'Collision')
    subscriptions[role] = simROS2.createSubscription('/' .. role .. '/cmd_vel', 'geometry_msgs/msg/Twist', callback)
end

local function applyNamedSpawn()
    local packed = sim.getStringProperty(sim.handle_app, 'namedParam.trackmakerSpawn', {noError = true})
    if packed == nil or packed == '' then return end
    local values = {}
    for token in string.gmatch(packed, '[^,]+') do
        values[#values + 1] = tonumber(token)
    end
    if #values ~= 8 then error('trackmakerSpawn must contain eight comma-separated finite numbers') end
    for _, value in ipairs(values) do
        if not finite(value) then error('trackmakerSpawn contains a non-finite value') end
    end

    local function placeRobot(robot, x, y, yaw)
        local p = sim.getObjectPosition(robot.base, sim.handle_world)
        sim.setObjectPosition(robot.base, sim.handle_world, {x, y, p[3]})
        sim.setObjectOrientation(robot.base, sim.handle_world, {0.0, 0.0, yaw - robot.yawOffset})
        pcall(sim.resetDynamicObject, robot.base)
        setWheelSpeeds(robot, 0.0, 0.0)
    end

    placeRobot(robots.defender, values[1], values[2], values[3])
    placeRobot(robots.attacker, values[4], values[5], values[6])
    local targetPosition = sim.getObjectPosition(target, sim.handle_world)
    sim.setObjectPosition(target, sim.handle_world, {values[7], values[8], targetPosition[3]})
    log('applied deterministic paired spawn: ' .. packed)
end

local function createDemoCamera(view)
    local oblique = view == 'oblique'
    local sensor = sim.createVisionSensor(
        1 + 4 + 64 + 128 + (oblique and 2 or 0),
        {960, 540, 0, 0},
        {0.05, 16.0, oblique and math.rad(58.0) or 11.8, 0.01, 0.0, 0.0, 0.94, 0.94, 0.94, 0.0, 0.0}
    )
    sim.setObjectAlias(sensor, 'TrackMaker_demo_camera')
    if oblique then
        sim.setObjectPosition(sensor, sim.handle_world, {0.0, -7.2, 5.4})
        sim.setObjectOrientation(sensor, sim.handle_world, {math.pi + math.atan(7.2, 5.2), 0.0, 0.0})
    else
        -- 11.8 m orthographic width fits the 6.4 m square arena in a 16:9 frame.
        sim.setObjectPosition(sensor, sim.handle_world, {0.0, 0.0, 7.0})
        sim.setObjectOrientation(sensor, sim.handle_world, {math.pi, 0.0, 0.0})
    end
    return sensor
end

function sysCall_init()
    -- Add-ons are initialized before the command-line scene is loaded.
    -- Scene object discovery therefore belongs in sysCall_beforeSimulation.
    ready = false
end

function sysCall_beforeSimulation()
    if ready then return end
    pcall(sim.setBoolParam, sim.boolparam_realtime_simulation, true)
    pcall(sim.setFloatParam, sim.floatparam_simulation_time_step, 0.05)

    target = findByAlias('TrackMaker_target')
    for _, handle in ipairs(sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)) do
        if string.find(alias(handle), 'TrackMaker_turtlebot4_obstacle_', 1, true) == 1 then
            obstacles[#obstacles + 1] = handle
        end
    end
    obstacleCollection = sim.createCollection(1)
    for _, handle in ipairs(obstacles) do
        sim.addItemToCollection(obstacleCollection, sim.handle_single, handle, 0)
    end
    configureRobot('defender', 0.234, math.rad(54.0), 'trackmakerDefenderCmdVel')
    configureRobot('attacker', 0.180, math.rad(108.0), 'trackmakerAttackerCmdVel')
    applyNamedSpawn()
    addPublisher('/tracking/target/pose', 'geometry_msgs/msg/PoseStamped', 'targetPose')
    addPublisher('/clock', 'rosgraph_msgs/msg/Clock', 'clock')
    local cameraParam = sim.getStringProperty(sim.handle_app, 'namedParam.trackmakerCamera', {noError = true})
    local cameraView = sim.getStringProperty(sim.handle_app, 'namedParam.trackmakerCameraView', {noError = true}) or 'overhead'
    cameraEnabled = cameraParam ~= '0'
    if cameraEnabled then
        addPublisher('/demo/camera/image_raw', 'sensor_msgs/msg/Image', 'camera')
        simROS2.publisherTreatUInt8ArrayAsString(publishers.camera)
        camera = createDemoCamera(cameraView)
    end
    ready = true
    log('ready: wheel joint control, truth poses, LaserScan, TF, /clock; camera=' .. tostring(cameraEnabled) .. ', view=' .. cameraView)
end

function sysCall_actuation()
    if not ready then return end
    local now = sim.getSimulationTime()
    for _, role in ipairs({'defender', 'attacker'}) do
        local robot = robots[role]
        if now - robot.lastCommandTime > commandTimeout then
            setWheelSpeeds(robot, 0.0, 0.0)
        else
            setWheelSpeeds(robot, robot.command.linear, robot.command.angular)
        end
    end
end

function sysCall_sensing()
    if not ready then return end
    sensingCount = sensingCount + 1
    local simulationTime = sim.getSimulationTime()
    local now = stamp(simulationTime)
    simROS2.publish(publishers.clock, {clock = now})

    local transforms = {}
    for _, role in ipairs({'defender', 'attacker'}) do
        local robot = robots[role]
        simROS2.publish(publishers[role .. 'Pose'], poseMessage(robot.base, role, now, robot.yawOffset))
        simROS2.publish(publishers[role .. 'Collision'], {data = hasObstacleCollision(robot)})
        transforms[#transforms + 1] = transformMessage(robot.base, role .. '/base_link', now, robot.yawOffset)
        transforms[#transforms + 1] = transformMessage(robot.lidarOrigin, role .. '/laser', now, robot.yawOffset)
        if sensingCount % 2 == 0 then
            simROS2.publish(publishers[role .. 'Scan'], laserMessage(robot, now))
        end
    end
    simROS2.publish(publishers.targetPose, poseMessage(target, 'target', now))
    transforms[#transforms + 1] = transformMessage(target, 'target/base_link', now)
    simROS2.sendTransforms(transforms)

    if cameraEnabled then
        sim.handleVisionSensor(camera)
        local image, resolution = sim.getVisionSensorImg(camera)
        simROS2.publish(publishers.camera, {
            header = {stamp = now, frame_id = 'demo_camera'},
            height = resolution[2], width = resolution[1], encoding = 'rgb8',
            is_bigendian = 0, step = resolution[1] * 3, data = image,
        })
    end
end

function sysCall_cleanup()
    if not ready then return end
    for _, role in ipairs({'defender', 'attacker'}) do
        if robots[role] ~= nil then pcall(setWheelSpeeds, robots[role], 0.0, 0.0) end
    end
    for _, handle in pairs(publishers) do pcall(simROS2.shutdownPublisher, handle) end
    for _, handle in pairs(subscriptions) do pcall(simROS2.shutdownSubscription, handle) end
    if camera >= 0 then pcall(sim.removeObjects, {camera}) end
    ready = false
end
