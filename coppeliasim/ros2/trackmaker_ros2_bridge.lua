-- TrackMaker calibratable digital twin V2.1 <-> ROS 2 Humble bridge.
-- Runtime motion is wheel-joint-only.  Pose writes are restricted to the
-- stopped-simulation spawn configuration in sysCall_beforeSimulation.

function sysCall_info()
    return {autoStart = true, menu = 'TrackMaker\nROS 2 V2.1 bridge'}
end

local sim = require('sim')
local simROS2 = require('simROS2')
local json = require('dkjson')

local MODULUS = 2147483647
local MULTIPLIER = 16807
local lidarRange = 3.0
local lidarRays = 64
local profile = nil
local robots = {}
local publishers = {}
local subscriptions = {}
local obstacles = {}
local obstacleSet = {}
local obstacleCollection = -1
local floor = -1
local floorTop = 0.02
local target = -1
local camera = -1
local cameraEnabled = true
local sensingCount = 0
local ready = false
local simulationStart = 0.0
local runtimeMetadata = nil

local function log(message)
    sim.addLog(sim.verbosity_scriptinfos, '[TrackMakerROS2V2] ' .. tostring(message))
end

local function finite(value)
    return type(value) == 'number' and value == value and value ~= math.huge and value ~= -math.huge
end

local function close(actual, expected, tolerance)
    return finite(actual) and finite(expected) and math.abs(actual - expected) <= (tolerance or 1e-6)
end

local function clamp(value, limit)
    if not finite(value) then return 0.0 end
    return math.max(-math.abs(limit), math.min(math.abs(limit), value))
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

local function findPhysical(role, kind, objectType)
    for _, handle in ipairs(sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)) do
        if objectType == nil or sim.getObjectType(handle) == objectType then
            local physicalRole = sim.getStringProperty(handle, 'customData.trackmakerPhysicalRole', {noError = true})
            local physicalKind = sim.getStringProperty(handle, 'customData.trackmakerPhysicalKind', {noError = true})
            if physicalRole == role and physicalKind == kind then return handle end
        end
    end
    error('required tagged physical object not found: ' .. role .. '/' .. kind)
end

local function getNamedString(name)
    return sim.getStringProperty(sim.handle_app, 'namedParam.' .. name, {noError = true})
end

local function newRng(seed)
    local state = math.floor(seed) % MODULUS
    if state <= 0 then state = state + MODULUS - 1 end
    return function()
        state = (state * MULTIPLIER) % MODULUS
        return state / MODULUS
    end
end

local function quaternionWithYawOffset(q, offset)
    local s, c = math.sin(offset * 0.5), math.cos(offset * 0.5)
    return {
        q[1] * c + q[2] * s,
        -q[1] * s + q[2] * c,
        q[4] * s + q[3] * c,
        q[4] * c - q[3] * s,
    }
end

local function poseMessage(handle, now, yawOffset)
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
    local dx, dy = math.cos(angle), math.sin(angle)
    return {
        -dy, 0.0, dx, origin[1],
         dx, 0.0, dy, origin[2],
        0.0, 1.0, 0.0, origin[3],
    }
end

local function laserMessage(robot, now)
    local p = sim.getObjectPosition(robot.lidarOrigin, sim.handle_world)
    local o = sim.getObjectOrientation(robot.baseFrame, sim.handle_world)
    local origin = {p[1], p[2], p[3]}
    local angleMin = -math.pi
    local increment = 2.0 * math.pi / lidarRays
    local ranges = {}
    for index = 0, lidarRays - 1 do
        local relativeAngle = angleMin + index * increment
        sim.setObjectMatrix(robot.lidar, rayMatrix(origin, o[3] + relativeAngle), sim.handle_world)
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

local function addPublisher(topic, messageType, key, qos)
    publishers[key] = simROS2.createPublisher(topic, messageType, 0, false, qos)
end

local function transientLocalQos()
    local zero = {sec = 0, nanosec = 0}
    return {
        history = simROS2.qos_history_policy.keep_last,
        depth = 1,
        reliability = simROS2.qos_reliability_policy.reliable,
        durability = simROS2.qos_durability_policy.transient_local,
        deadline = zero,
        lifespan = zero,
        liveliness = simROS2.qos_liveliness_policy.automatic,
        liveliness_lease_duration = zero,
        avoid_ros_namespace_conventions = false,
    }
end

local function publishEvent(robot, event)
    event.role = robot.role
    event.profile_id = profile.profile_id
    event.profile_checksum = profile.checksum
    simROS2.publish(publishers[robot.role .. 'Events'], {data = json.encode(event)})
end

local function dropoutActive(robot, received)
    local elapsed = received - simulationStart
    for _, window in ipairs(robot.actuator.dropout_windows_s or {}) do
        if elapsed >= window[1] and elapsed < window[2] then return true end
    end
    return false
end

local function receiveCommand(role, message)
    local robot = robots[role]
    if robot == nil then return end
    local received = sim.getSimulationTime()
    local linear = message.linear and message.linear.x or 0.0
    local angular = message.angular and message.angular.z or 0.0
    local isFinite = finite(linear) and finite(angular)
    if not isFinite then linear, angular = 0.0, 0.0 end
    local clampedLinear = clamp(linear, robot.actuator.max_linear_mps)
    local clampedAngular = clamp(angular, robot.actuator.max_angular_radps)
    local jitter = (2.0 * robot.random() - 1.0) * robot.actuator.uniform_jitter_s
    local scheduled = math.max(received, received + robot.actuator.fixed_delay_s + jitter)
    local presetDrop = dropoutActive(robot, received)
    local randomDrop = robot.random() < robot.actuator.packet_loss_probability
    local dropped = presetDrop or randomDrop
    robot.sequence = robot.sequence + 1
    local event = {
        phase = 'received',
        sequence = robot.sequence,
        receive_time_s = received,
        scheduled_time_s = scheduled,
        execute_time_s = json.null,
        requested_linear_mps = linear,
        requested_angular_radps = angular,
        clamped_linear_mps = clampedLinear,
        clamped_angular_radps = clampedAngular,
        finite = isFinite,
        jitter_s = jitter,
        dropped = dropped,
        drop_reason = presetDrop and 'preset_outage' or (randomDrop and 'packet_loss' or ''),
    }
    publishEvent(robot, event)
    if dropped then
        robot.droppedCount = robot.droppedCount + 1
        return
    end
    robot.queue[#robot.queue + 1] = event
    table.sort(robot.queue, function(first, second)
        if first.scheduled_time_s == second.scheduled_time_s then return first.sequence < second.sequence end
        return first.scheduled_time_s < second.scheduled_time_s
    end)
end

function trackmakerDefenderCmdVel(message)
    receiveCommand('defender', message)
end

function trackmakerAttackerCmdVel(message)
    receiveCommand('attacker', message)
end

local function deadband(value, gain, width)
    if value == 0.0 then return 0.0 end
    local sign = value > 0.0 and 1.0 or -1.0
    return sign * math.max(0.0, math.abs(value) - width) * gain
end

local function slew(current, desired, dt, acceleration, braking)
    local accelerating = current * desired >= 0.0 and math.abs(desired) > math.abs(current)
    local limit = accelerating and acceleration or braking
    local delta = clamp(desired - current, limit * dt)
    return current + delta
end

local function actuatorStep(robot, now, dt)
    local executed = {}
    while #robot.queue > 0 and robot.queue[1].scheduled_time_s <= now + 1e-12 do
        local event = table.remove(robot.queue, 1)
        event.execute_time_s = now
        robot.commandLinear = event.clamped_linear_mps
        robot.commandAngular = event.clamped_angular_radps
        robot.lastExecutionTime = now
        executed[#executed + 1] = event
        local lateness = now - event.scheduled_time_s
        robot.lastDelay = now - event.receive_time_s
        if lateness > profile.simulation.physics_step_s + 1e-9 then
            robot.deadlineMissCount = robot.deadlineMissCount + 1
        end
    end

    local watchdog = now - robot.lastExecutionTime > robot.actuator.watchdog_timeout_s
    if watchdog ~= robot.watchdog then
        publishEvent(robot, {
            phase = 'watchdog',
            time_s = now,
            watchdog_active = watchdog,
            last_execution_time_s = finite(robot.lastExecutionTime) and robot.lastExecutionTime or json.null,
        })
    end
    robot.watchdog = watchdog
    local linear = watchdog and 0.0 or robot.commandLinear
    local angular = watchdog and 0.0 or robot.commandAngular
    local alpha = robot.actuator.time_constant_s <= 1e-12 and 1.0 or (1.0 - math.exp(-dt / robot.actuator.time_constant_s))
    robot.lag[1] = robot.lag[1] + alpha * (linear - robot.lag[1])
    robot.lag[2] = robot.lag[2] + alpha * (angular - robot.lag[2])
    robot.bodyLinear = slew(
        robot.bodyLinear, robot.lag[1], dt,
        robot.actuator.acceleration_mps2, robot.actuator.braking_mps2
    )
    local halfTrack = robot.wheelSeparation * 0.5
    local requestedLeft = robot.bodyLinear - halfTrack * robot.lag[2]
    local requestedRight = robot.bodyLinear + halfTrack * robot.lag[2]
    robot.output = {
        deadband(requestedLeft, robot.actuator.left_gain, robot.actuator.left_deadband_mps),
        deadband(requestedRight, robot.actuator.right_gain, robot.actuator.right_deadband_mps),
    }
    robot.targetWheel = {robot.output[1] / robot.wheelRadius, robot.output[2] / robot.wheelRadius}
    sim.setJointTargetVelocity(robot.leftWheelJoint, robot.targetWheel[1])
    sim.setJointTargetVelocity(robot.rightWheelJoint, robot.targetWheel[2])
    robot.actualWheel = {sim.getJointVelocity(robot.leftWheelJoint), sim.getJointVelocity(robot.rightWheelJoint)}
    robot.filteredLinear = 0.5 * (robot.output[1] + robot.output[2])
    robot.filteredAngular = (robot.output[2] - robot.output[1]) / robot.wheelSeparation

    for _, event in ipairs(executed) do
        event.phase = 'executed'
        event.actual_delay_s = event.execute_time_s - event.receive_time_s
        event.filtered_linear_mps = robot.filteredLinear
        event.filtered_angular_radps = robot.filteredAngular
        event.target_wheel_radps = robot.targetWheel
        event.actual_wheel_radps = robot.actualWheel
        publishEvent(robot, event)
    end
end

local function configureVelocityJoint(handle, force)
    sim.setJointMode(handle, sim.jointmode_dynamic)
    sim.setObjectInt32Param(handle, sim.jointintparam_dynctrlmode, sim.jointdynctrl_velocity)
    sim.setJointTargetForce(handle, force)
    sim.setJointTargetVelocity(handle, 0.0)
end

local function configureRobot(role, callback, seedOffset)
    local configured = profile.robots[role]
    local seedText = getNamedString('trackmakerSeed') or '0'
    local episodeSeed = tonumber(seedText) or 0
    local chassis = findByAlias('TrackMaker_' .. role .. '_chassis', sim.sceneobject_shape)
    local lidar = findByAlias('TrackMaker_' .. role .. '_rplidar_sensor', sim.sceneobject_proximitysensor)
    local robot = {
        role = role,
        baseFrame = chassis,
        chassis = chassis,
        caster = findByAlias('TrackMaker_' .. role .. '_front_caster', sim.sceneobject_shape),
        leftWheelJoint = findByAlias('TrackMaker_' .. role .. '_left_wheel_joint', sim.sceneobject_joint),
        rightWheelJoint = findByAlias('TrackMaker_' .. role .. '_right_wheel_joint', sim.sceneobject_joint),
        leftWheel = findByAlias('TrackMaker_' .. role .. '_left_wheel', sim.sceneobject_shape),
        rightWheel = findByAlias('TrackMaker_' .. role .. '_right_wheel', sim.sceneobject_shape),
        lidar = lidar,
        lidarOrigin = lidar,
        actuator = configured.actuator,
        geometry = configured.geometry,
        wheelRadius = configured.wheel_radius_m,
        wheelSeparation = configured.wheel_separation_m,
        random = newRng(profile.seed + episodeSeed * 2 + seedOffset),
        queue = {}, sequence = 0,
        commandLinear = 0.0, commandAngular = 0.0,
        lag = {0.0, 0.0}, bodyLinear = 0.0, output = {0.0, 0.0},
        targetWheel = {0.0, 0.0}, actualWheel = {0.0, 0.0},
        filteredLinear = 0.0, filteredAngular = 0.0,
        lastExecutionTime = -math.huge,
        lastDelay = 0.0,
        watchdog = true,
        droppedCount = 0,
        deadlineMissCount = 0,
        contact = {},
    }
    robots[role] = robot
    configureVelocityJoint(robot.leftWheelJoint, robot.actuator.target_force_nm)
    configureVelocityJoint(robot.rightWheelJoint, robot.actuator.target_force_nm)
    sim.setObjectInt32Param(robot.lidar, sim.proxintparam_entity_to_detect, obstacleCollection)
    addPublisher('/tracking/' .. role .. '/pose', 'geometry_msgs/msg/PoseStamped', role .. 'Pose')
    addPublisher('/' .. role .. '/scan', 'sensor_msgs/msg/LaserScan', role .. 'Scan')
    addPublisher('/' .. role .. '/collision', 'std_msgs/msg/Bool', role .. 'Collision')
    addPublisher('/' .. role .. '/joint_targets', 'sensor_msgs/msg/JointState', role .. 'JointTargets')
    addPublisher('/' .. role .. '/joint_states', 'sensor_msgs/msg/JointState', role .. 'JointStates')
    addPublisher('/' .. role .. '/actuator_events', 'std_msgs/msg/String', role .. 'Events')
    addPublisher('/' .. role .. '/actuator_state', 'std_msgs/msg/String', role .. 'ActuatorState')
    subscriptions[role] = simROS2.createSubscription('/' .. role .. '/cmd_vel', 'geometry_msgs/msg/Twist', callback)
    return robot
end

local function requireClose(name, actual, expected, tolerance)
    if not close(actual, expected, tolerance) then
        error(name .. ' mismatch: expected ' .. tostring(expected) .. ', got ' .. tostring(actual))
    end
end

local function materialReadback(handle)
    return {
        friction = sim.getFloatProperty(handle, 'bullet.friction'),
        restitution = sim.getFloatProperty(handle, 'bullet.restitution'),
        linear_damping = sim.getFloatProperty(handle, 'bullet.linearDamping'),
        angular_damping = sim.getFloatProperty(handle, 'bullet.angularDamping'),
    }
end

local function verifyMaterial(name, handle, expected)
    local actual = materialReadback(handle)
    for key, value in pairs(expected) do requireClose(name .. '.' .. key, actual[key], value, 1e-6) end
    return actual
end

local function verifyRobot(robot)
    local expected = profile.robots[robot.role]
    local checksum = sim.getStringProperty(robot.chassis, 'customData.trackmakerProfileChecksum')
    if checksum ~= profile.checksum then error(robot.role .. ' chassis profile checksum mismatch') end
    local mass = sim.getShapeMass(robot.chassis)
    requireClose(robot.role .. '.chassis_mass', mass, expected.chassis.mass_kg, 1e-5)
    local inertia, com = sim.getShapeInertia(robot.chassis)
    local baseCenter = expected.geometry.base.center_m
    local expectedCom = {
        expected.chassis.center_of_mass_m[1] - baseCenter[1],
        expected.chassis.center_of_mass_m[2] - baseCenter[2],
        expected.chassis.center_of_mass_m[3] - baseCenter[3],
    }
    for index = 1, 9 do requireClose(robot.role .. '.chassis_inertia[' .. index .. ']', inertia[index], expected.chassis.inertia_kg_m2[index], 1e-5) end
    requireClose(robot.role .. '.chassis_com.x', com[4], expectedCom[1], 1e-5)
    requireClose(robot.role .. '.chassis_com.y', com[8], expectedCom[2], 1e-5)
    requireClose(robot.role .. '.chassis_com.z', com[12], expectedCom[3], 1e-5)
    requireClose(robot.role .. '.caster_mass', sim.getShapeMass(robot.caster), expected.caster.mass_kg, 1e-6)
    requireClose(robot.role .. '.left_wheel_mass', sim.getShapeMass(robot.leftWheel), expected.wheel.mass_kg_each, 1e-6)
    requireClose(robot.role .. '.right_wheel_mass', sim.getShapeMass(robot.rightWheel), expected.wheel.mass_kg_each, 1e-6)
    local casterInertia, casterCom = sim.getShapeInertia(robot.caster)
    local leftInertia, leftCom = sim.getShapeInertia(robot.leftWheel)
    local rightInertia, rightCom = sim.getShapeInertia(robot.rightWheel)
    for index = 1, 9 do
        requireClose(robot.role .. '.caster_inertia[' .. index .. ']', casterInertia[index], expected.caster.inertia_kg_m2[index], 1e-6)
        requireClose(robot.role .. '.left_wheel_inertia[' .. index .. ']', leftInertia[index], expected.wheel.inertia_kg_m2_each[index], 1e-6)
        requireClose(robot.role .. '.right_wheel_inertia[' .. index .. ']', rightInertia[index], expected.wheel.inertia_kg_m2_each[index], 1e-6)
    end
    for index, matrixIndex in ipairs({4, 8, 12}) do
        requireClose(robot.role .. '.caster_com[' .. index .. ']', casterCom[matrixIndex], expected.caster.center_of_mass_m[index], 1e-6)
        requireClose(robot.role .. '.left_wheel_com[' .. index .. ']', leftCom[matrixIndex], 0.0, 1e-6)
        requireClose(robot.role .. '.right_wheel_com[' .. index .. ']', rightCom[matrixIndex], 0.0, 1e-6)
    end
    if robot.leftWheelJoint == robot.rightWheelJoint then error(robot.role .. ' drive wheels do not have independent joints') end
    if sim.getJointType(robot.leftWheelJoint) ~= sim.joint_revolute or sim.getJointType(robot.rightWheelJoint) ~= sim.joint_revolute then
        error(robot.role .. ' drive wheel joint type mismatch')
    end
    local casterJoint = sim.getObjectParent(robot.caster)
    if sim.getObjectType(casterJoint) ~= sim.sceneobject_joint or sim.getJointType(casterJoint) ~= sim.joint_spherical then
        error(robot.role .. ' caster is not mounted by a spherical joint')
    end
    -- Track is defined by the revolute joint axes.  Shape reference frames can
    -- move slightly when CoppeliaSim reloads compound dynamics state.
    local left = sim.getObjectPosition(robot.leftWheelJoint, robot.baseFrame)
    local right = sim.getObjectPosition(robot.rightWheelJoint, robot.baseFrame)
    local separation = math.sqrt((left[1] - right[1]) ^ 2 + (left[2] - right[2]) ^ 2 + (left[3] - right[3]) ^ 2)
    requireClose(robot.role .. '.wheel_separation', separation, expected.wheel_separation_m, 1e-5)
    local radius = sim.getFloatProperty(robot.chassis, 'customData.trackmakerWheelRadius')
    requireClose(robot.role .. '.wheel_radius', radius, expected.wheel_radius_m, 1e-9)
    local targetForce = sim.getJointTargetForce(robot.leftWheelJoint)
    requireClose(robot.role .. '.target_force', targetForce, expected.actuator.target_force_nm, 1e-6)
    requireClose(robot.role .. '.right_target_force', sim.getJointTargetForce(robot.rightWheelJoint), expected.actuator.target_force_nm, 1e-6)
    verifyMaterial(robot.role .. '.chassis_material', robot.chassis, profile.materials.chassis)
    verifyMaterial(robot.role .. '.caster_material', robot.caster, profile.materials.caster)
    verifyMaterial(robot.role .. '.left_wheel_material', robot.leftWheel, profile.materials.wheel)
    verifyMaterial(robot.role .. '.right_wheel_material', robot.rightWheel, profile.materials.wheel)
    return {
        chassis_mass_kg = mass,
        chassis_center_of_mass_m = {com[4] + baseCenter[1], com[8] + baseCenter[2], com[12] + baseCenter[3]},
        chassis_inertia_frame = 'base_link',
        chassis_inertia_kg_m2 = inertia,
        caster_mass_kg = sim.getShapeMass(robot.caster),
        caster_center_of_mass_m = {casterCom[4], casterCom[8], casterCom[12]},
        caster_inertia_kg_m2 = casterInertia,
        caster_inertia_frame = 'caster_link',
        caster_mount_model = 'passive_spherical_joint',
        wheel_mass_kg_each = sim.getShapeMass(robot.leftWheel),
        left_wheel_inertia_kg_m2 = leftInertia,
        right_wheel_inertia_kg_m2 = rightInertia,
        wheel_inertia_frame = 'wheel_link',
        wheel_radius_m = radius,
        wheel_separation_m = separation,
        target_force_nm = targetForce,
    }
end

local function verifyRuntime()
    local sceneChecksum = sim.getStringProperty(sim.handle_scene, 'customData.trackmakerProfileChecksum')
    if sceneChecksum ~= profile.checksum then error('scene profile checksum mismatch') end
    local requestedChecksum = getNamedString('trackmakerProfileChecksum')
    if requestedChecksum ~= nil and requestedChecksum ~= '' and requestedChecksum ~= profile.checksum then
        error('requested profile checksum does not match scene')
    end
    local engine = sim.getIntArrayProperty(sim.handle_scene, 'dynamicsEngine')
    if engine[1] ~= profile.simulation.engine_index then error('runtime physics engine is not Bullet') end
    if engine[2] ~= profile.simulation.engine_version then error('runtime physics engine version mismatch') end
    local outerStep = sim.getFloatParam(sim.floatparam_simulation_time_step)
    local physicsStep = sim.getFloatProperty(sim.handle_scene, 'dynamicsStepSize')
    local solver = sim.getIntProperty(sim.handle_scene, 'bullet.iterations')
    local gravity = sim.getVector3Property(sim.handle_scene, 'gravity')
    requireClose('outer_step_s', outerStep, profile.simulation.outer_step_s, 1e-9)
    requireClose('physics_step_s', physicsStep, profile.simulation.physics_step_s, 1e-9)
    if solver ~= profile.simulation.solver_iterations then error('runtime solver iteration mismatch') end
    for index = 1, 3 do requireClose('gravity[' .. index .. ']', gravity[index], profile.simulation.gravity_mps2[index], 1e-9) end
    verifyMaterial('floor_material', floor, profile.materials.floor)
    for _, obstacle in ipairs(obstacles) do verifyMaterial('obstacle_material', obstacle, profile.materials.obstacle) end
    return {
        profile_checksum = profile.checksum,
        engine_index = engine[1],
        engine_version = engine[2],
        outer_step_s = outerStep,
        physics_step_s = physicsStep,
        solver_iterations = solver,
        gravity_mps2 = gravity,
        robots = {
            defender = verifyRobot(robots.defender),
            attacker = verifyRobot(robots.attacker),
        },
        introspection_passed = true,
    }
end

local function applyNamedSpawn()
    local packed = getNamedString('trackmakerSpawn')
    if packed == nil or packed == '' then return end
    local values = {}
    for token in string.gmatch(packed, '[^,]+') do values[#values + 1] = tonumber(token) end
    if #values ~= 8 then error('trackmakerSpawn must contain eight comma-separated finite numbers') end
    for _, value in ipairs(values) do if not finite(value) then error('trackmakerSpawn contains a non-finite value') end end

    local function placeRobot(robot, x, y, yaw)
        local p = sim.getObjectPosition(robot.chassis, sim.handle_world)
        sim.setObjectPose(robot.chassis, sim.handle_world, {x, y, p[3], 0.0, 0.0, math.sin(yaw * 0.5), math.cos(yaw * 0.5)})
        sim.resetDynamicObject(robot.chassis | sim.handleflag_model)
        sim.setJointTargetVelocity(robot.leftWheelJoint, 0.0)
        sim.setJointTargetVelocity(robot.rightWheelJoint, 0.0)
        local chassisPosition = sim.getObjectPosition(robot.chassis, sim.handle_world)
        local basePosition = sim.getObjectPosition(robot.baseFrame, sim.handle_world)
        local baseOrientation = sim.getObjectOrientation(robot.baseFrame, sim.handle_world)
        requireClose(robot.role .. '.spawn_chassis_x', chassisPosition[1], x, 1e-9)
        requireClose(robot.role .. '.spawn_chassis_y', chassisPosition[2], y, 1e-9)
        requireClose(robot.role .. '.spawn_base_x', basePosition[1], x, 1e-9)
        requireClose(robot.role .. '.spawn_base_y', basePosition[2], y, 1e-9)
        requireClose(robot.role .. '.spawn_base_yaw', baseOrientation[3], yaw, 1e-9)
    end

    placeRobot(robots.defender, values[1], values[2], values[3])
    placeRobot(robots.attacker, values[4], values[5], values[6])
    local targetPosition = sim.getObjectPosition(target, sim.handle_world)
    sim.setObjectPosition(target, sim.handle_world, {values[7], values[8], targetPosition[3]})
    log('applied stopped-simulation paired spawn: ' .. packed)
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
        sim.setObjectPosition(sensor, sim.handle_world, {0.0, 0.0, 7.0})
        sim.setObjectOrientation(sensor, sim.handle_world, {math.pi, 0.0, 0.0})
    end
    return sensor
end

local function contactPartners(handle)
    local partners = {}
    local index = 0
    while true do
        local colliding = sim.getContactInfo(sim.handle_all, handle, index)
        if colliding == nil or #colliding < 2 then break end
        local partner = colliding[1] == handle and colliding[2] or colliding[1]
        partners[partner] = true
        index = index + 1
    end
    return partners
end

local function chassisPhysicalMinZ(robot)
    local matrix = sim.getObjectMatrix(robot.chassis, sim.handle_world)
    local baseCenter = robot.geometry.base.center_m
    local function centerZ(component)
        local center = component.center_m
        local x, y, z = center[1] - baseCenter[1], center[2] - baseCenter[2], center[3] - baseCenter[3]
        return matrix[9] * x + matrix[10] * y + matrix[11] * z + matrix[12]
    end
    local function cylinderMinimum(component)
        local radial = component.radius_m * math.sqrt(matrix[9] * matrix[9] + matrix[10] * matrix[10])
        return centerZ(component) - radial - 0.5 * component.height_m * math.abs(matrix[11])
    end
    local function cuboidMinimum(component)
        local size = component.size_m
        local support = 0.5 * (
            math.abs(matrix[9]) * size[1] + math.abs(matrix[10]) * size[2] + math.abs(matrix[11]) * size[3]
        )
        return centerZ(component) - support
    end
    local components = {
        base = cylinderMinimum(robot.geometry.base),
        shell = cylinderMinimum(robot.geometry.shell),
        bumper = cuboidMinimum(robot.geometry.bumper),
    }
    return math.min(components.base, components.shell, components.bumper)
end

local function contactState(robot)
    local chassisPartners = contactPartners(robot.chassis)
    local casterPartners = contactPartners(robot.caster)
    local leftPartners = contactPartners(robot.leftWheel)
    local rightPartners = contactPartners(robot.rightWheel)
    -- Bullet can aggregate a passive caster contact onto the root body.  Use
    -- the profile geometry to distinguish support from chassis penetration.
    local rootFloor = chassisPartners[floor] == true
    local chassisMinimum = chassisPhysicalMinZ(robot)
    local chassisFloor = rootFloor and chassisMinimum <= floorTop + 0.0005
    local casterFloor = casterPartners[floor] == true or (rootFloor and not chassisFloor)
    local leftFloor = leftPartners[floor] == true
    local rightFloor = rightPartners[floor] == true
    local obstacle = false
    for handle, _ in pairs(obstacleSet) do
        if chassisPartners[handle] or casterPartners[handle] or leftPartners[handle] or rightPartners[handle] then
            obstacle = true
            break
        end
    end
    return {
        chassis_floor = chassisFloor,
        caster_floor = casterFloor,
        left_wheel_floor = leftFloor,
        right_wheel_floor = rightFloor,
        obstacle = obstacle,
        support_ok = casterFloor and leftFloor and rightFloor and not chassisFloor,
        chassis_physical_min_z_m = chassisMinimum,
        chassis_clearance_m = chassisMinimum - floorTop,
        floor_top_z_m = floorTop,
    }
end

local function jointState(robot, now, velocities)
    return {
        header = {stamp = now, frame_id = robot.role .. '/base_link'},
        name = {'left_wheel_joint', 'right_wheel_joint'},
        position = {},
        velocity = {velocities[1], velocities[2]},
        effort = {},
    }
end

local function diagnosticValues(robot)
    local contact = robot.contact
    local function kv(key, value) return {key = key, value = tostring(value)} end
    return {
        kv('profile_id', profile.profile_id),
        kv('profile_provenance', profile.provenance),
        kv('profile_calibration_state', profile.calibration_state),
        kv('profile_checksum', profile.checksum),
        kv('last_actuator_delay_s', robot.lastDelay),
        kv('watchdog_active', robot.watchdog),
        kv('queue_depth', #robot.queue),
        kv('dropped_commands', robot.droppedCount),
        kv('deadline_miss_count', robot.deadlineMissCount),
        kv('caster_floor_contact', contact.caster_floor),
        kv('left_wheel_floor_contact', contact.left_wheel_floor),
        kv('right_wheel_floor_contact', contact.right_wheel_floor),
        kv('chassis_floor_contact', contact.chassis_floor),
        kv('obstacle_contact', contact.obstacle),
        kv('support_ok', contact.support_ok),
        kv('controller_env_obstacle_mask', false),
        kv('action_shield', false),
    }
end

local function publishDiagnostics(now)
    local statuses = {}
    for _, role in ipairs({'defender', 'attacker'}) do
        local robot = robots[role]
        local level = (robot.contact.chassis_floor or not robot.contact.support_ok or robot.deadlineMissCount > 0) and 1 or 0
        statuses[#statuses + 1] = {
            level = level,
            name = 'trackmaker_bridge/' .. role,
            message = level == 0 and 'profiled_actuator_ok' or 'profiled_actuator_warning',
            hardware_id = 'coppeliasim_v2_1',
            values = diagnosticValues(robot),
        }
    end
    simROS2.publish(publishers.diagnostics, {header = {stamp = now, frame_id = 'map'}, status = statuses})
end

function sysCall_init()
    ready = false
end

function sysCall_beforeSimulation()
    if ready then return end
    local packedProfile = sim.getBufferProperty(sim.handle_scene, 'customData.trackmakerProfile', {noError = true})
    if packedProfile == nil then error('V2.1 scene profile is missing') end
    profile = sim.unpackTable(packedProfile)
    if profile.schema_version ~= '2.1' then error('unsupported profile schema_version: ' .. tostring(profile.schema_version)) end
    if profile.provenance ~= 'prior' and profile.provenance ~= 'measured' then error('invalid profile provenance') end
    if profile.provenance == 'prior' and profile.calibration_state ~= 'uncalibrated' then error('prior profile must be uncalibrated') end
    for _, role in ipairs({'defender', 'attacker'}) do
        if profile.robots[role].actuator.response_space ~= 'body_twist' then
            error(role .. ' actuator.response_space must be body_twist')
        end
    end
    simulationStart = sim.getSimulationTime()
    sim.setBoolParam(sim.boolparam_realtime_simulation, profile.simulation.realtime)

    floor = findByAlias('TrackMaker_turtlebot4_floor', sim.sceneobject_shape)
    floorTop = sim.getObjectPosition(floor, sim.handle_world)[3]
        + sim.getObjectFloatParam(floor, sim.objfloatparam_objbbox_max_z)
    target = findByAlias('TrackMaker_target')
    for _, handle in ipairs(sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)) do
        if string.find(alias(handle), 'TrackMaker_turtlebot4_obstacle_', 1, true) == 1 and sim.getObjectType(handle) == sim.sceneobject_shape then
            obstacles[#obstacles + 1] = handle
            obstacleSet[handle] = true
        end
    end
    obstacleCollection = sim.createCollection(1)
    for _, handle in ipairs(obstacles) do sim.addItemToCollection(obstacleCollection, sim.handle_single, handle, 0) end
    configureRobot('defender', 'trackmakerDefenderCmdVel', 101)
    configureRobot('attacker', 'trackmakerAttackerCmdVel', 211)
    applyNamedSpawn()

    addPublisher('/tracking/target/pose', 'geometry_msgs/msg/PoseStamped', 'targetPose')
    addPublisher('/clock', 'rosgraph_msgs/msg/Clock', 'clock')
    addPublisher('/diagnostics', 'diagnostic_msgs/msg/DiagnosticArray', 'diagnostics')
    addPublisher('/demo/profile_metadata', 'std_msgs/msg/String', 'profileMetadata', transientLocalQos())
    local cameraParam = getNamedString('trackmakerCamera')
    local cameraView = getNamedString('trackmakerCameraView') or 'overhead'
    cameraEnabled = cameraParam ~= '0'
    if cameraEnabled then
        addPublisher('/demo/camera/image_raw', 'sensor_msgs/msg/Image', 'camera')
        simROS2.publisherTreatUInt8ArrayAsString(publishers.camera)
        camera = createDemoCamera(cameraView)
    end

    local ok, metadataOrError = pcall(verifyRuntime)
    if not ok then
        for _, robot in pairs(robots) do
            pcall(sim.setJointTargetVelocity, robot.leftWheelJoint, 0.0)
            pcall(sim.setJointTargetVelocity, robot.rightWheelJoint, 0.0)
        end
        sim.stopSimulation()
        error('runtime introspection failed: ' .. tostring(metadataOrError))
    end
    runtimeMetadata = metadataOrError
    local metadata = {
        schema_version = profile.schema_version,
        profile_id = profile.profile_id,
        provenance = profile.provenance,
        calibration_state = profile.calibration_state,
        actuator_response_space = 'body_twist',
        seed = profile.seed,
        checksum = profile.checksum,
        runtime = runtimeMetadata,
    }
    simROS2.publish(publishers.profileMetadata, {data = json.encode(metadata)})
    ready = true
    log('ready profile=' .. profile.profile_id .. ' checksum=' .. profile.checksum .. '; runtime introspection passed')
end

function sysCall_dyn(inData)
    if not ready or inData.afterStep then return end
    requireClose('dynamic callback dt', inData.dt, profile.simulation.physics_step_s, 1e-9)
    local now = sim.getSimulationTime() + (inData.passCnt - 1) * inData.dt
    actuatorStep(robots.defender, now, inData.dt)
    actuatorStep(robots.attacker, now, inData.dt)
end

function sysCall_actuation()
    -- Wheel targets are updated at every 5 ms physics step in sysCall_dyn.
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
        robot.actualWheel = {sim.getJointVelocity(robot.leftWheelJoint), sim.getJointVelocity(robot.rightWheelJoint)}
        robot.contact = contactState(robot)
        simROS2.publish(publishers[role .. 'Pose'], poseMessage(robot.baseFrame, now, 0.0))
        simROS2.publish(publishers[role .. 'Collision'], {data = robot.contact.obstacle})
        simROS2.publish(publishers[role .. 'JointTargets'], jointState(robot, now, robot.targetWheel))
        simROS2.publish(publishers[role .. 'JointStates'], jointState(robot, now, robot.actualWheel))
        simROS2.publish(publishers[role .. 'ActuatorState'], {data = json.encode({
            time_s = simulationTime,
            role = role,
            requested_linear_mps = robot.watchdog and 0.0 or robot.commandLinear,
            requested_angular_radps = robot.watchdog and 0.0 or robot.commandAngular,
            filtered_linear_mps = robot.filteredLinear,
            filtered_angular_radps = robot.filteredAngular,
            target_wheel_radps = robot.targetWheel,
            actual_wheel_radps = robot.actualWheel,
            applied_wheel_torque_nm = {
                sim.getJointForce(robot.leftWheelJoint),
                sim.getJointForce(robot.rightWheelJoint),
            },
            watchdog_active = robot.watchdog,
            contact = robot.contact,
            queue_depth = #robot.queue,
            last_actuator_delay_s = robot.lastDelay,
            deadline_miss_count = robot.deadlineMissCount,
        })})
        transforms[#transforms + 1] = transformMessage(robot.baseFrame, role .. '/base_link', now, 0.0)
        transforms[#transforms + 1] = transformMessage(robot.lidarOrigin, role .. '/laser', now, 0.0)
        if sensingCount % 2 == 0 then simROS2.publish(publishers[role .. 'Scan'], laserMessage(robot, now)) end
    end
    simROS2.publish(publishers.targetPose, poseMessage(target, now, 0.0))
    transforms[#transforms + 1] = transformMessage(target, 'target/base_link', now, 0.0)
    simROS2.sendTransforms(transforms)
    publishDiagnostics(now)

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
    for _, robot in pairs(robots) do
        pcall(sim.setJointTargetVelocity, robot.leftWheelJoint, 0.0)
        pcall(sim.setJointTargetVelocity, robot.rightWheelJoint, 0.0)
    end
    for _, handle in pairs(publishers) do pcall(simROS2.shutdownPublisher, handle) end
    for _, handle in pairs(subscriptions) do pcall(simROS2.shutdownSubscription, handle) end
    if camera >= 0 then pcall(sim.removeObjects, {camera}) end
    ready = false
end
