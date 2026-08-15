# CoppeliaSim ROS 2 数字孪生 V2.1

最后更新：2026-08-15

## 范围与边界

V2.1 在 CoppeliaSim 4.10 / Bullet 中运行冻结的 A* HRL Defender（Protect + recurrent Chase）和
`rl_ppo_goal_rush` Attacker。它是可重复、可标定的仿真部署验证链，不参与正式训练，也不是实机验证。
本轮没有启动训练、接入硬件、OAK-D、SLAM、Nav2 或新增对手识别。

研究契约不变：Gym 正式训练仍使用 `lnenv`，速度保持 `A=2.0`、`D=2.6`；ROS 动作换算上限为
Defender `0.234 m/s, 0.942 rad/s`、Attacker `0.180 m/s, 1.885 rad/s`，Attacker policy command
slew 仍为 `0.486 m/s²`。V2 profile 中的 delay、lag、gain、deadband、wheel-plant acceleration 与
braking 是其后的可标定执行器模型，不回写训练配置。

## 权威资产与 provenance

- prior profile：`coppeliasim/profiles/trackmaker_turtlebot4_v2_1_prior.json`
- JSON schema：`coppeliasim/profiles/trackmaker_digital_twin_profile.schema.json`
- dense 场景：`coppeliasim/scenes/trackmaker_turtlebot4_v2_1_scene.ttt` 及同名 `.json`
- flat calibration 场景：`coppeliasim/scenes/trackmaker_turtlebot4_v2_1_calibration_scene.ttt` 及同名 `.json`
- 可重复 builder：`coppeliasim/tools/coppeliasim_build_digital_twin_v2_scene.py`

当前默认 profile 为 `trackmaker_turtlebot4_v2_1_body_twist_prior_20260815`，必须显式显示为
`provenance=prior`、`calibration_state=uncalibrated` 和 `actuator.response_space=body_twist`。运行时会把完整
profile 嵌入 scene，并逐项 introspect engine、dt、solver、gravity、质量、惯量、轮参数、caster mount、材料和
actuator；任何不一致都 fail closed。

标定器生成新的 `provenance=measured`、`calibration_state=calibrated` profile，不覆盖 prior，也不会自动
改变默认场景或默认 launcher。切换 profile 必须显式传入 `--profile` 并用该 profile 重新生成场景。

## 场景生成

生成 dense policy 场景：

```bash
conda run -n lnenv python coppeliasim/tools/coppeliasim_build_digital_twin_v2_scene.py \
  --density dense \
  --output coppeliasim/scenes/trackmaker_turtlebot4_v2_1_scene.ttt \
  --profile coppeliasim/profiles/trackmaker_turtlebot4_v2_1_prior.json \
  --skip-urdf-regenerate \
  --replace
```

生成只保留边界墙的 flat calibration 场景：

```bash
conda run -n lnenv python coppeliasim/tools/coppeliasim_build_digital_twin_v2_scene.py \
  --density none \
  --output coppeliasim/scenes/trackmaker_turtlebot4_v2_1_calibration_scene.ttt \
  --profile coppeliasim/profiles/trackmaker_turtlebot4_v2_1_prior.json \
  --skip-urdf-regenerate \
  --replace
```

builder 从 TurtleBot4 URDF 聚合 chassis mass/CoM/inertia，保留两个独立 drive wheels，并用独立动态
25 mm equivalent caster sphere + passive spherical joint 建模前支撑。可视 mesh 全部 non-respondable；
旧 URDF 动力学层和旧 obstacle aliases 会被原子替换，避免重复碰撞体。写入临时 `.ttt` 后先做 scene
validation，通过才 `os.replace` 到正式路径。以上命令使用仓库内已冻结且受 profile checksum 约束的 prepared
URDF；安装了 `turtlebot4_description` 时可去掉 `--skip-urdf-regenerate`，先由官方 xacro 重新生成。

当前 prior 的主要物理配置为：每车总质量 `5.39 kg`，chassis `4.98 kg`，每轮 `0.20 kg`，caster
`0.01 kg`；轮半径 `0.03575 m`、轮距 `0.233 m`；Bullet outer step `50 ms`、physics step `5 ms`、
solver `100` iterations、gravity `-9.81 m/s²`。floor/wheel friction 分别为 `0.85/1.15`，wheel force
上限为 `0.05 N·m`。这些值属于工程 prior，不能解释为实机测量。

## 单入口运行

默认入口同时启动 CoppeliaSim、`simROS2`、双策略推理、rosbag 和 MP4：

```bash
./coppeliasim/tools/run_ros2_demo.sh \
  --seed 2026081511 \
  --max-steps 60 \
  --output-dir outputs/coppeliasim/v2_1_demo_seed_2026081511
```

正式长 horizon 验证应显式使用 `--max-steps 449`。`--camera-view overhead|oblique` 选择录像视角；
`--no-media`、`--no-bag`、`--no-video` 控制产物。默认路径可由 `TRACKMAKER_ROS_ENV`、
`COPPELIASIM_ROOT`、`TRACKMAKER_SCENE` 和 `TRACKMAKER_PROFILE` 覆盖。manifest 始终取 scene 的同名
`.json`；manifest 与 profile checksum 必须一致，否则在启动仿真前拒绝运行。

每次 policy 运行保存：

- `episode.json`：终局 reason、逐步位姿、动作、HRL skill、五输入 timing、checkpoint 与 profile 元数据；
- `episode_bag/`：trajectory、TF、scan、cmd_vel、joint target/actual、actuator state/events、diagnostics；
- `episode.mp4`：1920×1080、H.264、20 fps，叠加 profile provenance、request/filter/actual、接触和 watchdog；
- `spawn.json`、`profile_input.json`、`profile_checksum.txt` 与进程日志。

十个固定 seeds 的 raw 矩阵入口为：

```bash
./coppeliasim/tools/run_ros2_seed_matrix.py \
  --seed-start 2026081510 --count 10 --max-steps 60 \
  --output-dir outputs/coppeliasim/v2_1_policy_matrix
```

每局独立重置场景和 recurrent state，checkpoint 使用 greedy 推理；capture、Target breach、timeout、
`defender_collision` draw 和 Attacker collision 分离记录。矩阵显式记录
`controller_env_obstacle_mask=false`、`action_shield=false`、`create3_reflexes_expected=false`。短 horizon
矩阵只用于 runtime/policy smoke，不能作为策略胜率或 legacy efficacy 对比。

## ROS 2 接口与 timing

| Topic | Type | 频率 / 说明 |
|---|---|---|
| `/tracking/{defender,attacker,target}/pose` | `geometry_msgs/PoseStamped` | 20 Hz，`map` |
| `/{defender,attacker}/scan` | `sensor_msgs/LaserScan` | 10 Hz，64 rays |
| `/{defender,attacker}/cmd_vel` | `geometry_msgs/Twist` | policy 约 9 Hz，20 Hz 重发 |
| `/{defender,attacker}/joint_targets` | `sensor_msgs/JointState` | 20 Hz，左右轮目标速度 |
| `/{defender,attacker}/joint_states` | `sensor_msgs/JointState` | 20 Hz，左右轮实际速度 |
| `/{defender,attacker}/actuator_state` | `std_msgs/String` | 20 Hz，request/filter/target/actual/contact/watchdog |
| `/{defender,attacker}/actuator_events` | `std_msgs/String` | receive/execute/drop 事件 |
| `/demo/profile_metadata` | `std_msgs/String` | reliable + transient-local，profile 与 runtime introspection |
| `/clock` | `rosgraph_msgs/Clock` | 20 Hz，仿真时间 |
| `/tf` | `tf2_msgs/TFMessage` | `map` → namespaced `base_link` / `laser` |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | bridge + policy timing/profile/actuator 状态 |
| `/demo/selected_skill` | `std_msgs/String` | `protect` / `chase` |
| `/demo/outcome` | `std_msgs/String` | reliable + transient-local 终局 JSON |

policy 在 CoppeliaSim 中只使用 `/clock`，实机模式才传 `--wall-time` 使用 steady clock。A/D/T pose、
Defender scan、Attacker scan 五个输入必须全部存在、fresh，且最大 timestamp skew 不超过 `50 ms`；缺失、
stale、skew、非有限值或非法 quaternion 都 fail closed 并持续发布零速。软件渲染变慢不会改变 policy step
尺度或 GRU 更新次数。

Defender 的 71 维 actor、Attacker 的 72 维 privileged、143 维 critic 与 Gym 投影一致；Attacker 模型
内部仍投影为 70 维 actor。Gym 顺时针正转向在 ROS `angular.z` 上显式反号。HRL top 与 recurrent Chase
状态跨 step 保留、每局 reset；Protect 使用冻结 `mlp_ctde`。

## 执行器与安全边界

profile 必须声明 `actuator.response_space=body_twist`。每个 `cmd_vel` 在固定 `5 ms` actuator tick 中按固定顺序处理：finite check → speed clamp → seeded
fixed delay/jitter/loss/outage queue → 机体 `Vx/ωz` first-order lag → `Vx` acceleration/braking slew →
差速轮转换 → left/right gain + deadband → wheel target → Bullet force-limited joint。这样高角速度不会因左右轮
独立限速而抵消机体平移分量。事件和 target/actual wheel speed 全量可观测；同 profile + seed
应产生相同 queue 和 target 序列。

0.5 s 没有新命令时 bridge watchdog 停车；终局和退出连续发布零命令。仿真中不使用 teleport、wheel-drop
hack、controller/env obstacle mask、action shield 或 Create 3 reflex。实机 Create 3 reflex 不得关闭，但其
结果必须与本仿真 raw 结果分开报告。

contact telemetry 区分左右 drive wheel、caster、chassis-floor、obstacle，并报告解析 chassis clearance。
`support_ok` 只表示两轮与 caster 均支撑；masked 结果或 mask 干预不能被解释为底层策略安全性。

## 标定流程

同一入口的 `--calibrate` 默认自动切换到 flat calibration scene：

```bash
./coppeliasim/tools/run_ros2_demo.sh --calibrate \
  --output-dir outputs/coppeliasim/v2_1_motion_calibration
```

计划覆盖多档正反直行、左右高低角速度、双向 arc、step、ramp、emergency stop、watchdog outage 和
graceful shutdown；不通过 pose reset 或 teleport 拼接轨迹。数据以 `/clock` 为时间基准，保存 portable JSON、
joint target/actual、actuator events、contact、diagnostics 和 rosbag。

launcher 随后自动运行 SciPy fitter，按固定 seed 做 deterministic `60/20/20` train/validation/test split，
以实际 `joint_actual_radps` 和轨迹拟合左右 gain/deadband、time constant、acceleration/braking、delay、
wheel radius 和 wheel separation；不得用 bridge 内部的 filtered state 代替 plant observation。
数据必须满足 support ≥ 0.95、无 chassis-floor/obstacle collision、观察到 watchdog、全部 phase 有样本，且
graceful shutdown 的实际轮速归零；否则拒绝拟合。synthetic truth-recovery 是拟合器门槛，不是实机证据。

旧 wheel-space actuator 生成的 calibration fit 或 measured scene 不得复用。切换 actuator response space
后必须重新采集数据、拟合 profile，并按新 checksum 重建 scene。

## 验证与当前证据

```bash
# 单元、schema、确定性、graceful-shutdown 回归
conda run -n lnenv python -m unittest discover -s coppeliasim/tests -p 'test_*.py' -v

# Gym–ROS 观测、动作和 Attacker 0.486 m/s² slew parity
conda run -n lnenv python coppeliasim/tools/trackmaker_gym_ros_parity.py \
  --manifest coppeliasim/scenes/trackmaker_turtlebot4_v2_1_scene.json

# 四 checkpoint checksum、CPU load、greedy/reset 确定性
conda run -n ros2humble python coppeliasim/tools/trackmaker_checkpoint_smoke.py

# 自动启动最终 scene 并检查 live topic/rate/runtime profile
./coppeliasim/tools/run_ros2_demo.sh --interface-smoke \
  --output-dir outputs/coppeliasim/v2_1_interface_smoke

# bag 轨迹与基础接口
conda run -n ros2humble python coppeliasim/tools/trackmaker_ros2_bag_check.py \
  --bag OUTPUT/episode_bag --episode OUTPUT/episode.json
```

interface smoke 的 ROS 频率按 `/clock` 跨度计算，并单独保存 wall-rate 作为 CPU 性能诊断，避免把非实时
仿真误判为 topic 合约降频。`outputs/` 只保存当前代码重新生成的本地证据，不作为长期源码资产。

## 未来实机替换边界

未来实机阶段删除 CoppeliaSim 和 Lua bridge，由 tracking 节点发布相同的 `PoseStamped(map)` 和 TF；两台
TurtleBot4 提供 namespaced `LaserScan` 并消费相同 `Twist`。推理、checkpoint 校验、观测投影、GRU 生命周期、
动作限幅、watchdog、终局记录和媒体工具保持不变。实机 profile 必须由独立测量数据生成并显式标记
`provenance=measured`，不得把当前本机仿真拟合结果当作实机参数。
