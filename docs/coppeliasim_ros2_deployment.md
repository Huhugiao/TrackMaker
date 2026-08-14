# CoppeliaSim ROS 2 部署与 Sim2Real 基线

## 交付范围

本模块在 CoppeliaSim 4.10 中运行冻结的 A* HRL Defender（legacy Protect + recurrent Chase）和
`rl_ppo_goal_rush` Attacker。ROS 2 推理节点只依赖标准消息和下面的接口；未来实机接入时保持推理节点不变，
将 CoppeliaSim 真值发布端替换为 `map` frame 的 tracking 节点即可。

正式训练仍在 `lnenv` 和 Gym TAD 环境进行；CoppeliaSim/ROS 2 只用于部署验证。

## 环境

已验证平台为 Ubuntu 22.04、ROS 2 Humble、CoppeliaSim 4.10 和 CPU PyTorch。部署环境与训练环境隔离：

```bash
conda create -y -n ros2humble --override-channels \
  -c robostack-staging -c conda-forge \
  python=3.10 ros-humble-desktop ffmpeg pip
conda run -n ros2humble python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
conda install -y -n ros2humble --override-channels \
  -c robostack-staging -c conda-forge \
  pyzmq cbor2 colcon-common-extensions c-compiler cxx-compiler make ninja
```

CoppeliaSim 自带的 `simROS2` 未生成 `PoseStamped` 和 `/clock` 类型时，执行一次：

```bash
./coppeliasim/tools/build_trackmaker_simros2.sh
```

脚本复制官方插件源码到 `outputs/coppeliasim/simros2_ws`，只扩展
`geometry_msgs/msg/PoseStamped` 和 `rosgraph_msgs/msg/Clock`，然后安装到 CoppeliaSim 目录。

## 单入口运行

默认入口同时启动 CoppeliaSim、`simROS2` 桥、双策略推理、rosbag 和录像：

```bash
./coppeliasim/tools/run_ros2_demo.sh \
  --seed 20260326 \
  --max-steps 449 \
  --output-dir outputs/coppeliasim/ros2_demo/seed_20260326
```

默认路径可用环境变量覆盖：

```bash
export TRACKMAKER_ROS_ENV=/path/to/ros2humble
export COPPELIASIM_ROOT=/path/to/CoppeliaSim_Edu_V4_10_0_rev0_Ubuntu22_04
```

录像默认使用严格俯视视角；需要带透视和遮挡关系的 3D 斜俯视角时增加 `--camera-view oblique`。

每次运行都会生成：

- `episode.json`：终局、逐步位姿、动作、速度、HRL skill、碰撞和 checkpoint 元数据；
- `episode_bag/`：可回放轨迹、TF、scan、cmd_vel、diagnostics 和终局；
- `episode.mp4`：1920×1080、H.264、`yuv420p`，带角色、skill、时间与终局叠加；
- `spawn.json` 与各进程日志。

十个固定 paired seeds 的无媒体验收入口为：

```bash
./coppeliasim/tools/run_ros2_seed_matrix.py \
  --seed-start 20260326 --count 10 --max-steps 449
```

默认矩阵不重复录制媒体，以减少软件渲染开销；加 `--with-media` 可为每个 seed 都生成 MP4 和 rosbag。

轮式底盘、零速和 watchdog 的开环标定使用同一入口：

```bash
./coppeliasim/tools/run_ros2_demo.sh --calibrate \
  --output-dir outputs/coppeliasim/ros2_acceptance/motion_calibration
```

## ROS 2 接口契约

| Topic | Type | 生产/消费 | Frame / 说明 |
|---|---|---|---|
| `/tracking/defender/pose` | `geometry_msgs/PoseStamped` | tracking → policy | `map` |
| `/tracking/attacker/pose` | `geometry_msgs/PoseStamped` | tracking → policy | `map` |
| `/tracking/target/pose` | `geometry_msgs/PoseStamped` | tracking → policy | `map` |
| `/defender/scan` | `sensor_msgs/LaserScan` | robot → policy | `defender/laser`，64 rays |
| `/attacker/scan` | `sensor_msgs/LaserScan` | robot → policy | `attacker/laser`，64 rays |
| `/defender/cmd_vel` | `geometry_msgs/Twist` | policy → robot | 20 Hz 重发 |
| `/attacker/cmd_vel` | `geometry_msgs/Twist` | policy → robot | 20 Hz 重发 |
| `/clock` | `rosgraph_msgs/Clock` | simulator → ROS | 仿真时间 |
| `/tf` | `tf2_msgs/TFMessage` | tracking → ROS | `map` → namespaced `base_link` / `laser` |
| `/diagnostics` | `diagnostic_msgs/DiagnosticArray` | policy → operator | freshness、skill、安全配置 |
| `/demo/selected_skill` | `std_msgs/String` | policy → recorder | `protect` / `chase` |
| `/demo/outcome` | `std_msgs/String` | policy → recorder | transient-local JSON |

`LaserScan` 采用 sensor-data QoS；终局采用 reliable + transient-local；其余控制和状态使用 reliable volatile。
所有 pose、scan、TF 和 `/clock` 使用同一仿真时间戳。两个机器人使用 `/defender`、`/attacker` namespace，
实机放在同一 ROS domain。

## 观测与动作契约

- Defender 从 A/D/T 真值位姿和 Defender LiDAR 构造 71 维 actor observation；静态地图遮挡、最后可见
  Attacker 状态和 `steps_since_observed` 与 Gym 保持一致。72 维 privileged observation 与 actor 拼为
  143 维 critic observation。
- HRL top 和 recurrent Chase 的 GRU 状态跨 step 保留，每局开始显式 reset；Protect 为冻结的 legacy
  `mlp_ctde` checkpoint，以保持 top checkpoint 的技能语义。
- Attacker 接收同一个 72 维真值状态，模型内部按训练代码投影为 70 维 attacker-centric actor
  observation，并确定性执行 `goal_rush`。
- 策略决策约 9 Hz，最后命令以 20 Hz 重发；CoppeliaSim 默认以 `/clock` 驱动两个定时器，软件渲染变慢时
  不会改变每个策略 step 的运动距离或 GRU 更新次数。训练的 20 Hz 运动尺度换算到 9 Hz 决策后，标称上限为
  Defender `0.234 m/s, 0.942 rad/s`，Attacker `0.180 m/s, 1.885 rad/s`，保留训练中的
  `A=2.0, D=2.6` 相对动力学。
- Gym 图像坐标的正转向为顺时针，发布 ROS `angular.z` 前会显式反号；Attacker 的 `0.6 px/step`
  加速度对应 `0.486 m/s²`。
- CoppeliaSim 只设置左右轮关节目标速度；运行时不修改机器人 pose，不启用 map lookahead mask 或 action shield。
- 虚拟 LiDAR 从各自真实 `rplidar_link` 发射，只检测地图障碍集合，不检测自身或另一台机器人；障碍高度为
  `0.5 m`，线段障碍采用与 Gym 相同的胶囊体（矩形主体 + 两个圆端）。

## 安全和终局

- 启动默认零速；命令必须为有限值并经过速度限幅；0.5 s 没有新命令时桥和推理节点均停车。
- 终局或退出时连续发布十次零速。实机侧不得关闭 Create 3 原生 reflex。
- 捕获使用 `0.45 m` 机身中心距和 30° 前向扇区，不依赖物理接触；Target breach 为 `0.35 m`。
- `defender_collision` 独立记录为终局 draw；Attacker 碰撞按训练契约作为逐步事件记录，不改写终局。
  masked/raw 不混报。本部署固定
  `controller_env_obstacle_mask=false`、`action_shield=false`。

## 验证

```bash
# Coppelia 核心单元测试（训练环境）
conda run -n lnenv python -m unittest discover -s coppeliasim/tests -v

# 128 个固定状态下逐值比较 71/72/143 维观测、动作和 Attacker 加速度
conda run -n lnenv python coppeliasim/tools/trackmaker_gym_ros_parity.py

# 四个真实 checkpoint、SHA-256、网络类型及 recurrent reset
conda run -n ros2humble python coppeliasim/tools/trackmaker_checkpoint_smoke.py

# 对运行中的闭环检查 Topic 类型、frame_id、时间戳和频率
conda run -n ros2humble python coppeliasim/tools/trackmaker_ros2_interface_smoke.py --duration-s 3

# 对运行中的 CoppeliaSim 逐束比较物理 LaserScan 与解析地图
conda run -n ros2humble python coppeliasim/tools/trackmaker_ros2_lidar_parity.py

# 校验 rosbag 接口类型及 episode.json 轨迹是否逐点包含在 bag 中
conda run -n ros2humble python coppeliasim/tools/trackmaker_ros2_bag_check.py \
  --bag outputs/coppeliasim/ros2_demo/seed_20260326/episode_bag \
  --episode outputs/coppeliasim/ros2_demo/seed_20260326/episode.json
```

动力学标定分别发送直行、正角速度、负角速度、零速和故意断流，确认 ROS yaw 与实际运动方向一致；停止发布
`cmd_vel` 后应在 0.5 s watchdog 窗口内归零。

最终场景对 seeds `20260336..20260345` 的 10 局 raw paired 结果为：5 局 capture、1 局 Target breach、
1 局 timeout、3 局 `defender_collision` draw；Attacker 障碍接触出现在 5/10 局。它证明两种任务终局均可在
真实轮关节运动下发生，但也表明 raw Defender/Attacker 仍不具备可靠碰撞安全性。由于
`controller_env_obstacle_mask=false`、`action_shield=false`，不得把成功视频解释为 100% 稳定部署。

当前无碰撞 capture 验收产物为
`outputs/coppeliasim/ros2_acceptance/acceptance_capture_seed_20260341_final/`：双方碰撞计数均为 0，终局中心距
`0.434 m`，大于两台物理机身接触距离；MP4 为 1920×1080、20 fps、10.15 s，不包含 checkpoint 加载前的
静止片段。bag 含双方各 216 个 20 Hz pose，episode 的 83 个决策位姿均可在 bag 中零误差找到；最大单帧位移
为 Defender `0.0121 m`、Attacker `0.0098 m`，全程最小中心距为 `0.405 m`（URDF 底盘接触距离
`0.328 m`），未发现 teleport、穿模或高度跳变。

最终运动标定的仿真/现实时间比为 `0.998`；两车直行响应为 `95.5%–95.6%`，左右转响应为
`96.5%–97.0%`，横向漂移不超过 `1.1 mm`。因此首版不再额外整体降速：当前速度已经是保留训练
`A=2.0, D=2.6` 相对动力学后的时间缩放值。实际场地若触发 Create 3 reflex，再按实机标定结果统一降低
两车线速度和角速度上限，不能只降低其中一方。

## 实机替换边界

实机阶段删除 CoppeliaSim 进程和 Lua bridge，由顶置相机、动捕或其他 tracking 节点发布相同的
`PoseStamped(map)` 和 TF；两台 TurtleBot4 继续提供 namespaced `LaserScan` 并消费相同 `Twist`。
推理、checkpoint 校验、观测投影、GRU 生命周期、动作限幅、watchdog、终局记录和 rosbag/视频工具均保持不变。
第一版不引入 OAK-D 检测、SLAM、Nav2 或新训练。

实机运行推理节点时传入 `--wall-time`，让相同定时器使用系统 ROS 时钟；CoppeliaSim 默认不传该参数。
