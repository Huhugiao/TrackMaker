# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码仓库中工作时提供指导。

## 项目概述

这是一个**追踪-防御 (TAD, Tracking-and-Attacking Defense)** 强化学习项目，其中**防御者 (Defender)** 智能体试图在**攻击者 (Attacker)** 到达静态**目标 (Target)** 之前捕获它。环境特点：

- 640x640 像素 2D 地图，含障碍物（墙壁、矩形、圆形、线段）
- 防御者视野受限（FOV：90° 锥形，范围 700）
- 64 线雷达用于障碍物检测
- 基于距离、角度和视线的捕获机制
- 基于 Pygame 的渲染与可视化

## 核心架构

### 环境 (`env.py`, `env_lib.py`, `map_config.py`)

- **`TADEnv` (又名 `TrackingEnv`)**：主 Gymnasium 环境，包含三个智能体
  - Defender：RL 控制的智能体，视野受限
  - Attacker：具备全局观测的对手智能体
  - Target：静态目标

- **奖励模式**：环境支持三种奖励模式：
  - `standard`：原始奖励，带时间惩罚 (-0.01/步)
  - `protect`：用于训练 **Protect 技能** - 无时间惩罚，有防御者-目标距离引导奖励（权重=6），超时计为防御者胜利 (+20)
  - `chase`：用于训练 **Chase 技能** - 有时间惩罚 (-0.02/步，总计约-10)，超时不计为胜利

- **关键观测**（Defender 72 维，Attacker 71 维）：
  - Defender：`[attacker_info(6), radar(64), target_info(2)]`，其中 `attacker_info = [distance, bearing, fov_edge, in_fov, occluded, unobserved_time]`
  - Attacker：`[position(3), defender_position(2), radar(64), target_position(2)]`

- **奖励结构**：取决于模式，但一般为：
  - Defender 捕获 Attacker：+20
  - Attacker 到达 Target 或碰撞：-20
  - 额外的模式特定奖励/惩罚（见上文）

- **碰撞检测**：Numba 加速的射线投射：
  - `ray_distance_grid()`：单次射线投射
  - `ray_distances_multi()`：批量射线投射
  - `is_point_blocked()`：点在障碍物内检查

### 策略模块

**基于规则的策略** (`rule_policies/`)：
- **`APFPolicy`**：人工势场导航基类
- **`AttackerAPFPolicy`**：Attacker 的 APF（吸引到 Target，避开障碍物）
- **`AttackerGlobalPolicy`**：使用 A* 算法的全局路径规划，将防御者视为动态障碍物
- **`DefenderAPFPolicy`**：Defender 的 APF（追踪移动的 Attacker，避开障碍物）

**基于 PPO 的策略** (`ppo/`)：
- **`Model`**：PPO actor-critic，带 GRU 序列预测器用于被遮挡的目标
- **`gru_predictor.py`**：当攻击者不在 FOV 内时预测其位置
- **`Training`**：使用 Ray 的分布式 RL，序列采用 TBPTT

### 基于技能的训练架构

项目为 Defender 采用**基于技能的训练方法**，配置在 `ppo/alg_parameters.py` 中：

1. **Protect 技能**：
   - **目标**：移动到并守卫 Target
   - **奖励模式**：`protect`
   - **GRU**：不使用（防御者知道目标位置）
   - **奖励**：防御者-目标距离引导（权重=6），胜利+20，失败-20，无时间惩罚
   - **对手**：使用 `AttackerGlobalPolicy`（A* 路径规划，避开防御者）
   - **模型路径**：`./models/defender_protect/`

2. **Chase 技能**：
   - **目标**：追踪并捕获 Attacker
   - **奖励模式**：`chase`
   - **GRU**：使用（当被遮挡时预测攻击者位置）
   - **奖励**：时间惩罚 (-0.02/步)，捕获+20，失败-20，超时=失败
   - **对手**：使用 `AttackerGlobalPolicy`（A* 路径规划，避开防御者）
   - **模型路径**：`./models/defender_chase/`

**通用组件**：
- `ppo/runner.py`：修改后的 `Runner` 类，接受 `skill_mode` 参数并创建带有相应 `reward_mode` 的环境
- 两个技能使用相同的训练基础设施 (`ppo/driver.py`)
- 两者使用相同的网络架构但不同的奖励信号
- 训练期间 Attacker 始终使用 `AttackerGlobalPolicy`

### 评估系统 (`ppo/vs.py`, `ppo/vs_episode.py`)

评估 Defender 对抗 Attacker 策略：
- Defender：`rl`（检查点）或 `defender_apf`
- Attacker：`attacker_apf`、`attacker_global`（A* 路径规划）

将统计 JSON 输出到 `output/` 目录。

## 常用开发命令

### 运行评估 (D vs A)
```bash
# APF Defender vs APF Attacker (100 回合)
python ppo/vs.py --defender defender_apf --attacker attacker_apf --episodes 100

# RL Defender vs Global Attacker 并生成 GIF
python ppo/vs.py --defender rl --defender-checkpoint ./models/.../checkpoint.pth \
    --attacker attacker_global --gif ./output/eval.gif --episodes 100

# 可视化单回合
python ppo/vs_episode.py --defender rl --attacker attacker_global
```

### 训练技能
```bash
# 步骤 1：编辑 ppo/alg_parameters.py
# 将 SetupParameters.SKILL_MODE 改为 "chase" 或 "protect"

# 步骤 2：训练
python ppo/driver.py
```

**在 `ppo/alg_parameters.py` 中：**
```python
class SetupParameters:
    # ...
    SKILL_MODE = "chase"  # 改为 "protect" 以训练 protect 技能
```

**模型根据技能保存到不同目录：**
- Chase 技能：`./models/defender_chase/`
- Protect 技能：`./models/defender_protect/`

## 重要配置

**环境参数** (`map_config.py`)：
- `EPISODE_LEN = 449` 最大步数
- `FOV_ANGLE = 90°`，`FOV_RANGE = 700`
- `RADAR_RAYS = 64`
- `defender_speed = 2.6`，`attacker_speed = 2.0`
- `capture_radius = 20`，`capture_sector_angle_deg = 30`
- `map_diagonal`：用于距离归一化的对角线长度

**训练参数** (`ppo/alg_parameters.py`)：
- `N_ENVS = 4` 并行环境
- `N_STEPS = 2048` rollout 长度
- `TBPTT_STEPS = 32` 用于 GRU 序列
- `SKILL_MODE = "chase"` 或 `"protect"`（控制训练哪个技能）
- `OPPONENT_TYPE = "random"`（使用 `RANDOM_OPPONENT_WEIGHTS`）

**障碍物密度** (`map_config.py`)：
- `ObstacleDensity.NONE` 或 `ObstacleDensity.DENSE`
- 调用 `set_obstacle_density(level)` 然后 `env_lib.build_occupancy()`

**Attacker Global 策略** (`rule_policies/attacker_global.py`)：
- 在 8 像素网格上使用 A* 路径规划
- 将防御者视为动态障碍物，可配置避让半径（默认 40px）
- 每 20 步重新规划路径
- 最高速度，不减速

## 关键实现细节

1. **归一化坐标**：所有观测归一化到 [-1, 1] 或 [0, 1]
   - 位置：`(coord / map_dim) * 2 - 1`
   - 角度：`degrees / 180.0`（或 `/180.0 - 1` 用于航向）
   - 距离：相对地图对角线裁剪

2. **GRU 预测**（Chase 技能使用，Protect 不使用）：
   - `get_normalized_attacker_info()` → `[rel_x_norm, rel_y_norm, is_visible]`，范围 [0,1]
   - `update_gru_sequence()` 更新预测器状态
   - `_get_defender_observation()` 在提供 `gru_prediction` 时使用预测

3. **捕获条件** (`env.py`)：
   - 距离 < `capture_radius`
   - 在航向的 `capture_sector_angle_deg/2` 范围内
   - 在 FOV 内（Defender 捕获 Attacker 时）
   - 视线未被阻挡

4. **渲染质量**：`map_config.set_render_quality('fast' | 'high')`
   - Fast：ssaa=1，无 AA，较短轨迹
   - High：ssaa=2，启用 AA，较长轨迹

5. **奖励函数** (`env_lib.py`)：
   - `reward_calculate_tad()`：标准奖励
   - `reward_calculate_protect()`：Protect 技能奖励
   - `reward_calculate_chase()`：Chase 技能奖励

## 文件位置

- 环境：`env.py`、`env_lib.py`、`map_config.py`
- 规则策略：`rule_policies/apf.py`、`rule_policies/attacker_apf.py`、`rule_policies/attacker_global.py`、`rule_policies/defender_apf.py`
- PPO 模型：`ppo/model.py`、`ppo/gru_predictor.py`、`ppo/nets.py`
- 训练：`ppo/driver.py`、`ppo/runner.py`、`ppo/alg_parameters.py`
- 评估：`ppo/vs.py`、`ppo/vs_episode.py`
- 输出：`output/*.json`、`./models/defender_protect/`、`./models/defender_chase/`

## 训练策略总结

**双技能方法**：
- **Protect**：学习将防御者定位在攻击者和目标之间，使用到目标的距离作为引导
- **Chase**：学习高效追踪和拦截攻击者，被遮挡时使用 GRU 预测

两个技能独立训练，对抗 `AttackerGlobalPolicy`，它提供一个强大的、战略性对手，能够在导航到目标的同时主动避开防御者。
