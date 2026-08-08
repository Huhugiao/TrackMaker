# Codex 项目记忆

最后更新：2026-08-07

## 记录原则

- 使用中文，只记录当前有效结论、硬约束、重要资产和下一步。
- 不记录逐次实验过程、中间指标、失败日志或可由代码直接读出的配置。
- 命令、标识符、路径和指标名称保留英文。

## 提交信息

建议使用一个 emoji 加一行简短说明，例如：`✨ 添加登录页`。

## 项目定位

- 研究两台 TurtleBot4 与目标构成的目标攻击防御（TAD）博弈：Attacker A、Defender D、Target T。
- 论文聚焦攻防策略、自博弈和策略边界，不做多机器人围捕。
- 当前主线是生成、筛选下层策略并研究真实互补性；暂不训练新 HRL top、meta-controller、局内 Router 或 payoff solver。

## 硬约束

- 测试、训练统一使用 Conda 环境 `lnenv`；当前机器为 CPU-only，正式训练使用 Gym TAD 环境。
- 保持速度 `A=2.0`、`D=2.6`。Attacker 拥有 A/D/T 精确真值状态，不采用局部感知或 sim-to-real 假设。
- 正式 Attacker 训练使用学习型 Defender；正式 Defender pool 只允许学习策略。规则策略仅用于诊断。
- 新主线禁止 BC、DAgger、teacher/anchor、KL-to-BC 及 A0/A1/A1b warm-start；相关方法只能经用户明确同意后作为独立 ablation。
- checkpoint 按固定种子的真实 `target_success_rate` 选择，不按 shaped reward 或旧 `win` 选择。
- `defender_collision` 计为 draw，不得暗中并入任一方成功率。
- `eval/vs.py`（历史路径 `ppo/vs.py`）是 learned/rule policy 的标准评测流程；常规对比必须复用其
  `run_evaluation` 或 CLI，只通过显式参数调整策略、checkpoint、episode、seed 和环境配置。
- 专项矩阵确需独立入口时，必须保持同一契约：每局重置环境和策略状态、checkpoint greedy 推理、
  paired fixed seeds、按终局 `reason` 分离 capture/target success/timeout/collision，并保存逐局记录；任何偏离必须写明。
- 启动 Defender 长训练前，所有冻结 Attacker 必须完成真实加载 smoke test，opponent spec 必须写入 run config。
- 启动训练后立即报告预计耗时；获得实际步速或首轮评测后重新校准。

## 当前有效方案

- Attacker pool：`heuristic_default`、`heuristic_evasive`、`heuristic_geometry_feint_v3`、
  `heuristic_occlusion_dash_v2`、`rl_ppo_goal_rush`、`rl_ppo_evasive`。两个 evasive 来源不同，不得合并统计。
- Attacker 正式入口为 `train/train_attacker_multistyle_ppo.py`；模型为 feed-forward multi-style PPO，
  无 RNN、action shield 或跨 style rollout relabel。active reward style 仅为 `goal_rush` 与 `evasive`。
- Attacker checkpoint 与 alias 的唯一来源是 `attacker/frozen_pool.py`。
- Defender active pool：
  - Protect：`models/defender_protect_mlp_ctde_frozen6_20260721_105148/best_balanced_model.pth`
  - recurrent Chase：`models/defender_chase_nmn_dual_gru_raw_dense_05-05-19-12/final_model.pth`
  - A* path-risk HRL：`models/hrl_ch2_m1_astar_cached_top_20260606_170036/best_model.pth`
- HRL 仅作为已冻结的 Chapter 2 基线保留；其内部继续使用训练时的旧 baseline 与旧 Chase 子技能。
- 已淘汰且不得回到默认主线：A0/A1/A1b 延续、recurrent PPO + shield、multi-style SAC、新 Chase、
  pursuer/interceptor/sentinel，以及未通过冻结门槛的 Router 系列。

## 当前研究重点

- 主要风险是“伪多样性”：名称、reward style 或轨迹不同，但真实成败高度重合。
- 后续应从相同 simulator snapshot 比较候选策略，定位共同失败状态和真正的 state-level 互补性，再决定是否生成 specialist。

## 权威文档

- 当前执行契约：`docs/lower_policy_generation_plan.md`
- 完整里程碑与模型保留边界：`docs/project_history.md`
- CoppeliaSim 仅作可选 TurtleBot4 验证模块，不是当前正式训练依赖。
