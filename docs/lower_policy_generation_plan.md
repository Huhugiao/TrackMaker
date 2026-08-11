# 下层攻防策略生成计划

最后更新：2026-08-11

## 当前研究范围

当前阶段生成、冻结、评测和筛选 Attacker/Defender 下层策略。Defender 只保留 Protect 与 Chase
两个底层技能；已冻结的 Chapter 2 A* path-risk HRL 仅作为现有组合策略参与评测，不继续训练其顶层。

统一环境契约为：单 Attacker、单 Defender、单非学习 Target；速度固定为 `A=2.0`、`D=2.6`；
Attacker 成功只认 `reason == attacker_caught_target`；`defender_collision` 为 draw；timeout 表示
Defender 阻止 Target breach，但必须与 capture 分开报告。

## Attacker 双通道生成

Attacker 同时使用两条生成路线：

1. heuristic learning：提出机制假设，实现确定性或固定种子的程序化策略，在真实环境中按 paired development/holdout seeds 评测；
2. pure RL：从随机权重训练连续控制策略，使用不同 reward style、课程或训练对手形成行为差异。

当前冻结池：

- `heuristic_default`
- `heuristic_evasive`
- `heuristic_geometry_feint_v3`
- `heuristic_occlusion_dash_v2`
- `rl_ppo_goal_rush`
- `rl_ppo_evasive`

程序化与 RL 的两个 evasive 必须使用显式名称，不能合并统计。RL checkpoint 为：

- `models/attacker_nmn_mlp_diversity_continuation_20260715_120331/best_goal_rush.pth`
- `models/attacker_nmn_mlp_diversity_continuation_20260715_120331/best_evasive.pth`

## Defender 两技能主线

正式底层技能只包括：

- Protect：`models/defender_protect_mlp_ctde_frozen6_20260721_105148/best_balanced_model.pth`
- recurrent Chase：`models/defender_chase_nmn_dual_gru_raw_dense_05-05-19-12/final_model.pth`

冻结 HRL 使用 Protect + Chase 两个内部技能；其 Protect checkpoint 为
`models/defender_protect_mlp_ctde_repro_20260526/final_model.pth`。规则策略只能用于环境诊断，
不能进入正式 Defender pool。

## 联合覆盖与效率目标

评测必须同时回答三个问题：

1. 联合覆盖：`Protect OR Chase` 能阻止多少 paired cases；
2. 总体性能：可实现的固定策略或切换策略能否提高 Defender success，并控制 breach/collision；
3. 效率：在成功局中能否提高 capture rate，并降低 capture step、episode length 和路径长度。

每个 Attacker × seed 必须保存 Protect 与 Chase 的逐局 paired 记录。报告至少包含：

- Protect-only、Chase-only、oracle union 的 capture/timeout/breach/collision；
- Protect-only success、Chase-only success、双方共同 success、双方共同 failure；
- 独占成功 cohort 的 capture step、episode length 与路径长度；
- 任意切换策略相对 Protect-only 和 Chase-only 的 paired 转移矩阵。

oracle union 只表示覆盖上界，不等于在线切换器的可实现性能。任何切换方案都必须独立报告
`success -> breach`、`timeout -> capture`、collision 和额外时长，不能只报告覆盖率。

## 策略生成门槛

新 Defender specialist 只允许从 Protect/Chase 的共同失败或效率劣势状态中训练。候选晋级必须满足：

- 使用未参与调参的 paired fixed-seed holdout；
- 在至少两个独立 seed block 中提供正向独占覆盖；
- 不显著降低 Protect + Chase 的联合成功率；
- 对新增成功局或已有成功局给出 capture/时长效率证据；
- checkpoint 按真实终局指标选择，不按 shaped reward 选择。

当前不启动新 HRL top、meta-controller、局内 Router 或 payoff solver。先确认两个底层技能的
联合覆盖边界和可训练的 failure-state cohort，再决定是否生成第三个 learned specialist。

## 工程边界

- 标准评测复用 `eval/vs.py`；专项矩阵必须保持相同终局和重置契约。
- 保留通用 heuristic registry、Attacker PPO、Defender PPO、checkpoint loader、paired evaluation 和 CoppeliaSim 验证链。
- active 模型目录只保留正式 checkpoint、必要 initializer、run config 和紧凑证据。
- 下层策略库和交叉矩阵稳定之前，不新增上层切换训练或 solver 依赖。
