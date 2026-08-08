# 下层攻防策略生成计划

最后更新：2026-07-22

## 当前研究范围

当前阶段生成、冻结、评测和筛选 Attacker/Defender 策略。暂不训练新的上层策略分配网络，
也不运行 payoff solver；已冻结的 Chapter 2 A* path-risk HRL 作为 active Defender 使用，
但不继续训练或修改其顶层及内部 baseline+chase 技能布局。

统一环境契约为：单 Attacker、单 Defender、单非学习 Target；真实速度
`A=2.0`、`D=2.6`；Attacker 成功只认 `reason == attacker_caught_target`；
`defender_collision` 为 draw；timeout 记录为 Defender 阻止 Target breach，但必须与真实 capture
分开报告。

## Attacker 双通道生成

Attacker 是更难通过梯度训练的一侧，因此同时使用两条生成路线：

1. heuristic learning：提出机制假设，实现确定性或带固定种子的程序化策略，在真实环境中按
   paired development/holdout seeds 评测，再按效果和行为差异晋级；
2. pure RL：从随机权重训练连续控制策略，使用不同 reward style、课程或训练对手形成行为差异，
   不使用 BC、DAgger、A0 warm-start、action shield 或 teacher anchor。

当前冻结候选：

| 正式名称 | 来源 | 定位 | 状态 |
| --- | --- | --- | --- |
| `heuristic_default` | programmatic | direct A* 进攻锚点 | 保留 |
| `heuristic_evasive` | programmatic | 生存/规避风格锚点 | 保留；不是性能晋级 |
| `heuristic_geometry_feint_v3` | heuristic learning | 初始几何门控诱骗 | holdout 晋级 |
| `heuristic_occlusion_dash_v2` | heuristic learning | 有界遮挡 staging 后突进 | holdout 晋级 |
| `rl_ppo_goal_rush` | pure RL PPO | 直接目标推进 | 晋级候选 |
| `rl_ppo_evasive` | pure RL PPO | 学习型规避 | 晋级候选 |

历史 A0、`progress_multipath_v3`、`deadline_risk_v2` 和其他失败 heuristic 只保留结论与
Git 历史，不在主工作树中保留实现。程序化与 RL 的两个 `evasive` 必须使用上述显式名称，
不能在结果表中合并。

两个正式 RL checkpoint 分别为：

- `models/attacker_nmn_mlp_diversity_continuation_20260715_120331/best_goal_rush.pth`
- `models/attacker_nmn_mlp_diversity_continuation_20260715_120331/best_evasive.pth`

## Defender RL-only 生成

正式 Defender 策略只允许由 RL 训练。规则 chase/protect 可以用于环境诊断和 sanity check，
但不进入正式 Defender pool，也不能替代正式 Attacker RL 训练所需的学习型 Defender。

2026-07-22 用户确认的正式学习型 Defender pool 为：

- `rl_protect_frozen6`
- `rl_recurrent_chase`
- `rl_hrl_astar_path_risk`

`rl_pursuer`、`rl_interceptor`、`rl_sentinel` 已在 6 个冻结 Attacker × 48 paired seeds 的
统一矩阵中被淘汰。三者的 Defender success 分别为 `61.46%`、`48.96%`、`37.50%`，均低于
baseline 的 `91.32%`、protect2 的 `88.89%`、HRL 历史基线的 `86.46%` 和 recurrent chase
的 `77.43%`。它们不再进入默认训练或评测池，专用实现和 checkpoint 已删除，只保留 reward
设计负结论。protect2 的同 seed 补充评测共 288 局，breach 为 `11.11%`；原始逐 episode
输出已清理，结论保留在 `docs/project_history.md`。

当前 checkpoint 映射为：

- Protect：`models/defender_protect_mlp_ctde_frozen6_20260721_105148/best_balanced_model.pth`
- recurrent chase：`models/defender_chase_nmn_dual_gru_raw_dense_05-05-19-12/final_model.pth`
- A* path-risk HRL：`models/hrl_ch2_m1_astar_cached_top_20260606_170036/best_model.pth`

同一冻结六对手、`187000..187099` paired seeds 的复测显示：新 Protect 为 `90.50%` Defender
success，旧 Protect2 为 `90.00%`；两者 payoff 基本重合。新 Chase 为 `75.67%`，旧 recurrent
Chase 为 `84.50%`，且旧模型在六个对手上均占优。因此旧 Protect2 和旧 standalone baseline
退出 active pool但保留历史 checkpoint，新 Chase不晋级；A* HRL 以不同的局内技能切换机制取代
旧 Protect2 的池位置。

淘汰模型的历史结论见 `docs/project_history.md`；已删除资产可从 Git 历史恢复。

## 统一晋级矩阵

下一份正式矩阵应覆盖全部 6 个 Attacker 和 3 个 active learned Defender。不同报告、不同 Defender 集合或
不同出生分布下的成功率不能直接横向排序。每个单元至少记录：

- Target breach、Defender capture、timeout、双方 collision；
- episode length、路径长度、Target progress；
- paired 独占成功/独占阻止及 exact test；
- 轨迹、切换、等待、绕行和碰撞等行为签名。

策略按以下顺序筛选：首先满足真实终局契约和可复现加载；其次检查是否被其他策略在 payoff 上
全面支配；最后检查行为是否与已有成员重复。specialist 可以因稳定的独占成功或分布覆盖价值保留，
但 shaped reward、单次高分或名称上的“风格”不能作为晋级证据。

## 互补性风险与后续思路

当前下层生成的主要风险是“伪多样性”：策略的名称、reward style 或轨迹外观不同，但真实成败高度重合，
没有形成可供未来上层选择器利用的条件互补。整局交叉矩阵可以检查不同对手与 seed 上的 episode 级互补，而局内选择
还需要进一步验证同一状态下的 state 级互补。

后续可从现有候选的失败局和分歧局中保存关键 simulator snapshot，并从相同状态分叉执行不同策略：部分策略
成功、部分失败的状态用于验证真实互补；全部策略都失败的状态可用于发现策略池的覆盖缺口，并指导新 specialist
的针对性生成。该思路暂作后续实验方向，当前不固定具体指标、训练流程或上层网络形式。

## 当前工程边界

- 保留通用 heuristic experiment/metrics/registry、active PPO、RL Defender 训练、
  checkpoint loader、paired evaluation 和 CoppeliaSim 验证链。
- active 模型目录只保留正式 checkpoint、必要 continuation initializer、run config 和紧凑证据；
  GIF、smoke checkpoint、重复 `best/latest/final` 和 raw trace 不长期保留。
- Defender trainer 已接入统一的 programmatic/checkpoint-backed 6-member Attacker spec；六个
  opponent 的 real-environment smoke 均已通过。2026-07-21 同时启动从零纯 PPO 的正式
  `150M nmn_dual_gru_raw` Chase 与 `30M mlp_ctde` Protect；两条 run 都把完整
  pool/spec/weights、reward contract 和 recurrent 参数写入 `run_config.json`。
- 下层策略库和交叉矩阵稳定之前，不新增上层网络、切换训练或 solver 依赖。
