# TrackMaker 项目历史与主线

最后更新：2026-07-22

本文档记录需要长期保留的研究结论、正式资产和当前工程方向。详细实验过程留在 Git 历史中，不再把失败分支和原始运行数据长期堆放在主工作树。

## 历史里程碑

- `713f8f8`：形成 learned protect 系列 Defender。当前正式 D0 为 `models/defender_protect2_dense_02-11-17-34/best_model.pth`。
- `bd2278b`：旧论文评测和展示链形成，提交说明为 `all data for article`。
- `f9f0f46`：recurrent chase 技能形成，正式 chase 模型为 `models/defender_chase_nmn_dual_gru_raw_dense_05-05-19-12/final_model.pth`。
- `68fbc60`：完成一次底层模型筛选。baseline 保留 MLP CTDE，失败的 GRU baseline 不再作为正式路线。
- `c05843e`：形成 Chapter 2 的 A* path-risk HRL。正式 80 局评测成功率为 `72.36%`，高于 Euclidean 对照的 `70.56%`。
- `95d3afb`、`e1a5ab1`：形成 CoppeliaSim/TurtleBot4 验证链。该部分保留为可选验证模块，不参与当前正式训练。
- `723e91b` 及其后的工作区改动：形成角色独立的 Attacker、learned protect2 对手和历史 BC+PPO A0。该谱系只保留结论与 Git 历史，不再延续。
- 2026-07-15：两风格 PPO bootstrap 与 diversity continuation 完成，形成 `goal_rush/evasive` 两个 RL Attacker checkpoint；pursuer/interceptor/sentinel 三个独立 RL Defender 完成长训练。
- 2026-07-18：程序化 Attacker heuristic learning 完成三轮开发和冻结 holdout；`geometry_feint_v3` 与 `occlusion_dash_v2` 晋级。传统 `evasive` 后续作为显式生存/规避风格锚点保留，不把它的低成功率解释为性能晋级。
- 2026-07-20：paired Defender 层级矩阵表明现有 HRL top 未超过独立 baseline/chase 子技能。该结果作为未来上层研究基线保留，但当前阶段不训练上层网络或运行 solver。
- 2026-07-20：6 个冻结 Attacker × 48 paired seeds 的统一矩阵中，pursuer/interceptor/sentinel 的 Defender success 分别仅为 `61.46%/48.96%/37.50%`，明显弱于 baseline、protect2、HRL 和 recurrent chase，三者从正式 Defender pool 淘汰并保留为 reward 设计负结果。protect2 同 seed 补充评测为 `88.89%` Defender success、`11.11%` breach。
- 2026-07-21：六个冻结 Attacker 已全部接入 Defender runner 并通过真实环境 smoke，随后完成
  baseline/chase 诊断训练。完整 run 资产已清理，只保留下述统一复测结论。
- 2026-07-22：在 6 个冻结 Attacker × 100 paired seeds 的统一复测中，新 Protect 的 Defender
  success 为 `90.50%`，与旧 Protect2 的 `90.00%` 没有显著差异，但相对旧 standalone baseline
  的 `88.17%` 有小幅配对提升；新 Chase 为 `75.67%`，被旧 recurrent Chase 的 `84.50%`
  在六个对手上全面支配。用户据此确认 active Defender pool 为“新 Protect + 旧 recurrent Chase +
  A* path-risk HRL”。旧 Protect2 和旧 standalone baseline 退出 active pool但保留为历史资产，
  新 Chase 不晋级。后续正式矩阵不再统计最近攻防距离。

### Chapter 2 复现边界

`train/train_regime_state_cf_top.py` 的历史默认初始化依赖
`models/hrl_regime_adaptive_toponly_20260513_222519/best_model.pth`。该文件曾脱离所有 branch、
tag 和 reflog，只剩本机不可达 Git tree 中的 blob
`f80018f4a66174bbb32185f34f592c2d3d0e6fb1`。本次清理已在 Git GC 前将它恢复到原路径；
文件大小为 1,209,525 bytes，SHA-256 为
`a3b3e59e9be9a63f7d9b597c813a4f7bdd2521645f22fdcb18f05613e489031b`。

默认历史初始化现可直接使用。若显式传入的 `SCF_INIT_TOP_PATH` 缺失，训练入口会在创建新实验
目录前明确失败。只有设置
`SCF_ALLOW_RANDOM_INIT=1` 才会从随机权重启动，这类运行属于新的非复现实验，配置中会记录
`init_mode=random_explicit`。

## Chapter 3 负结果

旧 `auto_hl` 分支进行了约 550 个 paired episodes 和 10 个诊断方向：2 个仅达到 diagnostic 结论，2 个因有害退化被拒绝，6 个没有稳定正信号。

最终结论是当前特征空间不足以支持可靠的 Chapter 3 controller，因此不启动该 controller 的正式训练。主工作树不再保留该 controller/gate 的实现和 `outputs/auto_hl*` raw/trace 数据，只保留负结论；完整原始总结可从提交 `c05843e` 的 `docs/2026-06-06-autoresearch-summary.md` 恢复。

## 正式保留资产

- active Protect：`models/defender_protect_mlp_ctde_frozen6_20260721_105148/best_balanced_model.pth`
- active recurrent Chase：`models/defender_chase_nmn_dual_gru_raw_dense_05-05-19-12/final_model.pth`
- active A* path-risk HRL：`models/hrl_ch2_m1_astar_cached_top_20260606_170036/best_model.pth`
- 历史 D0/Protect2：`models/defender_protect2_dense_02-11-17-34/best_model.pth`
- 历史 standalone baseline：`models/defender_baseline_mlp_ctde_repro_20260526/final_model.pth`
- Chapter 2 初始化 top：`models/hrl_regime_adaptive_toponly_20260513_222519/best_model.pth`
- Chapter 2 A* top：同上，现为 active HRL
- Chapter 2 Euclidean 对照：`models/hrl_ch2_m1_euclidean_top_20260606_163158/best_model.pth`
- PPO goal-rush Attacker：`models/attacker_nmn_mlp_diversity_continuation_20260715_120331/best_goal_rush.pth`
- PPO evasive Attacker：`models/attacker_nmn_mlp_diversity_continuation_20260715_120331/best_evasive.pth`

## 当前主线

1. 当前生成、冻结和筛选策略，不训练新的上层网络，也不运行 meta-solver；已冻结的 Chapter 2
   A* path-risk HRL 作为 active Defender 使用。
2. Attacker 同时使用 heuristic learning 与纯 RL：正式候选为程序化 `default/evasive`、
   `geometry_feint_v3`、`occlusion_dash_v2`，以及 PPO `goal_rush/evasive`。
3. Defender 正式 pool 为新 Protect、旧 recurrent Chase、A* path-risk HRL。旧 Protect2、旧 standalone
   baseline 和新 Chase只作历史对照；pursuer、interceptor、sentinel 只保留历史负结论。
4. 所有候选必须在同一环境契约和 paired fixed seeds 上形成交叉 outcome/行为矩阵；按真实终局、
   独占成功、行为差异和 payoff 非支配性晋级，不能按 shaped reward 或策略名称晋级。
5. active HRL 固定使用 Chapter 2 A* top 及其原始 baseline+chase 子技能；不以新 Protect替换其内部
   baseline，避免改变 checkpoint 的训练语义。

当前执行方案见 `docs/lower_policy_generation_plan.md`；PPO 细节见 `docs/selfplay_training_plan.md`。

## 2026-07-12 自博弈负结果

- D1 在冻结 A0 上训练后，于 `95000..95255` paired gate 中，D0 为 `234/256 = 91.41%`，D1 为 `231/256 = 90.23%`。D1 被拒绝。
- A1 从 A0 warm-start、对 D0 训练。默认分布 `97000..97255` 中，A0 为 `11/256`，A1 为 `8/256`；训练课程分层评测中，A0 为 `41/256`，A1 为 `37/256`。A1 被拒绝。
- A1b 的 bridge 加 gate-aligned 微调在默认分布 `99000..99255` 中与 A0 同为 `12/256`，没有净提升。A1b 被拒绝。
- 因此正式资产仍只有历史 A0 与 D0，没有正式 D1、A1 或 D2。线性 continuation 曾在 D1 被拒绝后仍启动 A1，说明“固定交替”调度不合理，现已废弃。
- A1、A1b、D1 只保留 compact manifest、run config 和最终 gate 结论；raw checkpoint、逐次评测和 TensorBoard 不再长期保留。一次性 launcher、promotion、gate 和 continuation 实现已从主工作树移除。

## 2026-07-13 至 2026-07-14 A_RL0 负结果与清理

- recurrent PPO、复杂单一奖励和后续 action shield 配方均未形成有效 Attacker；shield 还产生高介入率、接近 Target 时犹豫和零进度卡住。
- 2026-07-14 正式转向共享 replay、四奖励重标注的 feed-forward multi-style SAC。
- 该 SAC run 在约 `5.45M` 停止：课程成功率长期平台在 `25–28%`，默认分布始终为零，
  四 style 未形成稳定分化。主工作树已删除其实现、checkpoint、评测和日志，只保留结论。
- 2026-07-15 主线改为全 feed-forward multi-style PPO；课程升级由训练器内的确定性条件控制，
  不依赖人工或 subagent 主观判断。
- 首轮四 style PPO 在约 `503k` 停止；`goal_rush` 启动速度优于 `sparse_goal`，而
  `balanced` 与直接追击目标重叠。新训练契约只保留 `goal_rush` 与 `evasive`，四 style
  中断 checkpoint 与日志已从主工作树删除。
- 失败的 A_RL0/C0 原始输出约 100 MiB，以及旧 Attacker PPO/shield launcher、trainer、runner、安全检查模块和专用测试已删除；历史 A0 推理兼容和 checkpoint 也已移除。

## 保留边界

- 主工作树保留 active multi-style PPO trainer、checkpoint-backed policy pool、固定种子评测、正式 A* HRL 推理/评测和精选模型；失败 SAC、历史 A0 和三种 Defender reward-style 的实现由 Git 历史保存。
- Attacker BC、旧 pilot、写死 D1/A1/A1b/D2 的 launcher、promotion、gate 和 continuation 不继续维护；历史结论保留在本文和输出 manifest 中。
- deprecated DAgger/privileged/low-switch 训练入口由 Git 历史保存，不继续在主工作树维护。
- CoppeliaSim 仅保留 scene、URDF、核心环境适配、连接 smoke 和现有 checkpoint rollout；原生训练、BC、微调、realtime wrapper、GIF、录像和报告工具不继续维护。
- 论文一次性绘图、失效 grid/scatter 工具和被替代 checkpoint 不继续保留。
