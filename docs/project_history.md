# TrackMaker 项目历史与主线

最后更新：2026-08-12

本文档只记录需要长期保留的研究结论、正式资产和当前工程方向。详细实验过程由 Git 历史保存，
失败分支和原始运行数据不长期堆放在主工作树。

## 历史里程碑

- `f9f0f46`：形成 recurrent Chase，正式模型为 `models/defender_chase_nmn_dual_gru_raw_dense_05-05-19-12/final_model.pth`。
- `c05843e`：形成 Chapter 2 A* path-risk HRL；正式 80 局评测成功率为 `72.36%`，高于 Euclidean 对照的 `70.56%`。
- `95d3afb`、`e1a5ab1`：形成 CoppeliaSim/TurtleBot4 验证链；该模块不参与当前正式训练。
- 2026-07-15：两风格 PPO bootstrap 与 diversity continuation 完成，形成 `goal_rush/evasive` 两个 RL Attacker checkpoint。
- 2026-07-18：程序化 Attacker heuristic learning 完成开发和冻结 holdout；`geometry_feint_v3` 与 `occlusion_dash_v2` 晋级。
- 2026-07-20：paired Defender 层级矩阵表明现有 HRL top 未超过独立 Protect/Chase；当前阶段不训练新 top 或运行 solver。
- 2026-07-20：pursuer/interceptor/sentinel 在统一矩阵中的 Defender success 分别为 `61.46%/48.96%/37.50%`，三者退出正式 Defender pool。
- 2026-07-21：六个冻结 Attacker 全部接入 Defender runner 并通过真实环境 smoke，随后形成当前 Protect checkpoint。
- 2026-07-22：统一复测确认 active Defender pool 为 Protect、recurrent Chase 与 A* path-risk HRL；未通过门槛的新 Chase 不晋级。
- 2026-08-08 至 2026-08-10：完成带 controller obstacle mask 的 Protect 与 recurrent Chase 捕获
  互补性审计。诊断 gate 能将大量 timeout 转为 capture，但存在少量安全退化；该结果只证明
  shielded 执行系统的互补效率，不证明 raw 底层技能具备同等性能或碰撞安全性。
- 2026-08-11：补充出生与碰撞归因审计。默认出生只保证 Euclidean Target 时间优势，不覆盖
  障碍路径、朝向、capture 扇区和控制延迟；raw 小样本同时显示 Protect/Chase 均有非零 terminal
  collision。后续先分离 low-margin spawn、collision/control 与 high-margin strategy failure，
  不先继续全时点切换搜索。详细边界见 `docs/protect_chase_capture_conversion_audit.md`。
- 2026-08-11：形成只使用 Defender 64 维 radar 的恒速转向安全层。60 个 paired seeds、6 个
  Attacker、Protect/Chase 共 720 局中，terminal collision 从 `44` 降为 `0`，Defender success
  从 `580` 升为 `615`；旧 Chase 仍有 `1/273` 个 raw-success regression，因此该结果只证明
  当时 development 配置的表现，不证明冻结网络内生安全。下一轮两个底层技能使用同一安全层从
  随机权重重训，raw 与 radar-steer 继续分开评测。依赖全局地图且允许减速停车的旧
  controller/env obstacle mask 及其专项 gate 入口已经退役删除，历史 masked 结果只作审计。
- 2026-08-12：对抗性审查发现旧低速锁会修改真实安全的零速动作，已收缩为只在危险候选的
  同等小转向之间打破左右平局。新 holdout seeds `486300..486359` 上 collision `51 -> 7`、
  Defender success `615 -> 652`，但仍有 5 个 raw success 转 breach、3 个 raw success 转
  collision，主要集中于 recurrent Chase 的恒速 `unavoidable` 状态。因此该层可进入随机初始化
  重训实验，但不能无损套用到当前冻结 checkpoint；Protect/Chase 的单 PPO update smoke 已通过。
- 2026-08-12：用两个新 paired seed blocks 对 legacy Protect 与 frozen6 Protect 完成
  `raw/radar_steer` 交叉诊断。安全执行下，四个新增 Attacker 的高裕度、双方 collision-free
  cases 中 frozen6 Protect 成功率提高 `3.60pp`，paired gain/loss 为 `16/1`；证据主要来自两个
  程序化 Attacker，两个 RL Attacker 的独立贡献尚未确认。因 checkpoint 训练配方不完全同构，
  该结果只支持启动同配方 full-pool/leave-out 受控重训，不支持直接作因果声明。

## Chapter 2 复现边界

`train/train_regime_state_cf_top.py` 的历史默认初始化依赖
`models/hrl_regime_adaptive_toponly_20260513_222519/best_model.pth`。该 initializer 已恢复并保留；
文件大小为 1,209,525 bytes，SHA-256 为
`a3b3e59e9be9a63f7d9b597c813a4f7bdd2521645f22fdcb18f05613e489031b`。

若显式传入的 `SCF_INIT_TOP_PATH` 缺失，训练入口必须在创建实验目录前失败。只有设置
`SCF_ALLOW_RANDOM_INIT=1` 才允许随机初始化，并在配置中记录 `init_mode=random_explicit`。

## 正式保留资产

- active Protect：`models/defender_protect_mlp_ctde_frozen6_20260721_105148/best_balanced_model.pth`
- HRL 内部 Protect：`models/defender_protect_mlp_ctde_repro_20260526/final_model.pth`
- active recurrent Chase：`models/defender_chase_nmn_dual_gru_raw_dense_05-05-19-12/final_model.pth`
- active A* path-risk HRL：`models/hrl_ch2_m1_astar_cached_top_20260606_170036/best_model.pth`
- Chapter 2 初始化 top：`models/hrl_regime_adaptive_toponly_20260513_222519/best_model.pth`
- Chapter 2 Euclidean 对照：`models/hrl_ch2_m1_euclidean_top_20260606_163158/best_model.pth`
- PPO goal-rush Attacker：`models/attacker_nmn_mlp_diversity_continuation_20260715_120331/best_goal_rush.pth`
- PPO evasive Attacker：`models/attacker_nmn_mlp_diversity_continuation_20260715_120331/best_evasive.pth`

## 当前主线

1. Defender 底层只保留 Protect 与 Chase；所有代码、配置、模型元数据和评测结果使用这两个正式名称。
2. Attacker 同时使用 heuristic learning 与纯 RL：程序化候选为 `default/evasive`、`geometry_feint_v3`、`occlusion_dash_v2`，学习型候选为 PPO `goal_rush/evasive`。
3. 候选必须在同一环境契约和 paired fixed seeds 上形成交叉 outcome/行为矩阵，按真实终局、独占成功、行为差异和 payoff 非支配性晋级。
4. 研究首先对失败做出生裕度与碰撞控制归因，再提高 Protect + Chase 的真实联合覆盖；其次提高
   可实现总体胜率，最后优化 capture 与时长效率。
5. active HRL 固定使用 Chapter 2 A* top 及其 Protect + Chase 技能，不改变 checkpoint 的动作索引语义。

## 已确认负结果

- Chapter 3 controller 的既有特征空间不足以支持可靠调度，不启动正式训练。
- D1、A1、A1b 的历史 paired gate 均未形成稳定净提升，固定交替 continuation 已废弃。
- recurrent Attacker PPO、action shield、multi-style SAC 和四风格 PPO 均未形成可保留的正式策略。
- pursuer、interceptor、sentinel 与未通过统一复测的新 Chase 不进入正式 Defender pool。

## 保留边界

- 主工作树保留 active PPO trainer、checkpoint-backed policy pool、固定种子评测、正式 A* HRL 推理/评测和精选模型。
- BC、旧 pilot、写死 continuation/gate 的 launcher、失败 SAC 和废弃 controller 由 Git 历史保存，不继续维护。
- CoppeliaSim 只保留 scene、URDF、核心环境适配、连接 smoke 和现有 checkpoint rollout。
- 当前执行方案见 `docs/lower_policy_generation_plan.md`。
