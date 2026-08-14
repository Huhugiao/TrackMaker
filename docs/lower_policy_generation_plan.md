# 下层攻防策略生成计划

最后更新：2026-08-14

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

### Attacker pool 对 Defender 的已观测价值

使用未参与安全层选择的 paired seed blocks `686300..686359`、`786300..786359`，对 2026-05
legacy MLP Protect 与当前 frozen6 Protect 分别做 `raw/radar_steer` 交叉诊断，共 2,880 局。
`radar_steer` 下，当前 Protect 在全部 720 个 cases 上成功 `675` 局，legacy Protect 成功 `653`
局；paired 转移为 `31` 个旧失败转成功、`9` 个旧成功转失败。只看四个新增 Attacker 时为
`447/480` 对 `431/480`，paired `24/8`。

进一步限制为 `initial_target_time_margin >= 20` 且两个 checkpoint 均无 terminal collision：

- 四个新增 Attacker：`404/417` 对 `389/417`，提升 `3.60pp`，paired `16/1`；
- 两个新增程序化 Attacker：`202/209` 对 `192/209`，提升 `4.78pp`，paired `10/0`；
- 两个新增 RL Attacker：`202/208` 对 `197/208`，提升 `2.40pp`，paired `6/1`，现有样本不足以
  单独确认其贡献。

该结果说明 frozen6 Protect 在安全执行系统和高裕度策略 cohort 中存在正向关联，主要证据来自
`geometry_feint_v3/occlusion_dash_v2`；但两个 checkpoint 的 reward、对手池和选择流程并非只差
“是否加入新 Attacker”，因此不能作因果归因。若要宣称新增 Attacker 提升 Defender，必须从随机
权重使用同一训练配方和相同 `radar_steer` rollout，比较 full pool 与 leave-out pool；当前结果只
作为是否值得启动该受控重训的前置证据。

## Defender 两技能主线

正式底层技能只包括：

- Protect：`models/defender_protect_mlp_ctde_frozen6_20260721_105148/best_balanced_model.pth`
- recurrent Chase：`models/defender_chase_nmn_dual_gru_raw_dense_05-05-19-12/final_model.pth`

冻结 HRL 使用 Protect + Chase 两个内部技能；其 Protect checkpoint 为
`models/defender_protect_mlp_ctde_repro_20260526/final_model.pth`。规则策略只能用于环境诊断，
不能进入正式 Defender pool。

### Chase 训练与评测终局契约

Chase 的 PPO rollout 保持 2026-05 active recurrent Chase 的纯追捕语义：Attacker 接触 Target
不终止 episode，也不产生 Target breach 终局惩罚；只有 Defender 捕获 Attacker、Defender
碰撞或达到 449-step 上限才结束。超时记为 `timeout_task_failed`，不能作为 Chase 抓捕成功。

固定种子开发评测、checkpoint 选择和 holdout 评测必须切换到标准 TAD 终局：Attacker 接触
Target 立即结束并记为 `attacker_caught_target`；Defender capture、timeout、Target breach 和
`defender_collision` 分开统计，collision 仍为 draw。`run_config.json` 必须显式记录训练
`reward_mode=chase`、`target_breach=ignored_non_terminal`，以及评测
`evaluation_reward_mode=standard`、`terminal_contract=standard_tad`。不得用训练 rollout 的 Chase
终局口径生成正式 `target_success_rate` 或选择 checkpoint。

### Defender 本地雷达安全层

`envs/defender_radar_safety.py` 提供套在底层网络外的恒速转向投影。它只读取 Defender actor
观测中的 64 维 radar 与网络原始动作，不读取地图、Attacker/Target 真值或 privileged critic
状态。安全层按网络当前速度外推 6 步，从 33 个转向候选中选择离网络动作最近的安全转向；速度
分量逐位保留，不允许通过减速或停车避障。默认净空为 `agent_radius + 4.0`。安全动作（包括
低速或零速动作）必须严格透传；上一次逃逸方向只用于在后续同等小的危险动作修正之间打破
左右平局，不能用虚构高速轨迹覆盖网络动作。

专项 paired 评测使用 `eval/run_defender_hierarchy_attacker_matrix.py --defender-safety-mode
radar_steer`。早期 seeds `286300..286359` 被连续用于参数选择，且当时的低速锁会用虚构高速
覆盖真实安全动作；其 `collision=0/720` 只保留为 development 记录，不能作为当前实现的 holdout
证据。

修正后在未参与选择的 seeds `486300..486359`、6 个冻结 Attacker、Protect/Chase 各 360 局上：

- raw：collision `51/720`，Defender success `615/720`，capture `482/720`，target success `54/720`；
- radar-steer：collision `7/720`，Defender success `652/720`，capture `503/720`，target success
  `61/720`；逐步回放确认 7 次碰撞均在恒速候选全部不安全的 `unavoidable` 状态终止；
- Protect collision `32 -> 0`，但有 `1` 个 raw success 转为 breach；Chase collision `19 -> 7`，
  且有 `4` 个 raw success 转为 breach、`3` 个 raw success 转为 collision；
- 总体改善不能抵消 paired regression。当前冻结 checkpoint 不允许把 radar-steer 当作无损默认
  执行层，也不能宣称它保证零碰撞。

安全层训练模式必须在每个 run 启动前固定，不能中途切换。本轮 raw Chase 重训未晋级，
当前不继续训练新 Chase；若重启 safety execution 训练，必须作为独立 run 并保留 raw 对照。
Protect/Chase 的 32-step、单 PPO update smoke 只证明训练链可运行，不构成性能证据。禁止用 BC、
DAgger、teacher/anchor、KL-to-BC 或旧 checkpoint warm-start 消除 paired regression。

旧 controller/env obstacle mask 已从执行、训练和评测入口删除。它读取 simulator 全局地图，
枚举减速候选并允许停车，既不满足本地观测约束，也会引入卡死风险；历史 masked 结果仅作审计，
当前只允许 `raw` 与 `radar_steer` 两种执行模式。

## 联合覆盖与效率目标

在新的切换扫描或技能训练前，先完成失败归因：

1. `low-margin spawn`：按出生时 Euclidean/A* Target 时间裕度分层；低裕度是独立 stress cohort，
   不自动从总指标中删除，也不能直接归因于策略；
2. `collision/control`：记录 raw terminal collision、雷达安全层干预和不可避免标记；
3. `high-margin strategy failure`：高裕度、无碰撞控制混杂且 Protect/Chase 共同失败的状态。

只有第三类才用于判断底层策略覆盖缺口或继续做 snapshot 切换。当前不先运行全时点
recoverability sweep。

评测必须同时回答三个问题：

1. 联合覆盖：`Protect OR Chase` 能阻止多少 paired cases；
2. 总体性能：可实现的固定策略或切换策略能否提高 Defender success，并控制 breach/collision；
3. 效率：在成功局中能否提高 capture rate，并降低 capture step、episode length 和路径长度。

每个 Attacker × seed 必须保存 Protect 与 Chase 的逐局 paired 记录。报告至少包含：

- Protect-only、Chase-only、oracle union 的 capture/timeout/breach/collision；
- Protect-only success、Chase-only success、双方共同 success、双方共同 failure；
- 独占成功 cohort 的 capture step、episode length 与路径长度；
- 出生时 Euclidean/A* 时间裕度、绕障系数以及按裕度分层的 outcome；
- controller/env obstacle mask 状态必须显式为关闭，并记录 radar-steer 开关、干预率与不可避免率；
- 任意切换策略相对 Protect-only 和 Chase-only 的 paired 转移矩阵。

oracle union 只表示覆盖上界，不等于在线切换器的可实现性能。任何切换方案都必须独立报告
`success -> breach`、`timeout -> capture`、collision 和额外时长，不能只报告覆盖率。raw 与 radar-steer
结果属于不同执行系统，不能互相替代。

## 策略生成门槛

新 Defender specialist 只允许从完成上述归因后的 high-margin strategy failure 或明确的效率劣势
状态中训练。low-margin spawn 用于单独评估鲁棒边界；collision/control failure 优先归入安全控制，
不能直接包装成新策略需求。候选晋级必须满足：

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
