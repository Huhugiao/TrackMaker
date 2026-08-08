# Attacker 多风格 PPO 与能力门控课程

最后更新：2026-07-15

> 状态说明（2026-07-20）：本文保留 PPO Attacker 与第一轮 RL Defender 的训练契约。
> 当前课题阶段已经扩展为“heuristic learning + RL Attacker、仅 RL Defender”的下层策略生成，
> 暂不训练上层网络或运行 solver。正式策略池和晋级规则见
> `docs/lower_policy_generation_plan.md`。

## 当前结论

当前正式入口：

```bash
python train/train_attacker_multistyle_ppo.py
```

bootstrap 模型从随机权重开始，使用 feed-forward `nmn_mlp`、连续二维动作 `[turn, speed]`
和学习型 Defender。不使用 RNN、action shield/hard mask、BC、DAgger、teacher action、
A0 warm-start 或旧策略 KL anchor。当前多样性续训只从已完成的同谱系纯 RL PPO
checkpoint 初始化网络权重，并重置 optimizer、训练步数和学习率计划。

2026-07-14 启动的 multi-style SAC 在约 `5.45M/10M` 停止。它在训练课程上学到
约 `25–28%` 的有效行为，但从约 `3.25M` 后进入平台；默认分布四个 style 始终
`0%`，成功 seeds 高度重合。该负结果表明固定 `60/30/10` 难度混采与大 replay
没有产生所需的课程泛化和风格分化。

## 为什么改为全 PPO

- PPO 只使用当前策略刚采集的 rollout，课程升级后不会继续学习 replay 中的旧 easy
  或早期失败数据。
- 难度变化与 reward style 分离：训练器只改变环境难度，不同时改变策略目标。
- 每个 style 都有明确的 on-policy 数据来源，能够直接检查哪个 style 没有学会。
- 严格的 frontier gate 阻止 easy 成功率掩盖 medium/hard 的失败。

相关依据：

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)：
  clipped on-policy policy update。
- [Curriculum Learning](https://dl.acm.org/doi/10.1145/1553374.1553380)：
  从容易样本逐步增加任务难度。
- [Teacher-Student Curriculum Learning](https://proceedings.mlr.press/v70/matiisen17a.html)：
  根据当前能力而非固定时间表选择课程进度。
- [Policy invariance under reward transformations for stochastic games](https://arxiv.org/abs/1401.3907)：
  shaped return 只用于学习，最终策略仍按真实终局成功率选择。

## PPO 的多风格采样契约

active style 只保留 `goal_rush` 与 `evasive`。`sparse_goal` 和 `balanced` 在首轮
四 style PPO 约 `503k` 后移除：`goal_rush` 启动更快，且已覆盖不惩罚被捕获的直接追击；
`balanced` 与直接追击目标重叠。

环境每步计算统一的 8 维 reward features 和 2 维 reward vector，但 PPO worker
固定绑定一种 style，只读取 `reward_vector[style_id]`。一条由 `goal_rush` 策略生成的
轨迹不能更新 `evasive` policy head，因为它没有生成这些动作。

bootstrap 默认 12 个 Ray workers，每 style 6 个。当前多样性续训把更多采样预算给行为
差异更明显的 `evasive`：`goal_rush=4`、`evasive=8`。每轮严格执行：

```text
同步 policy version N
  -> 12 workers 各采 512 on-policy steps
  -> 校验所有 rollout 都属于 version N，且两 style 分别为 2048/4096 steps
  -> 按 style 分别计算/标准化 advantage
  -> 分层构造每个 minibatch，保证两 style 都参与；loss 按 style 等权平均
  -> policy version N+1
```

禁止沿用 SAC 的 double buffering。PPO 更新后，旧权重预采的下一批数据已是 stale
rollout，训练器会拒绝 policy version 不匹配的数据。

## 奖励风格

`defender_collision` 在所有风格中为 draw，不暗中算作 Attacker 成功。

| 风格 | success | capture | timeout | step | A* progress | evade | collision event |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `goal_rush` | `+20` | `0` | `0` | `-0.01` | `30` | `0` | `0` |
| `evasive` | `+20` | `-20` | `-5` | `-0.005` | `20` | `6` | `-2` |

两种 style 必须分别使用自身 on-policy rollout，不能互相重标注奖励。

## 网络与优化

- actor：70 维 Attacker-centric 真值观测；task geometry 与 64 维 radar 分支融合。
- value：72 维 privileged state，CTDE 训练。
- actor/value 各有共享 trunk 和两个独立 style head；每 style 有独立 `log_std`。
- 无 GRU、hidden state 或 TBPTT。
- `gamma=0.995`、`GAE lambda=0.95`、clip `0.2`、6 epochs。
- 每轮 `12 * 512 = 6144` environment steps；minibatch `1024`。
- bootstrap learning rate 从 `3e-4` 线性降到 `3e-5`；续训重置为从 `1e-4` 线性降到
  `3e-5`；target KL `0.03`。
- advantage 按 style 独立标准化；value loss 按 style 的 return scale 平衡。
- 第一轮总预算 `10M` environment steps；多样性续训预算 `6M`。

## 能力门控课程

真实速度始终为 `A=2.0`、`D=2.6`。课程只改变出生几何和障碍密度。

| 阶段 | 训练分布 | frontier gate | 全 style 门槛 | 最少驻留 |
| --- | --- | --- | ---: | ---: |
| `easy` | 100% easy | `easy_open` | `80%` | `500k` |
| `medium` | 20% easy + 80% medium | `contested_mixed` | `65%` | `750k` |
| `hard` | 20% medium + 80% hard | `hard_dense` | `50%` | `1M` |
| `default` | 20% hard + 80% native default | default | final | — |

每 `250k` 在固定 paired seeds 上评测当前 frontier。只有同时满足以下条件才升级：

1. 当前阶段达到最少 environment steps；
2. 两个 style 都达到门槛；
3. 上述条件连续 3 次成立。

单次高分、训练 recent success 或混合课程平均成功率都不能触发升级。升级前保存
promotion checkpoint，下一 rollout 边界统一 reset 全部 workers。连续 8 次 frontier
评测没有至少 `2%` 改进时输出 plateau alert，但不会擅自跳级。

第一轮实际跑到 `10.002M`：medium 成功率为 `goal_rush=65.62%`、`evasive=57.81%`；
default 成功率为 `3.12%/9.38%`，捕获率为 `84.38%/56.25%`。因此续训不再从 easy
重来，也不只追求最高胜率，而使用以下 default 常驻课程：

| 续训阶段 | 训练 episode 分布 | 核心门控 |
| --- | --- | --- |
| `default_bridge` | 60% medium + 10% hard + 30% default | 两 style 成功率、evasive 捕获率与捕获率降幅 |
| `default_mixed` | 35% medium + 20% hard + 45% default | 同上，连续 2 次达标 |
| `default_focus` | 15% medium + 20% hard + 65% default | final |

这里的百分比是每个 episode 开始时的抽样概率；日志中的 environment step 占比会因
不同难度 episode 长短不同而波动。`evasive` 相比 `goal_rush` 至少降低 15 个百分点的
Defender 捕获率才满足多样性门控，避免两个 head 收敛成同一种直接追击策略。

## 评测、checkpoint 与巡检

- frontier gate：每 style 64 局固定 seeds。
- default 分布：每 style 32 局；进入 final stage 后直接复用 64 局 default gate。
- `goal_rush` 按 default `target_success_rate` 选择 best；`evasive` 使用 default 成功率、
  相对捕获率降幅和 timeout 的组合分数，保留真正不同的行为而非只保留最高胜率。
- 每个 checkpoint 保存 optimizer、policy version、课程 stage 和门控状态，可恢复训练。
- `Ctrl-C` 会保存 `interrupted_model.pth`，避免再次丢失停止点权重。

训练器内条件是课程升级的唯一权威。subagent/人工巡检只检查进程退出、NaN、KL 或
clip fraction 异常、吞吐下降、长期零成功和 plateau alert，不做主观阶段切换。

## 验证与启动

```bash
python train/train_attacker_multistyle_ppo.py --dry-run \
  --num-envs 4 --rollout-steps 8 --output-dir /tmp/trackmaker_ppo_dry

python train/train_attacker_multistyle_ppo.py

python train/train_attacker_multistyle_ppo.py \
  --init-checkpoint models/attacker_nmn_mlp_multistyle_ppo_two_style_20260715_003835/final_model.pth \
  --curriculum-profile diversity_continuation \
  --style-env-counts goal_rush=4,evasive=8 \
  --num-envs 12 --max-steps 6000000 \
  --learning-rate 1e-4 --final-learning-rate 3e-5
```

正式启动后，用实际前几轮吞吐和首个 `250k` gate 重新估计完成时间。只有 Attacker
在默认分布形成有效挑战后，才进入 Defender 更新或 PSRO-lite。

## 筛选后的下一轮 Attacker 池

多样性续训完成后，使用相同的 64 个固定 seeds 对 learned best 与规则 Attacker 做了
medium/default 对照。下一轮 Defender 候选池只保留三个互补成员：

| alias | 类型 | 部署 head | 权重 | medium 成功率 | default 成功率 |
| --- | --- | --- | ---: | ---: | ---: |
| `default` | rule A* 追击 | — | `0.30` | `78.12%` | `10.94%` |
| `ppo_goal_rush` | PPO checkpoint | `goal_rush` | `0.30` | `64.06%` | `10.94%` |
| `ppo_evasive` | PPO checkpoint | `evasive` | `0.40` | `60.94%` | `9.38%` |

`ppo_goal_rush` 在 default 与 rule default 成功率相同，但捕获率从 `82.81%` 降到
`65.62%`；`ppo_evasive` 的 default 捕获率为 `31.25%`，因此两者不是规则策略的简单
复制。rule evasive 的 default 成功率仅 `1.56%`、超时率 `95.31%`，不进入主池。

Defender runner 使用 alias-keyed `learned_attacker_specs` 同时加载多个冻结 checkpoint，
每个 alias 显式记录 `checkpoint` 与 `reward_style`。正式训练前必须强制每个 alias 做真实
环境 smoke test，并把完整 pool/spec/weights 写入 run config 与 checkpoint metadata。

## Defender 第一轮多风格 best response

> 历史负结果（2026-07-20）：本节只记录已经完成的 pursuer/interceptor/sentinel 配方。
> 统一 6-Attacker × 48-seed 评测后，三者均从正式 Defender pool 淘汰；专用实现、入口、测试和
> checkpoint 已从主工作树删除，可从 Git 历史恢复。下一轮已回到旧 baseline reward，并使用
> `train/train_defender_baseline_ppo.py` 和完整六成员冻结 Attacker pool。

Attacker 池已经满足进入 Defender 轮次的条件。三个池成员在本轮完全冻结；训练期间不再
更新 Attacker，也不把 Defender 的 reward style 和对手身份绑定：每个 Defender 都按
`default=0.30`、`ppo_goal_rush=0.30`、`ppo_evasive=0.40` 采样同一个池。

本轮不是把三个 style 放进共享网络，而是启动三个独立的 feed-forward `mlp_ctde` PPO
checkpoint。共享 head/trunk 虽然节省采样，但也会给三个 best response 提供表征层面的
行为塌缩通道；独立模型更符合“形成 Defender 策略池”的目标。环境仍一次计算统一的
10 维 transition-local feature 和三维 reward vector，每个训练进程只消费自己 style 的
标量 reward。

环境的物理 outcome 标签保持统一，但 style utility 可以不同：timeout 仍记录为
Defender outcome win；对 `pursuer` 则按“没有完成抓捕”给予 `-6`，不把拖延当成它的
成功行为。

| 风格 | step | pursuit progress | blocking progress | race progress | control/步 | capture | breach | timeout | D collision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pursuer` | `-0.005` | `4` | `0` | `0` | `0` | `+10` | `-10` | `-6` | `-10` |
| `interceptor` | `-0.001` | `0` | `4` | `0` | `+0.005` | `+7` | `-12` | `+4` | `-10` |
| `sentinel` | `-0.0005` | `0` | `0` | `4` | `+0.008` | `+3` | `-15` | `+10` | `-10` |

三个 progress 不再是到三个虚拟目标点的距离，而是三个不同任务量：

- `pursuit_progress`：令 `g=max(distance(D,A)-capture_radius, 0)`，单步奖励特征为
  `(g_prev-g_curr)/g_initial`。分母是本局初始捕获边界距离。
- `blocking_progress`：截击误差由两部分组成：Defender 偏离有限 A-T 线段的三角不等式
  excess，以及 Defender 在线段上偏离 55% 控制位置的 station error。单步特征为
  `(cost_prev-cost_curr)/cost_initial`；这衡量“封堵几何改善”，不是到构造点的欧氏导航。
- `race_progress`：先分别用 Defender、Attacker 各自的初始 target 剩余距离归一化它们
  当步的目标进度，再计算 `defender_fractional_progress - attacker_fractional_progress`。
  因此 Defender 原地不动而 Attacker 接近目标时会得到负进度。

`intercept_control` 只在当前截击误差低于初始 A-T 距离的 10% 时每步 `+0.005`；
`guard_control` 只在 Defender 位于 target 60 px 内且比 Attacker 更接近 target 时每步
`+0.008`。完整 449 步上限分别约为 `2.25` 和 `3.59`。三个 progress 都用本局初始任务
距离/误差归一化，静态几何下正反移动累计为零，所有加权项通过
`info['defender_reward_terms']` 单独记录。

统一训练参数为：纯 RL、`gamma=0.995`、GAE `lambda=0.95`、clip `0.2`、6 epochs、
12 workers × 512 steps、minibatch 1024、学习率从 `3e-4` 线性降到 `3e-5`，每个
style `10M` environment steps。reward normalization 关闭，不用不可跨 style 比较的
shaped return 选模型。`pursuer` 的 best 按 capture rate、balanced best 按三个对手中的
最低 capture rate 选择，避免把 timeout outcome win 误选为追捕风格；`interceptor` 和
`sentinel` 按真实 Defender win/min-win 选择。每次评测对三个 Attacker 各跑 32 个固定
seeds，并记录 capture/timeout/breach/collision 结果分布。

三组训练入口与 smoke test 已随负结果代码清理，不再提供可执行命令。

进入下一轮 Attacker 更新前，不只比较总体胜率。至少要求三者对同一 paired seeds 都有
有效胜率，并检查行为签名：`pursuer` 的 capture rate/抓捕速度应领先，`sentinel` 的
timeout rate 应明显更高，`interceptor` 应主要沿 target-attacker 通道形成截击。没有形成
这些差异的 checkpoint 即使 shaped reward 很高也不进入 Defender 池。
