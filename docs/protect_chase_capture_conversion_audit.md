# Protect–Chase 捕获互补与安全边界

最后更新：2026-08-11

## 当前结论

带 controller obstacle mask 的 active Protect 与 recurrent Chase 存在 episode-level 效率互补。
冻结的双时点 gate 能显著把 Protect timeout 转成 capture，但仍出现极少量新增 breach，因此只能
作为 shielded 执行系统的诊断性效率基线，不能称为 raw 底层技能互补、逐局安全保证或可部署
Defender。

最终评测使用 6 个冻结 Attacker 和三个互不重叠的 paired seed blocks：
`988000..988099`、`1088000..1088099`、`1188000..1188099`，共 1,800 局：

| 系统 | Capture | Timeout | Breach | Collision |
| --- | ---: | ---: | ---: | ---: |
| masked Protect | 1023 | 589 | 188 | 0 |
| masked guarded Protect→Chase | 1244 | 371 | 185 | 0 |

paired 转换为 218 个 `timeout -> capture`、3 个 `breach -> capture`、1 个
`timeout -> breach` 和 1 个 `breach -> timeout`；全部 1,023 个 Protect capture 均被保留。
总体 capture 增加 221、timeout 减少 218、breach 减少 3，但严格的“每局都不新增 breach”不成立。

收益具有明显的对手条件性：221 个净 capture 增益中，`heuristic_evasive` 贡献 166 个，
`rl_ppo_evasive` 贡献 42 个；其余四个 Attacker 合计贡献 13 个。因此该结果证明特定失败分布上的
互补效率，不证明通用切换优势。

上述两个系统都在每一步对 Defender 原始动作应用
`BaseDefenderController._apply_defender_hard_obstacle_mask`。因此表中的 `collision=0` 只说明 mask
修正后的动作没有触发 terminal collision；入口没有记录 raw 危险动作、mask intervention 或
zero fallback，不能据此声称 Protect/Chase 本身具备碰撞安全性。`eval/vs.py` 的 standalone
Protect/Chase 标准执行默认不启用该 mask，二者属于不同执行系统。

## 出生与碰撞归因补充审计

默认出生先要求 Defender–Attacker 中心距离至少 `150`，Target 与任一 Agent 中心距离至少 `80`，
再检查 `dist(A,T) * 2.6 > dist(D,T) * 2.0`。该条件只表示 Euclidean 中心距离下 Defender 理论上
更早到达 Target，不考虑障碍绕行、初始朝向、capture 扇区、视线和控制延迟，也不是移动拦截的
reachability certificate。

对正式三个 seed blocks 的 300 个唯一出生状态做静态诊断；同一出生状态被 6 个 Attacker 重复
使用。使用 `grid_size=8`、`obstacle_padding=8` 的 A* Target 边界时间估计时，`4/300` 的裕度
不大于 0，`50/300` 的裕度小于 20 steps，且没有 curriculum fallback。A* 指标仍是静态最短路
近似，不能把这些状态直接判为严格物理不可解，但证明出生难度必须单独分层。

另从每个 block 等间隔抽取 10 个 seeds，即每个 block 的 offset `0,10,...,90`，与 6 个 Attacker
组成每技能 180 局的 raw、greedy、无 obstacle mask 诊断：

| 技能 | Capture | Timeout | Breach | Collision |
| --- | ---: | ---: | ---: | ---: |
| raw Protect | 99 | 54 | 17 | 10 |
| raw Chase | 120 | 9 | 29 | 22 |

Protect 与 Chase 分别有 `10/180` 和 `22/180` 局在原始动作首次撞障碍时终止。paired oracle union
成功 `162/180`：双方共同成功 120 局、Protect 独占成功 33 局、Chase 独占成功 9 局、共同失败
18 局；共同失败中 16 局为双方 breach，2 局为 Protect collision + Chase breach。

按出生时 Euclidean Target 边界时间裕度分层，oracle union 分别为：

| 出生裕度 | Paired cases | Union success | Common failure |
| --- | ---: | ---: | ---: |
| `margin <= 0` | 12 | 6 | 6 |
| `0 < margin < 20` | 24 | 17 | 7 |
| `margin >= 20` | 144 | 139 | 5 |

该 180 局结果是归因诊断，不是正式晋级矩阵；它表明低裕度出生和碰撞控制对失败有实质影响，
而高裕度样本中的 raw 共同失败较少。旧 1,800 局 masked gate 结果不能单独回答 raw 联合覆盖或
策略能力上限。

## 冻结双时点 gate

整局最多切换一次：

1. step 0–63 只执行 Protect。
2. 在 step 64 判断一次；若几何条件未成立，在 step 80 再判断一次，此后不再判断。
3. 每次计算：
   - `target_time_margin = tau_A_to_T - tau_D_to_T`
   - `tau_A_to_T = max(||A-T|| - reach_radius, 0) / 2.0`
   - `tau_D_to_T = max(||D-T|| - reach_radius, 0) / 2.6`
   - `attacker_target_progress_16 = ||A_(s-16)-T|| - ||A_s-T||`
4. 只有同时满足以下几何条件才提出 Chase：
   - `target_time_margin >= 40`
   - `||D-T|| >= 15`
   - `attacker_target_progress_16 <= 24`
5. proposal 后比较 Protect 与 reset Chase 的归一化动作。动作距离 `<= 2.0` 时清空 Chase GRU
   hidden 并持续 Chase 到终局；距离 `> 2.0` 时否决 Chase，并锁定 Protect 到终局。

gate 不读取 Attacker identity 或事后终局，但几何条件来自 simulator privileged A/D/T 真值。
动作 compatibility 也是经验 filter，不是 reachability certificate。该规则因此不进入正式 learned
Defender pool，也不替换 Chapter 2 HRL 内部技能。

## 评测与资产边界

- 环境：Conda `lnenv`，CPU-only，`A=2.0`、`D=2.6`。
- 每个系统都按相同 attacker/seed 独立完整 reset；checkpoint 使用 greedy 推理。
- recurrent Chase 在实际切换时显式 reset hidden，此后持续到真实终局。
- `defender_capture`、`target_success`、`timeout`、`defender_collision` 分开统计，collision 为 draw。
- Protect 与 candidate 都启用 controller obstacle mask；这是相对 standalone 标准评测的显式偏离。
- 评测入口保存 `run_config.json`、Protect 逐局结果、candidate 逐局结果和紧凑 summary；raw 输出不进入项目。

冻结 checkpoint：

- Protect：`models/defender_protect_mlp_ctde_frozen6_20260721_105148/best_balanced_model.pth`，
  SHA-256 `3b04bb6de12f67bb61186d41b1bf5f100e2fcba7ddd067222e5f65a08da4206b`。
- Chase：`models/defender_chase_nmn_dual_gru_raw_dense_05-05-19-12/final_model.pth`，
  SHA-256 `eb05578fae6498fd5379ff15efef9c99a0e25b1233dcf577929f7e80150e6a92`。

## 最小复现

```bash
protect_chase_smoke_dir=$(mktemp -d /tmp/protect_chase_guarded.XXXXXX)
/home/cyq/miniconda3/bin/conda run --no-capture-output -n lnenv \
  python eval/run_protect_chase_switch_sweep.py \
  --attackers heuristic_evasive --seeds 1188001 \
  --snapshot-start 64 --snapshot-interval 16 \
  --burst-steps 0 --hidden-modes reset \
  --gate-target-time-margin 40 \
  --gate-min-defender-target-distance 15 \
  --gate-max-attacker-target-progress-16 24 \
  --chase-max-protect-action-disagreement 2 \
  --direct-gate --gate-decision-end 80 \
  --output-dir "$protect_chase_smoke_dir"
rg -q '"protect_outcome": "timeout".*"candidate_outcome": "defender_capture".*"switch_step": 80' \
  "$protect_chase_smoke_dir/candidate_episodes.jsonl"
```

完整复现使用上述 6 个 Attacker，分别运行三个 100-seed blocks。入口为
`eval/run_protect_chase_switch_sweep.py`。

## 下一步

保留该规则作为 shielded 冻结技能互补效率基线，不再继续搜索瞬时 heuristic 阈值。下一步先在
统一 paired cases 上分离 low-margin spawn、collision/control 和 high-margin strategy failure，并
把 raw 与 masked 结果作为独立 ablation。只有 collision-free 的高裕度共同失败才进入 snapshot
切换或 learned 下层策略生成；不启动新的 top、局内 Router 或 payoff solver。
