# Protect–Chase 捕获互补与安全边界

最后更新：2026-08-10

## 当前结论

active Protect 与 recurrent Chase 存在稳定的 episode-level 互补。冻结的双时点 gate 能显著把
Protect timeout 转成 capture，但仍出现极少量新增 breach，因此只能作为诊断性效率基线，不能称为
逐局安全保证或可部署 Defender。

最终评测使用 6 个冻结 Attacker 和三个互不重叠的 paired seed blocks：
`988000..988099`、`1088000..1088099`、`1188000..1188099`，共 1,800 局：

| 系统 | Capture | Timeout | Breach | Collision |
| --- | ---: | ---: | ---: | ---: |
| Protect | 1023 | 589 | 188 | 0 |
| guarded Protect→Chase | 1244 | 371 | 185 | 0 |

paired 转换为 218 个 `timeout -> capture`、3 个 `breach -> capture`、1 个
`timeout -> breach` 和 1 个 `breach -> timeout`；全部 1,023 个 Protect capture 均被保留。
总体 capture 增加 221、timeout 减少 218、breach 减少 3，但严格的“每局都不新增 breach”不成立。

收益具有明显的对手条件性：221 个净 capture 增益中，`heuristic_evasive` 贡献 166 个，
`rl_ppo_evasive` 贡献 42 个；其余四个 Attacker 合计贡献 13 个。因此该结果证明特定失败分布上的
互补效率，不证明通用切换优势。

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

保留该规则作为冻结技能互补效率基线，不再继续搜索瞬时 heuristic 阈值。后续仍以相同 simulator
snapshot 的独占成功和共同失败分析指导 learned 下层策略生成，不启动新的 top、局内 Router 或
payoff solver。
