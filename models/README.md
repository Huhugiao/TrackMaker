# Model Retention Policy

Only active checkpoints, retained research milestones, and provenance-critical benchmarks should live here.

Current retained groups:

- `defender_protect_mlp_ctde_frozen6_20260721_105148`: active Protect; deploy
  `best_balanced_model.pth`.
- `defender_chase_nmn_dual_gru_raw_dense_05-05-19-12`: active recurrent Chase.
- `hrl_ch2_m1_astar_cached_top_20260606_170036`: active A* path-risk HRL; its frozen
  internal skill layout remains the original baseline + recurrent Chase pair.
- `defender_protect2_dense_02-11-17-34`: retired historical D0/Protect2 benchmark;
  do not include it in the active Defender pool.
- `defender_baseline_mlp_ctde_repro_20260526`: retired standalone baseline benchmark;
  retained because the active HRL still uses it internally.
- `hrl_regime_adaptive_toponly_20260513_222519`: recovered Chapter 2 initialization checkpoint.
- `hrl_ch2_m1_euclidean_top_20260606_163158`: retained Euclidean ablation.
- `attacker_nmn_mlp_multistyle_ppo_two_style_20260715_003835`: two-style PPO bootstrap; retain the continuation initializer and compact training/evaluation evidence.
- `attacker_nmn_mlp_diversity_continuation_20260715_120331`: formal RL Attacker; retain `best_goal_rush.pth`, `best_evasive.pth`, the run contract, and compact comparisons.

Failed four-style PPO/SAC, historical A0, and rejected pursuer/interceptor/sentinel checkpoints are kept only in Git history, not in the working tree.

Generated GIFs, smoke checkpoints, superseded `best/latest/final` variants, and TensorBoard-only copies should stay under ignored output directories or external experiment storage. A checkpoint is retained in the main tree only when it is selected by a documented metric, initializes a retained continuation, or is required to reproduce a recorded negative result.
