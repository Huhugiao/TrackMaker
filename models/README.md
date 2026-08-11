# Model Retention Policy

Only active checkpoints, retained research milestones, and required initializers live here.

Current retained groups:

- `defender_protect_mlp_ctde_frozen6_20260721_105148`: active Protect; deploy `best_balanced_model.pth`.
- `defender_protect_mlp_ctde_repro_20260526`: Protect checkpoint used by the frozen Chapter 2 HRL.
- `defender_chase_nmn_dual_gru_raw_dense_05-05-19-12`: active recurrent Chase.
- `hrl_ch2_m1_astar_cached_top_20260606_170036`: active A* path-risk HRL over Protect + Chase.
- `hrl_regime_adaptive_toponly_20260513_222519`: recovered Chapter 2 initialization checkpoint.
- `hrl_ch2_m1_euclidean_top_20260606_163158`: retained Euclidean ablation.
- `attacker_nmn_mlp_multistyle_ppo_two_style_20260715_003835`: two-style PPO initializer.
- `attacker_nmn_mlp_diversity_continuation_20260715_120331`: formal RL Attacker checkpoints.

Failed policy families and superseded checkpoints remain only in Git history. Generated GIFs, smoke checkpoints,
raw traces, and TensorBoard copies stay outside the tracked model tree.
