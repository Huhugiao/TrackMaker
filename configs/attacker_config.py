"""Configuration for Attacker training."""

from configs.map_config import ObstacleDensity
from configs.paths import CHECKPOINTS_DIR


class SetupParameters:
    """System-level settings for attacker training and evaluation."""

    SEED = 2026
    TRAIN_USE_RANDOM_SEED = False
    TRAIN_SEED_BLOCK_SIZE = 100000
    TRAIN_SEED_STRIDE = 1
    EVAL_USE_RANDOM_SEED = False
    EVAL_FIXED_SEED = 42000
    GPU_ID = 3

    OBSTACLE_DENSITY = ObstacleDensity.DENSE
    ENV_REWARD_MODE = "standard"
    # Active Attacker training is feed-forward, multi-style PPO.  Historical
    # checkpoints remain loadable for deployment and comparison.
    NETWORK_TYPE = "attacker_nmn_mlp_multistyle_ppo"

    # Attacker-first curriculum. Speeds stay at the physical defaults while
    # target geometry is sampled across attacker-favourable, contested, and
    # defender-favourable time-margin bins. Here margin = t_attacker - t_defender
    # for reaching the target, so negative values favour the attacker.
    CURRICULUM_ENABLED = True
    CURRICULUM_BINS = (
        {
            "name": "easy_open",
            "time_margin": (-120.0, -40.0),
            "attacker_target_astar_distance": (60.0, 140.0),
            "obstacle_densities": (ObstacleDensity.NONE, ObstacleDensity.SCATTERED),
            "obstacle_density_weights": {ObstacleDensity.NONE: 0.5, ObstacleDensity.SCATTERED: 0.5},
            "weight": 0.60,
        },
        {
            "name": "contested_mixed",
            "time_margin": (-40.0, 20.0),
            "attacker_target_astar_distance": (100.0, 220.0),
            "obstacle_densities": (ObstacleDensity.SCATTERED, ObstacleDensity.DENSE),
            "obstacle_density_weights": {ObstacleDensity.SCATTERED: 0.7, ObstacleDensity.DENSE: 0.3},
            "weight": 0.30,
        },
        {
            "name": "hard_dense",
            "time_margin": (20.0, 80.0),
            "attacker_target_astar_distance": (140.0, 300.0),
            "obstacle_densities": (ObstacleDensity.DENSE,),
            "obstacle_density_weights": {ObstacleDensity.DENSE: 1.0},
            "weight": 0.10,
        },
    )
    CURRICULUM_KEEP_BASE_SPEEDS = True

    # Competence-gated curriculum.  Difficulty and reward style are separate:
    # every stage is sampled evenly by all reward styles, while the stage only
    # advances after every style passes the same fixed-seed evaluation gate.
    # A small amount of the preceding difficulty is retained after each
    # transition to reduce catastrophic forgetting.
    CURRICULUM_STAGES = (
        {
            "name": "easy",
            "train_weights": {"easy_open": 1.0},
            "gate_bin": "easy_open",
            "gate_success_rate": 0.80,
            "min_environment_steps": 500_000,
            "required_consecutive_passes": 3,
        },
        {
            "name": "medium",
            "train_weights": {"easy_open": 0.20, "contested_mixed": 0.80},
            "gate_bin": "contested_mixed",
            "gate_success_rate": 0.65,
            "min_environment_steps": 750_000,
            "required_consecutive_passes": 3,
        },
        {
            "name": "hard",
            "train_weights": {"contested_mixed": 0.20, "hard_dense": 0.80},
            "gate_bin": "hard_dense",
            "gate_success_rate": 0.50,
            "min_environment_steps": 1_000_000,
            "required_consecutive_passes": 3,
        },
        {
            "name": "default",
            "train_weights": {"hard_dense": 0.20, "default": 0.80},
            "gate_bin": "default",
            "gate_success_rate": None,
            "min_environment_steps": 0,
            "required_consecutive_passes": 0,
        },
    )

    # Weight-only continuation after the basic easy/medium policy has learned.
    # Native default geometry is present from the first rollout.  Promotion is
    # multi-objective: goal_rush must remain useful, while evasive must retain a
    # success floor and demonstrate a real capture-rate advantage.
    DIVERSITY_CONTINUATION_STAGES = (
        {
            "name": "default_bridge",
            "train_weights": {
                "contested_mixed": 0.60,
                "hard_dense": 0.10,
                "default": 0.30,
            },
            "gate_bin": "default",
            "gate_success_rate": None,
            "gate_criteria": {
                "styles": {
                    "goal_rush": {"min_success_rate": 0.08},
                    "evasive": {
                        "min_success_rate": 0.08,
                        "max_capture_rate": 0.60,
                    },
                },
                "min_evasive_capture_reduction_vs_goal_rush": 0.15,
            },
            "min_environment_steps": 1_500_000,
            "required_consecutive_passes": 2,
        },
        {
            "name": "default_mixed",
            "train_weights": {
                "contested_mixed": 0.35,
                "hard_dense": 0.20,
                "default": 0.45,
            },
            "gate_bin": "default",
            "gate_success_rate": None,
            "gate_criteria": {
                "styles": {
                    "goal_rush": {"min_success_rate": 0.12},
                    "evasive": {
                        "min_success_rate": 0.12,
                        "max_capture_rate": 0.55,
                    },
                },
                "min_evasive_capture_reduction_vs_goal_rush": 0.15,
            },
            "min_environment_steps": 2_000_000,
            "required_consecutive_passes": 2,
        },
        {
            "name": "default_focus",
            "train_weights": {
                "contested_mixed": 0.15,
                "hard_dense": 0.20,
                "default": 0.65,
            },
            "gate_bin": "default",
            "gate_success_rate": None,
            "gate_criteria": None,
            "min_environment_steps": 0,
            "required_consecutive_passes": 0,
        },
    )

    # Attacker bootstrap must use an existing learned Defender, not a rule policy.
    # The selected protect2 checkpoint was the strongest existing model in the
    # fixed-seed six-rule evaluation run on 2026-07-11.
    DEFENDER_STRATEGY = "skill_protect"
    DEFENDER_STRATEGY_POOL = ["skill_protect"]
    DEFENDER_STRATEGY_WEIGHTS = {"skill_protect": 1.0}
    # Optional named historical policies, supplied as an alias-keyed JSON object.
    # Example: {"d0": {"strategy": "skill_protect", "checkpoint": "..."}}
    DEFENDER_POLICY_SPECS = {}
    PROTECT_SKILL_PATH = str(
        CHECKPOINTS_DIR / "defender_protect2_dense_02-11-17-34" / "best_model.pth"
    )

    # Hierarchical defender bottom-skill checkpoints.
    # 2-skill mode uses baseline + chase as the HRL primitive bank.
    HRL_NUM_SKILLS = 2
    HRL_TOP_POLICY_PATH = str(
        CHECKPOINTS_DIR / "hrl_ch2_m1_astar_cached_top_20260606_170036" / "best_model.pth"
    )
    HRL_PRIMARY_SKILL_PATH = str(
        CHECKPOINTS_DIR / "defender_baseline_mlp_ctde_repro_20260526" / "final_model.pth"
    )
    HRL_CHASE_SKILL_PATH = str(
        CHECKPOINTS_DIR / "defender_chase_nmn_dual_gru_raw_dense_05-05-19-12" / "final_model.pth"
    )
    HRL_BASELINE_SKILL_PATH = None
    HRL_DEVICE = "auto"
class RewardParameters:
    """Attacker-side dense reward shaping."""

    STEP_PENALTY = -0.04
    SUCCESS_REWARD = 20.0
    CAPTURE_PENALTY = -20.0
    TIMEOUT_PENALTY = -10.0
    ATTACKER_COLLISION_PENALTY = -5.0
    ATTACKER_COLLISION_MAX_EVENTS = 3
    DEFENDER_COLLISION_REWARD = 8.0

    TARGET_PROGRESS_SCALE = 30.0
    EVADE_PROGRESS_SCALE = 4.0
    EVADE_DISTANCE_THRESHOLD = 140.0
    # The environment exposes all active style rewards for auditing.  PPO
    # workers consume only their assigned style's scalar reward, preserving
    # the on-policy contract.  Keep two behaviorally distinct objectives:
    # direct goal pursuit and explicit evasion under immediate threat.
    MULTISTYLE_REWARDS = (
        {
            "name": "goal_rush",
            "description": "Risk-seeking shortest-path pursuit with no defender fear term.",
            "STEP_PENALTY": -0.01,
            "TARGET_PROGRESS_SCALE": 30.0,
            "SUCCESS_REWARD": 20.0,
        },
        {
            "name": "evasive",
            "description": "Risk-averse pursuit that values separation under immediate threat.",
            "STEP_PENALTY": -0.005,
            "TARGET_PROGRESS_SCALE": 20.0,
            "EVADE_PROGRESS_SCALE": 6.0,
            "ATTACKER_COLLISION_PENALTY": -2.0,
            "SUCCESS_REWARD": 20.0,
            "CAPTURE_PENALTY": -20.0,
            "TIMEOUT_PENALTY": -5.0,
        },
    )

    # A* progress does not punish necessary obstacle detours the way Euclidean
    # distance shaping does. "euclidean" remains available for ablations.
    TARGET_PROGRESS_METRIC = "astar"
    PATH_GRID_SIZE = 8.0
    PATH_OBSTACLE_PADDING = 10.0


class MultiStylePPOParameters:
    """On-policy hyperparameters for competence-gated multi-style PPO."""

    N_ENVS = 12
    ROLLOUT_STEPS = 512
    N_MAX_STEPS = int(1e7)
    MINIBATCH_SIZE = 1024
    N_EPOCHS = 6

    LEARNING_RATE = 3e-4
    LR_FINAL = 3e-5
    GAMMA = 0.995
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.20
    VALUE_CLIP_RANGE = 0.20
    VALUE_COEF = 0.50
    ENTROPY_COEF = 0.01
    MAX_GRAD_NORM = 1.0
    TARGET_KL = 0.03
    LOG_STD_MIN = -3.0
    LOG_STD_MAX = 0.5
    INITIAL_LOG_STD = -0.5

    EVAL_INTERVAL = 250_000
    GATE_EVAL_EPISODES_PER_STYLE = 64
    DEFAULT_EVAL_EPISODES_PER_STYLE = 32
    SAVE_INTERVAL = 500_000
    LOG_INTERVAL = 50_000
    CHECKPOINT_METRIC = "target_success_rate"
    PLATEAU_EVAL_PATIENCE = 8
    PLATEAU_MIN_DELTA = 0.02


class NetParameters:
    """Network dimensions for the attacker model."""

    OBS_DIM = 72
    SCALAR_LEN = 8
    RADAR_DIM = 64
    PRIVILEGED_CRITIC_OBS_DIM = OBS_DIM
    RADAR_EMBED_DIM = 8
    ENCODED_DIM = SCALAR_LEN + RADAR_EMBED_DIM
    ACTION_DIM = 2
    # Legacy networks retain their scalar initialization and unbounded parameter
    # so historical checkpoints continue to load without migration.
    INITIAL_LOG_STD = 0.0
    HIDDEN_DIM = 128
    NUM_GRU_LAYERS = 1
    BRANCH_HIDDEN_DIM = 64
    ACTOR_TASK_FEATURE_DIM = 12
    DECENTRALIZED_ACTOR_TASK_DIM = 6
    DECENTRALIZED_ACTOR_OBS_DIM = DECENTRALIZED_ACTOR_TASK_DIM + RADAR_DIM
    OBSTACLE_HIDDEN_DECAY = 0.75
