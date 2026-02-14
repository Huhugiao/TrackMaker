"""HRL 训练专用参数。"""


class HRLSetupParameters:
    """HRL 训练模式配置。"""
    SKILL_MODE = 'hrl'
    GPU_ID = 1


class HRLTrainingParameters:
    """HRL 顶层PPO超参数（独立于底层技能训练参数）。"""
    # Optimizer / PPO
    lr = 3e-3
    LR_FINAL = 1e-3
    LR_SCHEDULE = 'cosine'  # 'cosine' | 'linear' | 'constant'
    MINIBATCH_SIZE = 4096
    N_EPOCHS_INITIAL = 10
    N_EPOCHS_FINAL = 10
    VALUE_CLIP_RANGE = 0.2
    CLIP_RANGE = 0.2
    RATIO_CLAMP_MAX = 4.0
    EX_VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    MAX_GRAD_NORM = 0.5

    # Rollout / schedule
    N_ENVS = 4
    N_STEPS = 256
    N_MAX_STEPS = 2e6
    LOG_EPOCH_STEPS = int(1e4)
    TBPTT_STEPS = 32

    # RL returns
    GAMMA = 0.95
    LAM = 0.95
    REWARD_NORMALIZATION = True

    # Keep IL fields for compatibility, though HRL top-level currently runs pure RL.
    TRAINING_MODE = "rl"
    IL_INITIAL_WEIGHT = 1.0
    IL_FINAL_WEIGHT = 0.0
    IL_ANNEAL_STEPS = 1e7


class HRLRecordingParameters:
    """HRL 训练记录频率。"""
    EVAL_INTERVAL = 100000
    SAVE_INTERVAL = 300000
    GIF_INTERVAL = 1000000
    TRAJ_INTERVAL = 200000
    EVAL_EPISODES = 48
    TENSORBOARD = True


class HRLEnvTrainParameters:
    """HRL 训练环境配置。"""
    # 将训练episode最大步长单独收敛到HRL配置，避免依赖全局默认值。
    EPISODE_LEN = 300

    ATTACKER_STRATEGY = 'random'
    PREDICTOR_HIDDEN_DIM = 64
    PREDICTOR_LR = 1e-3
    PREDICTOR_TRAIN = True

    # 固定决策步长：每个高层动作持续15个环境步。
    HOLD_MIN = 1
    HOLD_MAX = 1
    DISABLE_HOLD_CONTROL = True
