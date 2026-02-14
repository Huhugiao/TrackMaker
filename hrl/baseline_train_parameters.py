"""Baseline 端到端 PPO 训练参数。"""


class BaselineSetupParameters:
    """Baseline 训练运行配置。"""
    SKILL_MODE = 'baseline'
    GPU_ID = 1


class BaselineTrainingParameters:
    """Baseline PPO 超参数（独立于底层技能与HRL）。"""
    # Optimizer / PPO
    lr = 1e-3
    LR_FINAL = 3e-4
    LR_SCHEDULE = 'cosine'  # 'cosine' | 'linear' | 'constant'
    MINIBATCH_SIZE = 4096
    N_EPOCHS_INITIAL = 10
    N_EPOCHS_FINAL = 10
    VALUE_CLIP_RANGE = 0.2
    CLIP_RANGE = 0.2
    RATIO_CLAMP_MAX = 4.0
    EX_VALUE_COEF = 0.5
    ENTROPY_COEF = 0.02
    MAX_GRAD_NORM = 0.5

    # Rollout / schedule
    N_ENVS = 4
    N_STEPS = 2048
    N_MAX_STEPS = 3e7
    LOG_EPOCH_STEPS = int(1e4)
    TBPTT_STEPS = 32

    # RL returns
    GAMMA = 0.95
    LAM = 0.95
    REWARD_NORMALIZATION = True

    # Baseline uses pure RL training.
    TRAINING_MODE = 'rl'
    IL_INITIAL_WEIGHT = 0.0
    IL_FINAL_WEIGHT = 0.0
    IL_ANNEAL_STEPS = 1


class BaselineRecordingParameters:
    """Baseline 训练记录频率。"""
    EVAL_INTERVAL = 100000
    SAVE_INTERVAL = 300000
    GIF_INTERVAL = 1000000
    TRAJ_INTERVAL = 200000
    EVAL_EPISODES = 48
    TENSORBOARD = True


class BaselineEnvTrainParameters:
    """Baseline 训练环境配置。"""
    EPISODE_LEN = 300
    REWARD_MODE = 'baseline'
    ATTACKER_STRATEGY = 'random'  # random = 每局从支持策略集中随机采样
