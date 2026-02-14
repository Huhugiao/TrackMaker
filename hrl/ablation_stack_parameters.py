"""GRU/Macro/NMN/CTDE 消融训练参数（顺序训练两底层+一上层）。"""


class AblationSetupParameters:
    """运行级配置。"""
    GPU_ID = 1
    BASE_OUTPUT_DIR = 'models'


class AblationSkillTrainingParameters:
    """底层技能（MLP no-CTDE）训练超参数。"""
    lr = 1e-3
    LR_FINAL = 3e-4
    LR_SCHEDULE = 'cosine'
    MINIBATCH_SIZE = 4096
    N_EPOCHS_INITIAL = 10
    N_EPOCHS_FINAL = 10
    VALUE_CLIP_RANGE = 0.2
    CLIP_RANGE = 0.2
    RATIO_CLAMP_MAX = 4.0
    EX_VALUE_COEF = 0.5
    ENTROPY_COEF = 0.02
    MAX_GRAD_NORM = 0.5

    N_ENVS = 4
    N_STEPS = 2048
    N_MAX_STEPS = 1.2e7
    LOG_EPOCH_STEPS = int(1e4)
    TBPTT_STEPS = 32

    GAMMA = 0.95
    LAM = 0.95
    REWARD_NORMALIZATION = True

    TRAINING_MODE = 'rl'
    IL_INITIAL_WEIGHT = 0.0
    IL_FINAL_WEIGHT = 0.0
    IL_ANNEAL_STEPS = 1


class AblationTopTrainingParameters:
    """上层网络（MLP no-CTDE）训练超参数。"""
    lr = 3e-3
    LR_FINAL = 1e-3
    LR_SCHEDULE = 'cosine'
    MINIBATCH_SIZE = 4096
    N_EPOCHS_INITIAL = 10
    N_EPOCHS_FINAL = 10
    VALUE_CLIP_RANGE = 0.2
    CLIP_RANGE = 0.2
    RATIO_CLAMP_MAX = 4.0
    EX_VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    MAX_GRAD_NORM = 0.5

    N_ENVS = 4
    N_STEPS = 2048
    N_MAX_STEPS = 2.4e7
    LOG_EPOCH_STEPS = int(1e4)
    TBPTT_STEPS = 32

    GAMMA = 0.95
    LAM = 0.95
    REWARD_NORMALIZATION = True

    TRAINING_MODE = 'rl'
    IL_INITIAL_WEIGHT = 0.0
    IL_FINAL_WEIGHT = 0.0
    IL_ANNEAL_STEPS = 1


class AblationRecordingParameters:
    """统一记录频率。"""
    EVAL_INTERVAL = 100000
    SAVE_INTERVAL = 300000
    GIF_INTERVAL = 1000000
    TRAJ_INTERVAL = 200000
    EVAL_EPISODES = 48
    TENSORBOARD = True


class AblationEnvParameters:
    """环境配置。"""
    EPISODE_LEN = 300
    ATTACKER_STRATEGY = 'switch_random'
