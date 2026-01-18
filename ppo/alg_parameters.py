"""
PPO算法参数配置 - TAD (Tracking-and-Attacking Defense) 项目

此文件定义了PPO训练的所有超参数，包括：
- 系统设置
- 训练超参数
- 网络结构参数
- 记录参数

环境说明：
- Defender: RL控制的智能体，视野受限
- Attacker: 使用规则策略的对手
- Target: 静态目标
"""

import datetime
from map_config import EnvParameters, ObstacleDensity


class SetupParameters:
    """
    系统设置参数
    """
    # --- 随机种子设置 ---
    SEED = 1234              # 基础随机种子（用于网络初始化等的可复现性）
    
    # 训练时的种子设置
    TRAIN_USE_RANDOM_SEED = True   # 训练时是否使用随机种子（True=每次reset随机，False=固定种子）
    
    # 评估时的种子设置
    EVAL_USE_RANDOM_SEED = True    # 评估时是否随机（True=随机，False=固定）
    EVAL_FIXED_SEED = 42           # 评估时的固定种子（当EVAL_USE_RANDOM_SEED=False时使用）
    
    # --- GPU设置 ---
    NUM_GPU = 1              # 使用的GPU数量
    USE_GPU_LOCAL = False    # 是否在本地（Runner）使用GPU
    USE_GPU_GLOBAL = False   # 是否在全局（Driver/Learner）使用GPU
    
    # 预训练模型路径
    PRETRAINED_DEFENDER_PATH = None  # 预训练Defender模型路径
    
    # 障碍物密度等级 (none, dense)
    OBSTACLE_DENSITY = ObstacleDensity.DENSE
    
    # 技能模式: "chase" 或 "protect"
    SKILL_MODE = "protect"


class TrainingParameters:
    """
    训练超参数
    """
    # --- 优化器设置 ---
    lr = 3e-4                # 初始学习率
    LR_FINAL = 1e-5          # 最终学习率
    LR_SCHEDULE = 'cosine'   # 学习率调度方式 ('cosine', 'linear', 'constant')
    
    # --- 训练流程设置 ---
    N_ENVS = 4               # 并行环境数量
    N_STEPS = 2048           # 每个环境采样的步数 (PPO Rollout Length)
    N_MAX_STEPS = 4e7        # 最大训练总步数
    LOG_EPOCH_STEPS = int(1e4) # 每隔多少步记录一次日志
    
    MINIBATCH_SIZE = 64      # PPO更新的Mini-batch大小
    N_EPOCHS_INITIAL = 10    # N_EPOCHS 初始值
    N_EPOCHS_FINAL = 10      # N_EPOCHS 最终值 (线性衰减)
    
    # --- 序列长度设置 ---
    TBPTT_STEPS = 32         # 截断反向传播的时间步长 (Context Window大小)
    
    # --- PPO 核心参数 ---
    VALUE_CLIP_RANGE = 0.2   # Value Loss的Clip范围
    CLIP_RANGE = 0.2         # Policy Loss的Clip范围 (PPO Clip)
    RATIO_CLAMP_MAX = 4.0    # Importance Sampling Ratio的最大值
    EX_VALUE_COEF = 0.5      # Value Loss的系数
    ENTROPY_COEF = 0.01      # Entropy Bonus的系数
    MAX_GRAD_NORM = 0.5      # 梯度裁剪阈值
    GAMMA = 0.99             # 折扣因子
    LAM = 0.95               # GAE参数 lambda
    
    # --- 模仿学习 (IL) 设置 ---
    # 训练模式: 'mixed' (IL+RL), 'rl' (Pure RL), 'il' (Pure IL)
    TRAINING_MODE = "rl"  # 测试模式：纯RL训练（无模仿学习）
    
    # IL 余弦退火参数
    IL_INITIAL_WEIGHT = 1.0      # 初始IL权重
    IL_FINAL_WEIGHT = 0.0        # 最终IL权重
    IL_ANNEAL_STEPS = 1e7  # 退火步数
    
    # --- 对手策略设置 ---
    # 对手类型: 'random' (按权重随机采样对手策略)
    OPPONENT_TYPE = "random"
    
    # 随机对手权重 (攻击者策略权重)
    # 根据alg_parameter里配置的概率权重来采样
    RANDOM_OPPONENT_WEIGHTS = {
        "attacker": {
            # "attacker_apf": 0.5,
            # "attacker_global": 1.0,  # 全局路径规划Attacker
            "attacker_static": 1.0,  # 测试用：静止的Attacker
        }
    }


class NetParameters:
    """
    网络结构参数
    """
    # Radar Encoding
    RADAR_DIM = 64           # 原始雷达维度
    RADAR_EMBED_DIM = 8      # 雷达编码后维度
    
    # Defender Observation (71维)
    # [attacker_info(5), target_info(2), radar(64)]
    # attacker_info: [distance, bearing, fov_edge, is_visible, unobserved_time]
    # target_info: [distance, bearing]
    DEFENDER_SCALAR_LEN = 5  # Defender标量部分 (attacker_info)
    TARGET_SCALAR_LEN = 2    # Target标量部分 (到Target的距离和方位)
    
    # Actor观测: Defender视角
    ACTOR_SCALAR_LEN = DEFENDER_SCALAR_LEN + TARGET_SCALAR_LEN  # 5 + 2 = 7
    
    # Privileged观测: Attacker完整状态 (用于Critic CTDE)
    PRIVILEGED_SCALAR_LEN = 7  # Attacker标量部分
    
    # Input Vectors (Scalar + Embedded Radar)
    # RAW dimensions (for buffers and env interaction)
    ACTOR_RAW_LEN = ACTOR_SCALAR_LEN + RADAR_DIM           # 7 + 64 = 71
    PRIVILEGED_RAW_LEN = PRIVILEGED_SCALAR_LEN + RADAR_DIM # 7 + 64 = 71
    CRITIC_RAW_LEN = ACTOR_RAW_LEN + PRIVILEGED_RAW_LEN    # 71 + 71 = 142
    
    # ENCODED dimensions (for network internal processing)
    ACTOR_VECTOR_LEN = ACTOR_SCALAR_LEN + RADAR_EMBED_DIM      # 7 + 8 = 15
    PRIVILEGED_LEN = PRIVILEGED_SCALAR_LEN + RADAR_EMBED_DIM   # 7 + 8 = 15
    CRITIC_VECTOR_LEN = ACTOR_VECTOR_LEN + PRIVILEGED_LEN      # 15 + 15 = 30
    
    ACTION_DIM = 2           # 动作维度 (Angle, Speed)
    
    # MLP 参数
    HIDDEN_DIM = 256         # 隐藏层维度
    NUM_HIDDEN_LAYERS = 3    # 隐藏层层数
    
    # 上下文窗口长度 (用于数据处理)
    CONTEXT_WINDOW = TrainingParameters.TBPTT_STEPS
    CONTEXT_LEN = CONTEXT_WINDOW  # 兼容gru_predictor


class RecordingParameters:
    """
    日志与记录参数
    """
    EXPERIMENT_PROJECT = "TAD_PPO"
    
    # 根据技能模式自动命名
    EXPERIMENT_NAME = f"defender_{SetupParameters.SKILL_MODE}"
    
    ENTITY = "user"
    EXPERIMENT_NOTE = "TAD Defender PPO training with IL+RL hybrid"
    TIME = datetime.datetime.now().strftime("_%m-%d-%H-%M")
    
    RETRAIN = False           # 是否继续训练 (加载权重和进度)
    FRESH_RETRAIN = False     # 仅加载模型权重，重置训练进度和学习率调度
    RESTORE_DIR = ""          # 恢复模型的目录
    
    TENSORBOARD = True        # 是否使用TensorBoard
    TXT_LOG = True            # 是否记录TXT日志
    
    # 路径设置 (包含障碍物密度等级)
    _DENSITY_TAG = f'_{SetupParameters.OBSTACLE_DENSITY}'
    SUMMARY_PATH = f'./models/{EXPERIMENT_NAME}{_DENSITY_TAG}{TIME}'
    MODEL_PATH = f'./models/{EXPERIMENT_NAME}{_DENSITY_TAG}{TIME}'
    GIFS_PATH = f'./models/{EXPERIMENT_NAME}{_DENSITY_TAG}{TIME}/gifs'
    
    # 频率设置
    EVAL_INTERVAL = 100000    # 评估间隔 (步数)
    SAVE_INTERVAL = 300000    # 保存模型间隔 (步数)
    BEST_INTERVAL = 0         # (未使用)
    GIF_INTERVAL = 500000     # 保存GIF间隔 (步数)
    EVAL_EPISODES = 48        # 评估时的对局数
    
    # Loss 名称列表 (用于日志记录)
    LOSS_NAME = [
        'total', 'policy', 'entropy', 'value', 'adv_std',
        'approx_kl', 'value_clip_frac', 'clipfrac', 'grad_norm', 'adv_mean'
    ]
