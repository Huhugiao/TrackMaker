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
from pathlib import Path

from configs.map_config import EnvParameters, ObstacleDensity
from configs.paths import CHECKPOINTS_DIR


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
    GPU_ID = 0               # 使用的GPU序号（0或1），所有训练都在这个GPU上运行
    
    # 障碍物密度等级 (none, dense)
    OBSTACLE_DENSITY = ObstacleDensity.DENSE
    
    # 技能模式: "chase", "protect1", "protect2", "baseline"
    # protect1: 导航到target阶段 (静止对手, 到达即成功)
    # protect2: 保护target阶段 (导航对手, 任务胜负条件)
    # baseline: 端到端PPO基线（老版密集引导奖励）
    SKILL_MODE = "baseline"

    # 底层网络类型：
    # - "nmn"
    # - "nmn_ctde"
    # - "nmn_ctde_shared"
    # - "nmn_ctde_task_shared"
    # - "nmn_ctde_task_shared_distill"（belief / privileged latent distillation）
    # - "nmn_no_shared_radar"（nmn消融：actor/critic 不共享 radar encoder）
    # - "nmn_dual_gru_raw"（tracking/raw radar 分离GRU，非CTDE critic）
    # - "nmn_dual_gru_raw_ctde"（dual-GRU actor + CTDE dual-GRU critic，仅保留最小 recurrent 入口）
    # - "nmn_gru"
    # - "mlp"（底层训练时默认按非CTDE版本使用）
    # - "mlp_ctde"
    BOTTOM_NETWORK_TYPE = "mlp_ctde"

    # 下一轮 Defender 的六成员冻结 Attacker 池。程序化与 PPO alias 显式区分，
    # 每个 policy 等权；learned alias 不能依赖 checkpoint 默认 head。
    TRAIN_ATTACKER_STRATEGIES = [
        "heuristic_default",
        "heuristic_evasive",
        "heuristic_geometry_feint_v3",
        "heuristic_occlusion_dash_v2",
        "rl_ppo_goal_rush",
        "rl_ppo_evasive",
    ]
    TRAIN_ATTACKER_STRATEGY_WEIGHTS = [1.0 / 6.0] * 6
    TRAIN_ATTACKER_STRATEGY_WEIGHT_SCHEDULE = None  # 例如 [[0.55,0.10,0.35],[0.42,0.22,0.36]]
    TRAIN_ATTACKER_STRATEGY_WEIGHT_SWITCH_STEPS = None  # 例如 [0, 1200000]
    TRAIN_LEARNED_ATTACKER_SPECS = {
        "rl_ppo_goal_rush": {
            "checkpoint": str(
                CHECKPOINTS_DIR
                / "attacker_nmn_mlp_diversity_continuation_20260715_120331"
                / "best_goal_rush.pth"
            ),
            "reward_style": "goal_rush",
        },
        "rl_ppo_evasive": {
            "checkpoint": str(
                CHECKPOINTS_DIR
                / "attacker_nmn_mlp_diversity_continuation_20260715_120331"
                / "best_evasive.pth"
            ),
            "reward_style": "evasive",
        },
    }
    TRAIN_LEARNED_ATTACKER_ALIAS = "attacker_rl"


class TrainingParameters:
    """
    训练超参数
    """
    # --- 优化器设置 ---
    lr = 1e-3                # 初始学习率
    LR_FINAL = 3e-4          # 最终学习率
    LR_SCHEDULE = 'cosine'   # 学习率调度方式 ('cosine', 'linear', 'constant')
    
    # --- 训练流程设置 ---
    N_ENVS = 6               # 并行环境数量
    N_STEPS = 2048           # 每个环境采样的步数 (PPO Rollout Length)
    N_MAX_STEPS = 3e7        # 最大训练总步数
    # NMN-CL 两阶段课程学习（仅对 NMN 系列底层网络生效）:
    # 阶段1: 无障碍环境 + dummy obstacle input，先学习任务分支
    # 阶段2: 恢复真实障碍与真实 obstacle input，继续联合微调
    ENABLE_NMN_CL = False
    NMN_CL_STAGE1_STEPS = int(1e7)
    NMN_CL_STAGE2_STEPS = int(4e7)
    NMN_CL_START_STAGE = 1
    NMN_CL_STAGE1_OBSTACLE_DENSITY = ObstacleDensity.NONE
    NMN_CL_STAGE2_OBSTACLE_DENSITY = SetupParameters.OBSTACLE_DENSITY
    LOG_EPOCH_STEPS = int(1e4) # 每隔多少步记录一次日志
    
    MINIBATCH_SIZE = 4096    # PPO更新的Mini-batch大小
    N_EPOCHS = 6             # PPO每批数据重复更新轮数
    
    # --- 序列长度设置 ---
    TBPTT_STEPS = 128        # 截断反向传播的时间步长 (Context Window大小)
    
    # --- PPO 核心参数 ---
    VALUE_CLIP_RANGE = 0.2   # Value Loss的Clip范围
    CLIP_RANGE = 0.2         # Policy Loss的Clip范围 (PPO Clip)
    RATIO_CLAMP_MAX = 4.0    # Importance Sampling Ratio的最大值
    EX_VALUE_COEF = 0.5      # Value Loss的系数
    ENTROPY_COEF = 0.02      # Entropy Bonus的系数
    ADV_ACTION_BC_COEF = 0.0 # 纯RL: 正优势样本上的采样动作回归，默认关闭
    ADV_ACTION_BC_MAX_WEIGHT = 3.0
    MAX_GRAD_NORM = 0.3      # 梯度裁剪阈值
    AUX_DISTILL_COEF = 0.2   # belief / privileged latent distillation 辅助损失权重
    MULTITASK_CHASE_VALUE_COEF = 0.05
    MULTITASK_BASELINE_VALUE_COEF = 0.05
    MULTITASK_COLLISION_COEF = 0.02
    MULTITASK_AUX_RETURN_CLIP = 20.0
    GAMMA = 0.95            # 折扣因子
    REWARD_NORMALIZATION = True  # 奖励标准化（Running Return Normalization）
    LAM = 0.95               # GAE参数 lambda

    # Optional PPO behavior anchor for fine-tuning a distilled policy.
    # Default is off; experiments can enable it from the launch script.
    POLICY_ANCHOR_ENABLE = False
    POLICY_ANCHOR_NETWORK_TYPE = "mlp"
    POLICY_ANCHOR_CHECKPOINT = None
    POLICY_ANCHOR_COEF = 0.0
    BEST_MODEL_METRIC = "reward"  # reward, win, or capture
    EARLY_STOP_ENABLED = False
    EARLY_STOP_MIN_STEPS = 0
    EARLY_STOP_PATIENCE = 5
    EARLY_STOP_MIN_DELTA = 0.0
    BALANCED_EVAL_ATTACKERS = ()
    BALANCED_EVAL_EPISODES = 0
    BALANCED_EVAL_METRIC = "mean_win"
    
    # --- 模仿学习 (IL) 设置 ---
    # 训练模式: 'mixed' (IL+RL), 'rl' (Pure RL), 'il' (Pure IL)
    TRAINING_MODE = "rl"  # 测试模式：纯RL训练（无模仿学习）
    
    # IL 余弦退火参数
    IL_INITIAL_WEIGHT = 1.0      # 初始IL权重
    IL_FINAL_WEIGHT = 0.0        # 最终IL权重
    IL_ANNEAL_STEPS = 1e7  # 退火步数


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
    PRIVILEGED_SCALAR_LEN = 8  # Attacker标量部分 (含defender朝向)
    
    # Input Vectors (Scalar + Embedded Radar)
    # RAW dimensions (for buffers and env interaction)
    ACTOR_RAW_LEN = ACTOR_SCALAR_LEN + RADAR_DIM           # 7 + 64 = 71
    PRIVILEGED_RAW_LEN = PRIVILEGED_SCALAR_LEN + RADAR_DIM # 8 + 64 = 72
    CRITIC_RAW_LEN = ACTOR_RAW_LEN + PRIVILEGED_RAW_LEN    # 71 + 72 = 143
    
    # ENCODED dimensions (for network internal processing)
    ACTOR_VECTOR_LEN = ACTOR_SCALAR_LEN + RADAR_EMBED_DIM      # 7 + 8 = 15
    PRIVILEGED_LEN = PRIVILEGED_SCALAR_LEN + RADAR_EMBED_DIM   # 8 + 8 = 16
    CRITIC_VECTOR_LEN = ACTOR_VECTOR_LEN + PRIVILEGED_LEN      # 15 + 16 = 31
    
    ACTION_DIM = 2           # 动作维度 (Angle, Speed)
    
    # MLP 参数 (用于HRL顶层的CTDE网络)
    HIDDEN_DIM = 128         # 隐藏层维度
    NUM_HIDDEN_LAYERS = 3    # 隐藏层层数
    # HRL顶层动作配置：
    # - 默认仍兼容旧版「仅选skill」模式
    # - 当启用 duration bins 时，顶层动作为联合离散宏动作(skill × duration_bin)
    HRL_NUM_SKILLS = 2
    HRL_DURATION_BINS = (1,)
    HRL_NUM_DURATION_BINS = len(HRL_DURATION_BINS)
    HRL_TOP_DISCRETE_ACTION_DIM = HRL_NUM_SKILLS * HRL_NUM_DURATION_BINS
    HRL_TOP_ACTION_DIM = HRL_NUM_SKILLS + 1

    # NMN (Neural Modular Network) 参数 (用于底层技能训练)
    NMN_BRANCH_DIM = 32      # 并行分支输出维度 (跟踪分支/避障分支各32)
    NMN_MERGED_DIM = 64      # 合并层维度 (2 * BRANCH_DIM)
    NMN_CRITIC_HIDDEN = 64   # Critic MLP隐藏层维度
    NMN_CRITIC_LAYERS = 2    # Critic MLP隐藏层数
    NMN_DUAL_GRU_TRACKING_HIDDEN = 32
    NMN_DUAL_GRU_OBSTACLE_HIDDEN = 32
    NMN_DUAL_GRU_OBSTACLE_DECAY = 0.75
    NMN_DUAL_GRU_INITIAL_LOG_STD = -1.0
    NMN_DUAL_GRU_MIN_LOG_STD = -2.0
    NMN_DUAL_GRU_MAX_LOG_STD = -1.0
    NMN_DUAL_GRU_LOG_STD_ANNEAL_ENABLE = False
    NMN_DUAL_GRU_LOG_STD_ANNEAL_START = 0
    NMN_DUAL_GRU_LOG_STD_ANNEAL_END = 0
    NMN_DUAL_GRU_LOG_STD_ANNEAL_FINAL_MAX = -1.0
    NMN_DUAL_GRU_INITIAL_MEAN_BIAS = (0.0, 0.0)
    NMN_DUAL_GRU_POLICY_HEAD_GAIN = 1.41421356237
    NMN_DUAL_GRU_RESIDUAL_WARMUP_STEPS = 0
    NMN_DUAL_GRU_DEEP_HIDDEN = 64
    NMN_DUAL_GRU_DEEP_MERGED_DIM = 128
    NMN_DUAL_GRU_DEEP_READOUT_LAYERS = 3
    HRL_TOP_DUAL_GRU_BRANCH_HIDDEN = 32
    HRL_TOP_DUAL_GRU_OBSTACLE_DECAY = 0.75
    MULTITASK_AUX_HIDDEN = 64
    AUX_POLICY_MIN_LOG_STD = -2.0
    AUX_POLICY_MAX_LOG_STD = -0.2
    
    # 上下文窗口长度 (用于数据处理)
    CONTEXT_WINDOW = TrainingParameters.TBPTT_STEPS
    CONTEXT_LEN = CONTEXT_WINDOW  # 兼容gru_predictor


class RecordingParameters:
    """
    日志与记录参数
    """
    EXPERIMENT_PROJECT = "TAD_PPO"
    
    # 根据技能模式自动命名
    EXPERIMENT_NAME = f"defender_{SetupParameters.SKILL_MODE}_{SetupParameters.BOTTOM_NETWORK_TYPE}"
    
    ENTITY = "user"
    EXPERIMENT_NOTE = "TAD Defender PPO training with IL+RL hybrid"
    TIME = datetime.datetime.now().strftime("_%m-%d-%H-%M")
    
    RETRAIN = False           # 是否继续训练 (加载权重和进度)
    FRESH_RETRAIN = False     # 仅加载模型权重，重置训练进度和学习率调度
    RESTORE_DIR = None        # 恢复模型的目录(None表示从头训练)
    
    TENSORBOARD = True        # 是否使用TensorBoard
    TXT_LOG = True            # 是否记录TXT日志
    
    # 路径设置 (包含障碍物密度等级)
    _DENSITY_TAG = f'_{SetupParameters.OBSTACLE_DENSITY}'
    MODEL_PATH = str(CHECKPOINTS_DIR / f"{EXPERIMENT_NAME}{_DENSITY_TAG}{TIME}")
    SUMMARY_PATH = str(Path(MODEL_PATH) / "summary")
    GIFS_PATH = str(Path(MODEL_PATH) / "gifs")
    
    # 频率设置
    EVAL_INTERVAL = 100000    # 评估间隔 (步数)
    SAVE_INTERVAL = 300000    # 保存模型间隔 (步数)
    BEST_INTERVAL = 0         # (未使用)
    GIF_INTERVAL = 2000000     # 保存GIF间隔 (步数)
    TRAJ_INTERVAL = 800000    # 保存轨迹图间隔 (步数)
    EVAL_EPISODES = 48        # 评估时的对局数
    
    # Loss 名称列表 (用于日志记录)
    LOSS_NAME = [
        'total', 'policy', 'entropy', 'value', 'adv_std',
        'approx_kl', 'value_clip_frac', 'clipfrac', 'grad_norm', 'adv_mean'
    ]
