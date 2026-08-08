"""HRL 训练专用参数。"""

from configs.paths import CHECKPOINTS_DIR


class HRLEnvTrainParameters:
    """HRL 训练环境配置。"""
    # 将训练episode最大步长单独收敛到HRL配置，避免依赖全局默认值。
    EPISODE_LEN = 449

    # 底层技能模型路径：固定为 2-skill baseline+chase。
    # 支持相对路径或绝对路径。
    PRIMARY_SKILL_NAME = "baseline"
    PROTECT_MODEL_PATH = str(CHECKPOINTS_DIR / "defender_baseline_mlp_ctde_repro_20260526" / "final_model.pth")
    CHASE_MODEL_PATH = str(CHECKPOINTS_DIR / "defender_chase_nmn_dual_gru_raw_dense_05-05-19-12" / "final_model.pth")
    BASELINE_MODEL_PATH = None

    ATTACKER_STRATEGY = 'random'
    # 训练对手策略池（可自由选择子集）：
    # None / [] -> 使用默认 TRAINING_STRATEGIES / or ['default', 'evasive']
    ATTACKER_TRAINING_STRATEGIES = ['default', 'evasive']
    # Runner推理设备：根据测速结论默认使用CPU（False）。
    RUNNER_USE_GPU = False
    RUNNER_GPU_ID = 0

    # 联合离散宏动作：top action = skill × duration_bin。
    # 默认保持旧版兼容（仅单步duration），避免影响其他HRL训练流程；
    # 需要启用macro_step时，请在单独launcher里显式覆写为多档duration。
    MACRO_DURATION_BINS = [1]
    # 时长先验代价：默认关闭，仅在启用多档macro duration时配合开启。
    MACRO_DURATION_COST = 0.0

    # 宏动作提前打断：默认关闭，避免改变现有训练行为；
    # 启用macro_step实验时再显式打开。
    ENABLE_EARLY_INTERRUPT = False
    EARLY_INTERRUPT_MIN_STEPS = 1
    EARLY_INTERRUPT_VISIBILITY_CHANGE = True
    EARLY_INTERRUPT_PRIMARY_URGENCY = 0.60
    EARLY_INTERRUPT_CHASE_URGENCY = 0.40

    # 旧版连续hold控制保留作兼容；联合离散宏动作启用时不会走这条分支。
    HOLD_MIN = 1
    HOLD_MAX = 1
    DISABLE_HOLD_CONTROL = True
