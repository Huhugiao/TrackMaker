from __future__ import annotations

"""
TAD PPO 工具函数
"""

import os
import os.path as osp
import random
import numpy as np
from typing import Dict, List, Optional
from configs import map_config

from PIL import Image as PILImage

try:
    import torch
except ImportError:
    torch = None

from configs.skill_config import SetupParameters, TrainingParameters, NetParameters
from configs.map_config import EnvParameters


def get_device(prefer_gpu: bool = True, gpu_id: int = None):
    """
    安全地获取可用的计算设备。
    
    此函数可以在没有安装CUDA的机器上安全运行，不会抛出异常。
    
    注意：在Ray worker中，Ray会通过CUDA_VISIBLE_DEVICES隔离GPU，
    所以worker内部只能看到1个GPU（索引为0），即使系统有多个GPU。
    
    Args:
        prefer_gpu: 是否优先使用GPU（如果可用）
        gpu_id: 指定GPU序号（0或1等），为None时使用SetupParameters.GPU_ID
                在Ray worker中此参数会被忽略，直接使用cuda:0
        
    Returns:
        torch.device: 可用的设备（'cuda:X' 或 'cpu'）
    """
    import os
    
    if torch is None:
        raise RuntimeError("PyTorch 未安装，无法获取训练/推理设备。")

    if not prefer_gpu:
        return torch.device('cpu')
    
    # 检测是否在Ray worker中运行
    # Ray会设置CUDA_VISIBLE_DEVICES来隔离GPU，此时worker只能看到分配给它的那个GPU
    in_ray_worker = 'RAY_WORKER_PID' in os.environ or 'CUDA_VISIBLE_DEVICES' in os.environ
    
    # 获取GPU ID
    if gpu_id is None:
        from configs.skill_config import SetupParameters
        gpu_id = SetupParameters.GPU_ID
    
    try:
        # 检查PyTorch是否编译了CUDA支持
        if not torch.cuda.is_available():
            return torch.device('cpu')
        
        # 尝试获取GPU数量，这会触发CUDA初始化
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return torch.device('cpu')
        
        # 在Ray worker中，由于GPU隔离，直接使用cuda:0
        if in_ray_worker:
            gpu_id = 0
        elif gpu_id >= device_count:
            # 主进程中检查gpu_id范围
            print(f"[警告] GPU_ID={gpu_id} 超出范围，只有{device_count}个GPU，使用GPU 0")
            gpu_id = 0
        
        # 尝试在指定GPU上创建一个小张量来验证CUDA是否真正可用
        try:
            device_str = f'cuda:{gpu_id}'
            test_tensor = torch.zeros(1, device=device_str)
            del test_tensor
            return torch.device(device_str)
        except Exception:
            return torch.device('cpu')
            
    except Exception:
        # 任何异常都回退到CPU
        return torch.device('cpu')


def is_gpu_available() -> bool:
    """
    安全地检查GPU是否可用。
    
    此函数可以在没有安装CUDA的机器上安全运行。
    
    Returns:
        bool: GPU是否可用
    """
    if torch is None:
        return False
    return get_device(prefer_gpu=True).type == 'cuda'


def get_num_gpus() -> int:
    """
    安全地获取可用的GPU数量。
    
    此函数可以在没有安装CUDA的机器上安全运行。
    
    Returns:
        int: 可用的GPU数量（如果CUDA不可用则返回0）
    """
    try:
        if torch is None:
            return 0
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()
    except Exception:
        return 0


def print_device_info():
    """打印设备信息，用于调试"""
    if torch is None:
        print("=" * 50)
        print("设备信息 (Device Information)")
        print("=" * 50)
        print("PyTorch未安装")
        print("=" * 50)
        return

    device = get_device(prefer_gpu=True)
    num_gpus = get_num_gpus()
    
    print("=" * 50)
    print("设备信息 (Device Information)")
    print("=" * 50)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {is_gpu_available()}")
    print(f"GPU数量: {num_gpus}")
    print(f"当前使用设备: {device}")
    
    if device.type == 'cuda':
        try:
            print(f"GPU名称: {torch.cuda.get_device_name(0)}")
            print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        except Exception as e:
            print(f"无法获取GPU详细信息: {e}")
    print("=" * 50)



def get_free_ram_gb() -> float:
    """
    获取当前系统空闲RAM大小（GB）。
    
    Returns:
        float: 空闲RAM大小，单位GB。如果无法获取则返回0。
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)
    except ImportError:
        # psutil未安装，尝试读取/proc/meminfo (Linux)
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        # MemAvailable: 12345678 kB
                        parts = line.split()
                        return float(parts[1]) / (1024 ** 2)  # kB -> GB
        except Exception:
            pass
        return 0.0
    except Exception:
        return 0.0


def get_adjusted_n_envs(
    base_n_envs: int,
    ram_threshold_gb: float = 20.0,
    multiplier: int = 16,
    allow_scale_up: bool = True,
) -> int:
    """
    根据空闲RAM大小动态调整并行环境数量。
    
    如果空闲RAM大于阈值，则将环境数量乘以倍数。
    
    Args:
        base_n_envs: 基础并行环境数量
        ram_threshold_gb: RAM阈值（GB），默认20GB
        multiplier: 乘数，默认16倍
        allow_scale_up: 是否允许在大内存机器上自动放大环境数量
        
    Returns:
        int: 调整后的并行环境数量
    """
    free_ram = get_free_ram_gb()
    
    if allow_scale_up and free_ram > ram_threshold_gb:
        adjusted = base_n_envs * multiplier
        print(f"[RAM检测] 空闲RAM: {free_ram:.1f}GB > {ram_threshold_gb}GB 阈值")
        print(f"[RAM检测] 并行环境数量: {base_n_envs} -> {adjusted} (x{multiplier})")
        return adjusted
    elif (not allow_scale_up) and free_ram > ram_threshold_gb:
        print(f"[RAM检测] 空闲RAM: {free_ram:.1f}GB > {ram_threshold_gb}GB 阈值")
        print(f"[RAM检测] 当前为CPU运行，禁用并行环境自动放大: {base_n_envs}")
        return base_n_envs
    else:
        print(f"[RAM检测] 空闲RAM: {free_ram:.1f}GB <= {ram_threshold_gb}GB 阈值")
        print(f"[RAM检测] 保持并行环境数量: {base_n_envs}")
        return base_n_envs


def configure_cpu_runtime(n_envs: int):
    """CPU-only训练时限制数学库线程，降低Ray多进程过度抢占。"""
    if torch is None:
        return
    if get_device(prefer_gpu=True).type != 'cpu':
        return

    cpu_count = os.cpu_count() or 1
    default_threads = max(1, min(4, int(cpu_count) // max(1, int(n_envs) + 1)))
    raw_threads = os.environ.get("TRACKMAKER_CPU_TORCH_THREADS")
    try:
        torch_threads = max(1, int(raw_threads)) if raw_threads not in (None, "") else default_threads
    except ValueError:
        print(f"[CPU适配] 忽略非法TRACKMAKER_CPU_TORCH_THREADS={raw_threads!r}")
        torch_threads = default_threads

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, str(torch_threads))

    try:
        torch.set_num_threads(torch_threads)
        torch.set_num_interop_threads(max(1, min(2, torch_threads)))
    except RuntimeError:
        pass
    print(f"[CPU适配] torch/mkl threads={torch_threads}, runners={n_envs}, cpu_count={cpu_count}")


def print_ram_info():
    """打印RAM信息，用于调试"""
    free_ram = get_free_ram_gb()
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_ram = mem.total / (1024 ** 3)
        used_ram = mem.used / (1024 ** 3)
        print(f"[RAM信息] 总计: {total_ram:.1f}GB, 已用: {used_ram:.1f}GB, 空闲: {free_ram:.1f}GB")
    except ImportError:
        print(f"[RAM信息] 空闲: {free_ram:.1f}GB (安装psutil可获取更多信息)")

def set_global_seeds(i: int):
    if torch is None:
        np.random.seed(i)
        random.seed(i)
        return
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)
    torch.backends.cudnn.deterministic = True


def _avg(vals):
    if vals is None:
        return None
    if isinstance(vals, (list, tuple)) and len(vals) > 0 and isinstance(vals[0], (list, tuple, np.ndarray)):
        return np.nanmean(vals, axis=0)
    if isinstance(vals, (list, tuple, np.ndarray)):
        return float(np.nanmean(vals)) if len(vals) > 0 else 0.0
    return vals


def write_to_tensorboard(global_summary, step: int,
                         performance_dict: Optional[Dict] = None,
                         mb_loss: Optional[List] = None,
                         imitation_loss: Optional[List] = None,
                         q_loss: Optional[float] = None,
                         evaluate: bool = True,
                         greedy: bool = True):
    if global_summary is None:
        return
    
    if imitation_loss is not None:
        global_summary.add_scalar('Loss/Imitation', imitation_loss[0], step)
        if len(imitation_loss) > 1:
            global_summary.add_scalar('Train/Imitation_Grad_Norm', imitation_loss[1], step)
    
    if q_loss is not None:
        global_summary.add_scalar('Loss/Q_Loss', q_loss, step)
    
    if performance_dict:
        prefix = 'Eval' if evaluate else 'Train'
        key_map = {
            'per_r': 'Reward',
            'per_episode_len': 'Episode_Length',
            'win': 'Win_Rate',
            'target_success': 'Target_Success_Rate',
            'defender_collision': 'Defender_Collision_Rate',
            'defender_capture': 'Defender_Capture_Rate',
            'timeout': 'Timeout_Rate',
            'other_terminal': 'Other_Terminal_Rate',
            'attacker_collision_event': 'Attacker_Collision_Event_Rate',
            'defender_collision_event': 'Defender_Collision_Event_Rate',
            'attacker_collision_steps': 'Attacker_Collision_Steps',
            'defender_collision_steps': 'Defender_Collision_Steps',
            'attacker_collision_step_rate': 'Attacker_Collision_Step_Rate',
            'defender_collision_step_rate': 'Defender_Collision_Step_Rate',
            'target_net_path_progress': 'Target_Net_Path_Progress',
            'target_best_path_progress': 'Target_Best_Path_Progress',
            'target_final_path_distance': 'Target_Final_Path_Distance',
            'attacker_hard_mask_intervention': 'Attacker_Hard_Mask_Intervention_Rate',
            'attacker_hard_mask_zero_fallback': 'Attacker_Hard_Mask_Zero_Fallback_Rate',
            'attacker_hard_mask_unsafe_passthrough': 'Attacker_Hard_Mask_Unsafe_Passthrough_Rate',
            'attacker_hard_mask_action_delta': 'Attacker_Hard_Mask_Action_Delta',
            'attacker_hard_mask_recovery_active': 'Attacker_Hard_Mask_Recovery_Active_Rate',
            'attacker_hard_mask_recovery_triggered': 'Attacker_Hard_Mask_Recovery_Trigger_Rate',
            'attacker_shield_penalty': 'Attacker_Shield_Penalty',
        }
        for k, v in performance_dict.items():
            val = _avg(v)
            if val is not None:
                name = key_map.get(k, k)
                global_summary.add_scalar(f'{prefix}/{name}', val, step)
                if k in ['per_r', 'per_episode_len'] and len(v) > 1:
                    std_val = float(np.nanstd(v))
                    global_summary.add_scalar(f'{prefix}/{name}_Std', std_val, step)
    
    if mb_loss:
        loss_vals = np.nanmean(np.asarray(mb_loss, dtype=np.float32), axis=0)
        
        mapping = {
            0: 'Total',
            1: 'Policy',
            2: 'Entropy',
            3: 'Value',
            4: 'Adv_Std',
            5: 'Approx_KL',
            7: 'Clip_Frac',
            8: 'Grad_Norm',
            9: 'Adv_Mean'
        }
        
        for idx, val in enumerate(loss_vals):
            if idx in mapping:
                name = mapping[idx]
                global_summary.add_scalar(f'Loss/{name}', float(val), step)
    
    global_summary.flush()


def make_trajectory_plot(trajectory_data, file_name, dpi=150):
    """
    Generate academic-style trajectory plot image commonly seen in
    pursuit-evasion / target-attacker-defender papers.

    Args:
        trajectory_data: dict with keys:
            'defender_traj': list of (x,y) tuples
            'attacker_traj': list of (x,y) tuples
            'target_pos': (x, y) — static target position
            'obstacles': list of obstacle dicts from map_config
            'width', 'height': map dimensions
            'win': bool — whether defender won
            'skill_mode': str — 'protect1', 'protect2', 'chase', etc.
            'episode_len': int (optional)
            'episode_reward': float (optional)
        file_name: output image path
        dpi: resolution (default 150 for screen, use 300 for paper)
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib import font_manager
        from matplotlib.collections import PatchCollection, LineCollection
        import numpy as np
    except ImportError:
        print("[WARN] matplotlib not installed, skipping trajectory plot")
        return

    d_traj = trajectory_data.get('defender_traj', [])
    a_traj = trajectory_data.get('attacker_traj', [])
    target_pos = trajectory_data.get('target_pos', None)
    obstacles = trajectory_data.get('obstacles', None)
    if not obstacles:
        obstacles = getattr(map_config, 'obstacles', [])
    w = trajectory_data.get('width', 640)
    h = trajectory_data.get('height', 640)
    win = trajectory_data.get('win', False)
    skill_mode = trajectory_data.get('skill_mode', '')
    ep_len = trajectory_data.get('episode_len', len(d_traj))
    ep_reward = trajectory_data.get('episode_reward', None)
    hrl_skill_names = trajectory_data.get('hrl_skill_names', None)
    annotation_lines = trajectory_data.get('annotation_lines', [])
    locale = str(trajectory_data.get('locale', 'en')).strip().lower()
    labels = dict(trajectory_data.get('labels', {}) or {})
    title_override = trajectory_data.get('title', None)
    caption_text = trajectory_data.get('caption', None)
    caption_divider = bool(trajectory_data.get('caption_divider', False))
    caption_y = float(trajectory_data.get('caption_y', 0.035 if caption_divider else 0.015))
    caption_divider_y = float(trajectory_data.get('caption_divider_y', caption_y + 0.065))
    legend_kwargs = dict(trajectory_data.get('legend_kwargs', {}) or {})
    subplot_adjust = dict(trajectory_data.get('subplot_adjust', {}) or {})
    figure_size = trajectory_data.get('figure_size', (8.6, 5.8))
    colors = dict(trajectory_data.get('colors', {}) or {})
    show_capture_sector = bool(trajectory_data.get('show_capture_sector', True))

    is_zh = locale.startswith('zh')

    def _pick_font(candidates):
        for font_name in candidates:
            try:
                font_manager.findfont(font_name, fallback_to_default=False)
                return font_name
            except Exception:
                continue
        return None

    if is_zh:
        font_family = 'serif'
        font_list_key = 'font.serif'
        font_candidates = [
            'AR PL UMing CN',
            'AR PL UKai CN',
            'Noto Serif CJK JP',
            'DejaVu Serif',
        ]
        default_font_list = ['AR PL UMing CN', 'DejaVu Serif']
    else:
        font_family = 'serif'
        font_list_key = 'font.serif'
        font_candidates = ['Times New Roman', 'DejaVu Serif']
        default_font_list = ['DejaVu Serif']

    selected_font = _pick_font(font_candidates)
    resolved_font_list = ([selected_font] if selected_font else []) + [
        font_name for font_name in font_candidates if font_name != selected_font
    ]

    default_labels = {
        'xlabel': '横坐标（像素）' if is_zh else '$x$ (pixels)',
        'ylabel': '纵坐标（像素）' if is_zh else '$y$ (pixels)',
        'defender': '防守方' if is_zh else 'Defender',
        'attacker': '进攻方' if is_zh else 'Attacker',
        'target': '目标点' if is_zh else 'Target',
        'skill_protect': '防守方技能：保护' if is_zh else 'Defender skill: protect',
        'skill_chase': '防守方技能：追击' if is_zh else 'Defender skill: chase',
        'skill_baseline': '防守方技能：基线' if is_zh else 'Defender skill: baseline',
        'success': '成功' if is_zh else 'Success',
        'failure': '失败' if is_zh else 'Failure',
        'mode_protect1': '保护阶段一（导航）' if is_zh else 'Protect Phase I (Navigation)',
        'mode_protect2': '保护阶段二（防御）' if is_zh else 'Protect Phase II (Defense)',
        'mode_chase': '追击阶段' if is_zh else 'Chase (Pursuit)',
        'mode_tad': '完整任务' if is_zh else 'TAD (Full Task)',
        'mode_hrl': '分层技能控制' if is_zh else 'HRL',
    }

    def _label(key, fallback=None):
        if key in labels:
            return str(labels[key])
        if key in default_labels:
            return str(default_labels[key])
        return str(fallback if fallback is not None else key)

    # ---------- Style ----------
    style_rc = {
        'font.family': font_family,
        'font.size': 10,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        'axes.unicode_minus': False,
    }
    style_rc[font_list_key] = resolved_font_list or default_font_list
    plt.rcParams.update(style_rc)

    fig, ax = plt.subplots(1, 1, figsize=figure_size)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # match screen coordinates (y down)
    ax.set_xlabel(_label('xlabel'), fontsize=11)
    ax.set_ylabel(_label('ylabel'), fontsize=11)

    # --- Obstacles ---
    obs_patches = []
    for obs in obstacles:
        if obs['type'] == 'rect':
            obs_patches.append(
                mpatches.Rectangle((obs['x'], obs['y']), obs['w'], obs['h'])
            )
        elif obs['type'] == 'circle':
            obs_patches.append(
                mpatches.Circle((obs['cx'], obs['cy']), obs['r'])
            )
        elif obs['type'] == 'segment':
            thick = float(obs.get('thick', 8.0))
            x1, y1, x2, y2 = obs['x1'], obs['y1'], obs['x2'], obs['y2']
            dx, dy = x2 - x1, y2 - y1
            length = max(1e-6, (dx**2 + dy**2)**0.5)
            nx, ny = -dy / length * thick / 2, dx / length * thick / 2
            verts = [(x1+nx, y1+ny), (x2+nx, y2+ny),
                     (x2-nx, y2-ny), (x1-nx, y1-ny)]
            obs_patches.append(mpatches.Polygon(verts))
    if obs_patches:
        pc = PatchCollection(obs_patches, facecolor='#d0d0d4',
                             edgecolor='#505058', linewidth=0.6, zorder=2)
        ax.add_collection(pc)

    # --- Color palette ---
    c_def = colors.get('defender', '#3264DC')   # defender blue
    c_atk = colors.get('attacker', '#DC5038')   # attacker red
    c_tgt = colors.get('target', '#32B450')     # target green
    skill_color_map = {
        'protect': colors.get('protect', '#4E79A7'),
        'chase': colors.get('chase', '#F28E2B'),
        'baseline': colors.get('baseline', '#59A14F'),
    }

    # --- Trajectories ---
    def _plot_trajectory(traj, color, label, skill_trace=None):
        if len(traj) < 2:
            return set()
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        n = len(xs)

        used_skills = set()
        is_hrl = str(skill_mode).lower() == 'hrl'
        skills = list(skill_trace) if isinstance(skill_trace, (list, tuple, np.ndarray)) else []
        if len(skills) == n:
            seg_skills = skills[1:]
        elif len(skills) == (n - 1):
            seg_skills = skills
        else:
            seg_skills = []

        # Main line
        ax.plot(
            xs, ys,
            color=color,
            linewidth=1.4,
            alpha=0.55 if (is_hrl and seg_skills) else 0.75,
            zorder=4,
            label=label,
        )

        if is_hrl and seg_skills:
            segments = []
            seg_colors = []
            for i in range(n - 1):
                raw_skill = seg_skills[i]
                if isinstance(raw_skill, (int, np.integer)):
                    resolved_skill_names = list(hrl_skill_names) if isinstance(hrl_skill_names, (list, tuple, np.ndarray)) else []
                    if 0 <= int(raw_skill) < len(resolved_skill_names):
                        skill_name = str(resolved_skill_names[int(raw_skill)]).strip().lower()
                    else:
                        skill_name = 'protect' if int(raw_skill) == 0 else 'chase' if int(raw_skill) == 1 else None
                else:
                    skill_name = str(raw_skill).strip().lower()
                seg_color = skill_color_map.get(skill_name, color)
                if skill_name in skill_color_map:
                    used_skills.add(skill_name)
                segments.append([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])])
                seg_colors.append(seg_color)
            lc = LineCollection(segments, colors=seg_colors, linewidths=1.8, alpha=0.9, zorder=5)
            ax.add_collection(lc)

        # Start marker (square)
        ax.plot(xs[0], ys[0], 's', color=color, markersize=7,
                markeredgecolor='white', markeredgewidth=1.0, zorder=6)
        # End marker (larger circle)
        ax.plot(xs[-1], ys[-1], 'o', color=color, markersize=8,
                markeredgecolor='white', markeredgewidth=1.0, zorder=6)

        # Time labels at start and end
        ax.annotate(f'$t_0$', (xs[0], ys[0]), textcoords='offset points',
                   xytext=(6, -6), fontsize=8, color=color, zorder=7)
        ax.annotate(f'$t_{{{n}}}$', (xs[-1], ys[-1]), textcoords='offset points',
                   xytext=(6, -6), fontsize=8, color=color, zorder=7)
        return used_skills

    d_skill_trace = trajectory_data.get('defender_skill_trace', [])
    used_skills = _plot_trajectory(d_traj, c_def, _label('defender'), skill_trace=d_skill_trace)
    if a_traj:
        _plot_trajectory(a_traj, c_atk, _label('attacker'))

    if used_skills:
        skill_legend_labels = {
            'protect': _label('skill_protect'),
            'chase': _label('skill_chase'),
            'baseline': _label('skill_baseline'),
        }
        for skill_name in ('protect', 'baseline', 'chase'):
            if skill_name in used_skills:
                ax.plot([], [], color=skill_color_map[skill_name], linewidth=1.4, alpha=0.75,
                        label=skill_legend_labels[skill_name])

    # --- Target ---
    if target_pos:
        tx, ty = target_pos
        target_r = getattr(map_config, 'target_radius', 16)
        circle_outer = plt.Circle((tx, ty), target_r, fill=False,
                                  edgecolor=c_tgt, linewidth=1.8, linestyle='--', zorder=3)
        circle_inner = plt.Circle((tx, ty), target_r * 0.3, fill=True,
                                  facecolor=c_tgt, edgecolor='none', zorder=3)
        ax.add_patch(circle_outer)
        ax.add_patch(circle_inner)
        # Cross-hair
        ch = target_r * 0.5
        ax.plot([tx-ch, tx+ch], [ty, ty], color=c_tgt, linewidth=0.8, alpha=0.6, zorder=3)
        ax.plot([tx, tx], [ty-ch, ty+ch], color=c_tgt, linewidth=0.8, alpha=0.6, zorder=3)
        ax.annotate(_label('target'), (tx, ty), textcoords='offset points',
                   xytext=(8, 8), fontsize=8, color=c_tgt,
                   fontstyle='normal' if is_zh else 'italic', zorder=7)

    # --- Capture sector at defender final position ---
    capture_r = trajectory_data.get('capture_radius', getattr(map_config, 'capture_radius', 20))
    capture_angle = trajectory_data.get('capture_sector_angle_deg',
                                         getattr(map_config, 'capture_sector_angle_deg', 30))
    d_theta_list = trajectory_data.get('defender_theta', [])
    if show_capture_sector and d_traj and d_theta_list:
        last_dx, last_dy = d_traj[-1]
        last_theta = d_theta_list[-1]
        c_cap = '#E86040'
        # Wedge: counterclockwise in data coords = clockwise on screen (inverted y)
        cap_wedge = mpatches.Wedge(
            (last_dx, last_dy), capture_r,
            last_theta - capture_angle / 2.0,
            last_theta + capture_angle / 2.0,
            facecolor=c_cap, alpha=0.18,
            edgecolor=c_cap, linewidth=1.2, linestyle='--', zorder=2)
        ax.add_patch(cap_wedge)

    # --- Title ---
    if title_override is None:
        mode_labels = {
            'protect1': _label('mode_protect1'),
            'protect2': _label('mode_protect2'),
            'chase': _label('mode_chase'),
            'tad': _label('mode_tad'),
            'hrl': _label('mode_hrl'),
        }
        title = mode_labels.get(str(skill_mode).lower(), str(skill_mode).upper())
        outcome = _label('success' if win else 'failure')
        title_str = f'{title} — {outcome}'
        if ep_len is not None:
            title_str += f'  ($T={ep_len}$)'
        ax.set_title(title_str, fontsize=12, fontweight='bold', pad=10)
    elif str(title_override).strip():
        ax.set_title(str(title_override), fontsize=12, fontweight='bold', pad=10)

    if annotation_lines:
        ann_text = '\n'.join(str(line) for line in annotation_lines if str(line).strip())
    else:
        ann_text = ''

    def _to_axes_point(point):
        if point is None:
            return None
        px, py = float(point[0]), float(point[1])
        x_axes = np.clip(px / max(float(w), 1e-6), 0.0, 1.0)
        y_axes = np.clip(1.0 - (py / max(float(h), 1e-6)), 0.0, 1.0)
        return float(x_axes), float(y_axes)

    interest_points = []
    for point in d_traj:
        xy = _to_axes_point(point)
        if xy is not None:
            interest_points.append((xy[0], xy[1], 1.0))
    for point in a_traj:
        xy = _to_axes_point(point)
        if xy is not None:
            interest_points.append((xy[0], xy[1], 1.0))
    for point in (d_traj[:1] + d_traj[-1:] + a_traj[:1] + a_traj[-1:]):
        xy = _to_axes_point(point)
        if xy is not None:
            interest_points.append((xy[0], xy[1], 2.5))
    if target_pos:
        xy = _to_axes_point(target_pos)
        if xy is not None:
            interest_points.append((xy[0], xy[1], 4.0))

    def _rect_overlap_area(rect_a, rect_b):
        ax0, ay0, ax1, ay1 = rect_a
        bx0, by0, bx1, by1 = rect_b
        ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        oy = max(0.0, min(ay1, by1) - max(ay0, by0))
        return ox * oy

    def _rect_score(rect):
        x0, y0, x1, y1 = rect
        score = 0.0
        for px, py, weight in interest_points:
            if x0 <= px <= x1 and y0 <= py <= y1:
                score += float(weight)
        return score

    def _candidate_rects(width_axes, height_axes, pad=0.02):
        width_axes = float(np.clip(width_axes, 0.10, 0.90))
        height_axes = float(np.clip(height_axes, 0.08, 0.90))
        return {
            'upper right': (1.0 - pad - width_axes, 1.0 - pad - height_axes, 1.0 - pad, 1.0 - pad),
            'upper left': (pad, 1.0 - pad - height_axes, pad + width_axes, 1.0 - pad),
            'lower left': (pad, pad, pad + width_axes, pad + height_axes),
            'lower right': (1.0 - pad - width_axes, pad, 1.0 - pad, pad + height_axes),
        }

    if ann_text:
        fig.text(
            0.015,
            0.87,
            ann_text,
            ha='left',
            va='top',
            fontsize=8.0,
            color='#202020',
            bbox=dict(
                boxstyle='round,pad=0.28',
                facecolor='white',
                edgecolor='#b8b8b8',
                linewidth=0.7,
                alpha=0.92,
            ),
        )

    # --- Legend ---
    legend_defaults = {
        'loc': 'upper left',
        'bbox_to_anchor': (1.02, 1.0),
        'borderaxespad': 0.0,
        'fontsize': 9,
        'framealpha': 0.85,
        'edgecolor': '#cccccc',
        'fancybox': False,
    }
    legend_defaults.update(legend_kwargs)
    if legend_defaults.get('bbox_to_anchor', None) is None:
        legend_defaults.pop('bbox_to_anchor', None)
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles and legend_labels:
        ax.legend(**legend_defaults)

    # --- Tick formatting ---
    ax.tick_params(direction='in', length=3, width=0.6, labelsize=9)

    if ann_text:
        default_adjust = {'left': 0.23, 'right': 0.76, 'top': 0.90, 'bottom': 0.12}
    else:
        default_adjust = {'left': 0.10, 'right': 0.76, 'top': 0.90, 'bottom': 0.12}
    if caption_text:
        default_adjust['bottom'] = max(default_adjust['bottom'], 0.18)
    default_adjust.update(subplot_adjust)
    fig.subplots_adjust(**default_adjust)

    if caption_text:
        if caption_divider:
            fig.add_artist(plt.Line2D(
                [0.05, 0.95], [caption_divider_y, caption_divider_y],
                transform=fig.transFigure, color='#b8b8b8', linewidth=0.8
            ))
        fig.text(
            0.5,
            caption_y,
            str(caption_text),
            ha='center',
            va='bottom',
            fontsize=13 if is_zh else 12,
            color='#202020',
        )

    os.makedirs(osp.dirname(file_name) if osp.dirname(file_name) else '.', exist_ok=True)
    fig.savefig(file_name, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

    file_size = os.path.getsize(file_name)
    size_str = f"{file_size / 1024:.1f}KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f}MB"
    print(f"Trajectory plot saved: {file_name} ({size_str})")



def make_gif(images, file_name, fps=20, quality='high'):
    """
    生成高质量GIF，同时控制文件大小
    
    Args:
        images: 帧列表或numpy数组
        file_name: 输出文件路径
        fps: 帧率
        quality: 'high' (高画质), 'medium' (平衡), 'low' (小体积)
    """
    if PILImage is None:
        return
    
    if isinstance(images, list):
        frames = [np.asarray(img, dtype=np.uint8) for img in images]
    else:
        frames = np.asarray(images, dtype=np.uint8)
    
    if isinstance(frames, np.ndarray) and frames.ndim == 4:
        frames = [frames[i] for i in range(frames.shape[0])]
    
    if len(frames) == 0:
        return
    
    # 根据quality设置参数
    quality_settings = {
        'high': {'max_side': 800, 'colors': 256},
        'medium': {'max_side': 640, 'colors': 192},
        'low': {'max_side': 480, 'colors': 128},
    }
    settings = quality_settings.get(quality, quality_settings['high'])
    
    max_side = getattr(map_config, 'gif_max_side', settings['max_side'])
    num_colors = getattr(map_config, 'gif_colors', settings['colors'])
    
    os.makedirs(osp.dirname(file_name), exist_ok=True)
    duration_ms = int(1000.0 / max(int(fps), 1))
    
    pil_frames = []
    for fr in frames:
        h, w = fr.shape[0], fr.shape[1]
        scale = 1.0
        if max(h, w) > max_side and max_side > 0:
            scale = float(max_side) / float(max(h, w))
        
        img = PILImage.fromarray(fr)
        if scale < 0.999:
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h), resample=PILImage.LANCZOS)
        pil_frames.append(img)
    
    if not pil_frames:
        return
    
    # 量化：关闭抖动避免彩色噪点
    try:
        # 使用MEDIANCUT + 无抖动 = 干净的颜色
        base_img = pil_frames[0].quantize(
            method=PILImage.Quantize.MEDIANCUT, 
            colors=num_colors,
            dither=PILImage.Dither.NONE  # 关闭抖动，避免彩色小点
        )
        final_frames = [base_img]
        for img in pil_frames[1:]:
            q_img = img.quantize(palette=base_img, dither=PILImage.Dither.NONE)
            final_frames.append(q_img)
    except Exception:
        final_frames = pil_frames
    
    # 保存GIF
    final_frames[0].save(
        file_name,
        save_all=True,
        append_images=final_frames[1:],
        optimize=True,
        duration=duration_ms,
        loop=0,
        disposal=2
    )
    
    # 打印文件大小信息
    file_size = os.path.getsize(file_name)
    size_str = f"{file_size / 1024:.1f}KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f}MB"
    print(f"GIF saved: {file_name} (frames={len(frames)}, size={size_str})")


def update_perf(one_ep, perf):
    perf['per_r'].append(one_ep['episode_reward'])
    perf['per_episode_len'].append(one_ep['num_step'])


def build_critic_observation(actor_obs, opponent_obs=None):
    actor_vec = np.asarray(actor_obs, dtype=np.float32).reshape(-1)
    
    if opponent_obs is not None:
        opponent_vec = np.asarray(opponent_obs, dtype=np.float32).reshape(-1)
    else:
        opponent_vec = np.zeros(NetParameters.PRIVILEGED_RAW_LEN, dtype=np.float32)
    
    return np.concatenate([actor_vec, opponent_vec])


def get_ray_temp_dir() -> str:
    """
    获取Ray临时目录路径。
    
    在hp机器上（通过检查主机名或磁盘路径），使用空间充足的位置。
    注意：路径不能包含中文字符，否则Ray会报UnicodeEncodeError
    
    前提条件（hp机器）：需要先执行以下命令创建绑定挂载：
        sudo mkdir -p /mnt/data
        sudo mount --bind "/media/hp/新加卷" /mnt/data
    
    Returns:
        str: Ray临时目录路径，如果是默认位置则返回None
    """
    import socket
    import os
    import getpass
    import tempfile
    
    hostname = socket.gethostname().lower()

    def _try_make_dir(path: str):
        try:
            os.makedirs(path, exist_ok=True)
            test_path = os.path.join(path, ".write_test")
            with open(test_path, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(test_path)
            return path
        except OSError as exc:
            print(f"[Ray] 临时目录不可用，跳过: {path} ({exc})")
            return None

    env_ray_tmp = os.environ.get("TRACKMAKER_RAY_TMPDIR")
    if env_ray_tmp not in (None, ""):
        ray_tmp = _try_make_dir(os.path.abspath(os.path.expanduser(env_ray_tmp)))
        if ray_tmp is not None:
            print(f"[Ray] 使用TRACKMAKER_RAY_TMPDIR: {ray_tmp}")
            return ray_tmp
    
    # 检测是否在hp机器上（通过主机名或特定路径存在）
    is_hp_machine = (
        'hp' in hostname or 
        os.path.exists('/media/hp') or
        'h3cdesk' in hostname.lower()
    )
    
    if is_hp_machine:
        # hp机器上使用绑定挂载的ASCII路径
        # /mnt/data 是 /media/hp/新加卷 的绑定挂载，有6.7T空间
        ray_tmp = _try_make_dir('/mnt/data/ray_tmp')
        if ray_tmp is not None:
            print(f"[Ray] 检测到hp机器，临时目录设置为: {ray_tmp}")
            return ray_tmp

    user = getpass.getuser()
    fallback_tmp = os.path.join(tempfile.gettempdir(), f'trackmaker_ray_tmp_{user}')
    ray_tmp = _try_make_dir(fallback_tmp)
    if ray_tmp is not None:
        print(f"[Ray] 使用fallback临时目录: {ray_tmp}")
        return ray_tmp
    
    # 其他机器使用默认位置
    print("[Ray] 未找到可用自定义临时目录，使用Ray默认临时目录")
    return None
