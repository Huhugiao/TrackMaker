"""
TAD PPO 工具函数
"""

import os
import os.path as osp
import random
import numpy as np
import torch
from typing import Dict, List, Optional
import map_config

from PIL import Image as PILImage

from ppo.alg_parameters import SetupParameters, TrainingParameters, NetParameters
from map_config import EnvParameters


def get_device(prefer_gpu: bool = True, gpu_id: int = None) -> torch.device:
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
    
    if not prefer_gpu:
        return torch.device('cpu')
    
    # 检测是否在Ray worker中运行
    # Ray会设置CUDA_VISIBLE_DEVICES来隔离GPU，此时worker只能看到分配给它的那个GPU
    in_ray_worker = 'RAY_WORKER_PID' in os.environ or 'CUDA_VISIBLE_DEVICES' in os.environ
    
    # 获取GPU ID
    if gpu_id is None:
        from ppo.alg_parameters import SetupParameters
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
    return get_device(prefer_gpu=True).type == 'cuda'


def get_num_gpus() -> int:
    """
    安全地获取可用的GPU数量。
    
    此函数可以在没有安装CUDA的机器上安全运行。
    
    Returns:
        int: 可用的GPU数量（如果CUDA不可用则返回0）
    """
    try:
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.device_count()
    except Exception:
        return 0


def print_device_info():
    """打印设备信息，用于调试"""
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


def get_adjusted_n_envs(base_n_envs: int, ram_threshold_gb: float = 20.0, multiplier: int = 6) -> int:
    """
    根据空闲RAM大小动态调整并行环境数量。
    
    如果空闲RAM大于阈值，则将环境数量乘以倍数。
    
    Args:
        base_n_envs: 基础并行环境数量
        ram_threshold_gb: RAM阈值（GB），默认20GB
        multiplier: 乘数，默认6倍
        
    Returns:
        int: 调整后的并行环境数量
    """
    free_ram = get_free_ram_gb()
    
    if free_ram > ram_threshold_gb:
        adjusted = base_n_envs * multiplier
        print(f"[RAM检测] 空闲RAM: {free_ram:.1f}GB > {ram_threshold_gb}GB 阈值")
        print(f"[RAM检测] 并行环境数量: {base_n_envs} -> {adjusted} (x{multiplier})")
        return adjusted
    else:
        print(f"[RAM检测] 空闲RAM: {free_ram:.1f}GB <= {ram_threshold_gb}GB 阈值")
        print(f"[RAM检测] 保持并行环境数量: {base_n_envs}")
        return base_n_envs


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
            'win': 'Win_Rate'
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
    
    hostname = socket.gethostname().lower()
    
    # 检测是否在hp机器上（通过主机名或特定路径存在）
    is_hp_machine = (
        'hp' in hostname or 
        os.path.exists('/media/hp') or
        'h3cdesk' in hostname.lower()
    )
    
    if is_hp_machine:
        # hp机器上使用绑定挂载的ASCII路径
        # /mnt/data 是 /media/hp/新加卷 的绑定挂载，有6.7T空间
        ray_tmp = '/mnt/data/ray_tmp'
        os.makedirs(ray_tmp, exist_ok=True)
        print(f"[Ray] 检测到hp机器，临时目录设置为: {ray_tmp}")
        return ray_tmp
    
    # 其他机器使用默认位置
    return None
