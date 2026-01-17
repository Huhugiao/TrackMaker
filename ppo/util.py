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
