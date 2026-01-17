import os
import os.path as osp
import random
import numpy as np
import torch
from PIL import Image as PILImage
import map_config

from ppo.alg_parameters import NetParameters

def set_global_seeds(i):
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    np.random.seed(i)
    random.seed(i)
    torch.backends.cudnn.deterministic = True

def write_to_tensorboard(global_summary, step, performance_dict=None,
                  
                         mb_loss=None, imitation_loss=None, q_loss=None,
                         evaluate=True, greedy=True):
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
        key_map = {'per_r': 'Reward', 'per_episode_len': 'Episode_Length', 'win': 'Win_Rate'}
        for k, v in performance_dict.items():
            vals = v if isinstance(v, (list, tuple)) else [v]
            if not vals:
                continue

            if isinstance(vals[0], (list, tuple, np.ndarray)):
                val = float(np.nanmean(vals, axis=0))
            else:
                val = float(np.nanmean(vals))

            if val is not None:
                name = key_map.get(k, k)
                global_summary.add_scalar(f'{prefix}/{name}', val, step)
                if k in ['per_r', 'per_episode_len'] and len(vals) > 1:
                    std_val = float(np.nanstd(vals))
                    global_summary.add_scalar(f'{prefix}/{name}_Std', std_val, step)

    if mb_loss:
        loss_vals = np.nanmean(np.asarray(mb_loss, dtype=np.float32), axis=0)
        mapping = {0: 'Total', 1: 'Policy', 2: 'Entropy', 3: 'Value',
                   4: 'Adv_Std', 5: 'Approx_KL', 6: 'GRU_Loss', 7: 'Clip_Frac', 8: 'Grad_Norm', 9: 'Adv_Mean'}

        for idx, val in enumerate(loss_vals):
            if idx in mapping:
                global_summary.add_scalar(f'Loss/{mapping[idx]}', float(val), step)

    global_summary.flush()

def make_gif(images, file_name, fps=20):
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

    max_side = getattr(map_config, 'gif_max_side', 640)
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

    base_img = pil_frames[0].quantize(method=PILImage.ADAPTIVE, colors=256, dither=PILImage.NONE)
    final_frames = [base_img]
    for img in pil_frames[1:]:
        q_img = img.quantize(palette=base_img, dither=PILImage.NONE)
        final_frames.append(q_img)

    final_frames[0].save(file_name, save_all=True, append_images=final_frames[1:], optimize=True, duration=duration_ms, loop=0)
    print(f"GIF saved: {file_name} (frames={len(frames)})")

def build_critic_observation(actor_obs):
    return np.asarray(actor_obs, dtype=np.float32).reshape(-1)
