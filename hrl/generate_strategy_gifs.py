#!/usr/bin/env python
"""
生成每个Attacker策略与Defender模型对抗的GIF
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import TADEnv
from ppo.nets import DefenderNetMLP
from ppo.util import build_critic_observation, make_gif
from rule_policies.attacker_global import AttackerGlobalPolicy, ALL_STRATEGIES

def generate_strategy_gif(
    strategy_name: str,
    model_path: str,
    output_dir: str,
    max_steps: int = 500,
    device: str = 'cpu'
):
    """为单个策略生成GIF"""
    print(f"\n{'='*50}")
    print(f"Generating GIF for: {strategy_name}")
    print(f"{'='*50}")
    
    # 加载Defender模型
    defender_net = DefenderNetMLP().to(device)
    defender_net.eval()
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        defender_net.load_state_dict(checkpoint['model'])
    else:
        defender_net.load_state_dict(checkpoint)
    
    # 创建环境和Attacker策略
    env = TADEnv(reward_mode='protect2')
    attacker_policy = AttackerGlobalPolicy(strategy=strategy_name)
    
    # Reset
    obs, _ = env.reset(seed=42)  # 固定种子保证可复现
    defender_obs, attacker_obs = obs
    attacker_policy.reset()
    
    frames = []
    episode_reward = 0
    result = "timeout"
    
    for step in range(max_steps):
        # 渲染帧
        frame = env.render(mode='rgb_array')
        if frame is not None:
            frames.append(frame)
        
        # Defender动作（使用模型）
        with torch.no_grad():
            obs_tensor = torch.as_tensor(defender_obs, dtype=torch.float32, device=device).unsqueeze(0)
            critic_obs = build_critic_observation(defender_obs, attacker_obs)
            critic_tensor = torch.as_tensor(critic_obs, dtype=torch.float32, device=device).unsqueeze(0)
            mean, _, _ = defender_net(obs_tensor, critic_tensor)
            defender_action = torch.tanh(mean).cpu().numpy()[0]
        
        # Attacker动作
        attacker_action, _ = attacker_policy.get_action_with_info(attacker_obs)
        
        # Step
        obs, reward, term, trunc, info = env.step(
            action=defender_action,
            attacker_action=attacker_action
        )
        defender_obs, attacker_obs = obs
        episode_reward += reward
        
        if term or trunc:
            result = info.get('reason', 'unknown')
            break
    
    # 保存GIF
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, f"vs_{strategy_name}.gif")
    make_gif(frames, gif_path, fps=20, quality='high')
    
    win = info.get('win', False) if 'info' in dir() else False
    print(f"  Result: {result}")
    print(f"  Steps: {step + 1}")
    print(f"  Defender Win: {win}")
    print(f"  Saved: {gif_path}")
    
    env.close()
    return {
        'strategy': strategy_name,
        'result': result,
        'steps': step + 1,
        'win': win,
        'gif_path': gif_path
    }

def main():
    model_path = "/home/ace/miniconda3/envs/lnenv/trackmaker/models/defender_protect_dense_01-28-11-28/protect_rl_01-28-11-28/models/best_model.pth"
    output_dir = "/home/ace/miniconda3/envs/lnenv/trackmaker/models/attacker_strategy_gifs"
    
    print("="*60)
    print("ATTACKER STRATEGY GIF GENERATOR")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Strategies: {len(ALL_STRATEGIES)}")
    print("="*60)
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)
    
    results = []
    for strategy in ALL_STRATEGIES:
        result = generate_strategy_gif(
            strategy_name=strategy,
            model_path=model_path,
            output_dir=output_dir,
            max_steps=500
        )
        results.append(result)
    
    # 汇总
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    wins = sum(1 for r in results if r['win'])
    
    for r in results:
        status = "✅ DEF WIN" if r['win'] else "❌ ATK WIN"
        print(f"  {r['strategy']:20s} {status:12s} {r['result']:30s} ({r['steps']} steps)")
    
    print(f"\nDefender Win Rate: {wins}/{len(results)} ({100*wins/len(results):.1f}%)")
    print(f"\nGIFs saved to: {output_dir}")

if __name__ == "__main__":
    main()
