 
import sys
import os
import numpy as np
import torch

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hrl.hrl_env import HRLEnv
from ppo.util import get_device, print_device_info

def test():
    model_path = "/home/cyq/miniconda3/envs/lnenv/TrackMaker/models/defender_protect2_dense_01-29-10-05/best_model.pth"
    print(f"Testing HRLEnv with model: {model_path}")
    
    # 使用安全的GPU检测
    device = get_device(prefer_gpu=True)
    print_device_info()
    print(f"Using device: {device}")
    
    try:
        env = HRLEnv(protect_model_path=model_path, attacker_strategy='random', device=str(device))
        obs, info = env.reset()
        defender_obs, attacker_obs = obs
        print("Reset successful.")
        print(f"Defender obs shape: {defender_obs.shape}")
        print(f"Attacker obs shape: {attacker_obs.shape}")
        print(f"Current attacker strategy: {env.attacker_policy.strategy}")
        
        for i in range(10):
            action = np.random.uniform(-1, 1, size=(2,))
            obs, reward, term, trunc, info = env.step(action)
            
            weights = info['weights']
            print(f"\nStep {i+1}:")
            print(f"High Level Action: {action}")
            print(f"Weights (Protect/Chase): {weights}")
            print(f"Reward: {reward:.3f}")
            
            if term or trunc:
                print(f"Episode finished. Win: {info.get('win', 'N/A')}")
                break
                
        env.close()
        print("\nTest Finished Successfully.")
        
    except Exception as e:
        print("\nTEST FAILED with error:")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
