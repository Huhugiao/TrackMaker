#!/usr/bin/env python
"""
测试所有Attacker策略
"""
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import TADEnv
from rule_policies.attacker_global import AttackerGlobalPolicy, ALL_STRATEGIES, STRATEGY_CONFIGS

def test_strategy(strategy_name, num_steps=100, render=False):
    """测试单个策略"""
    print(f"\n{'='*60}")
    print(f"Testing Strategy: {strategy_name}")
    print(f"Config: {STRATEGY_CONFIGS.get(strategy_name, {})}")
    print(f"{'='*60}")
    
    try:
        env = TADEnv(reward_mode='standard')
        policy = AttackerGlobalPolicy(strategy=strategy_name)
        
        obs, _ = env.reset()
        defender_obs, attacker_obs = obs
        policy.reset()
        
        total_reward = 0
        positions = []
        
        for step in range(num_steps):
            action, info = policy.get_action_with_info(attacker_obs)
            
            # 记录位置
            att_pos = policy.denormalize_pos(attacker_obs[0], attacker_obs[1])
            positions.append(att_pos.copy())
            
            # 环境step（defender随机动作）
            defender_action = np.random.uniform(-1, 1, size=(2,))
            obs, reward, term, trunc, env_info = env.step(
                action=defender_action, 
                attacker_action=action
            )
            defender_obs, attacker_obs = obs
            total_reward += reward
            
            if term or trunc:
                print(f"Episode ended at step {step+1}: {env_info.get('reason', 'unknown')}")
                break
        
        # 统计移动
        positions = np.array(positions)
        if len(positions) > 1:
            total_distance = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            avg_speed = total_distance / len(positions)
        else:
            avg_speed = 0
            
        print(f"✅ Strategy '{strategy_name}' works!")
        print(f"   - Steps completed: {step+1}")
        print(f"   - Average speed: {avg_speed:.2f} px/step")
        print(f"   - Final position: ({positions[-1][0]:.1f}, {positions[-1][1]:.1f})")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Strategy '{strategy_name}' FAILED!")
        print(f"   Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("ATTACKER STRATEGY TEST SUITE")
    print(f"Testing {len(ALL_STRATEGIES)} strategies")
    print("="*60)
    
    results = {}
    for strategy in ALL_STRATEGIES:
        results[strategy] = test_strategy(strategy, num_steps=200)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    failed = len(results) - passed
    
    for strategy, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {strategy:20s} {status}")
    
    print(f"\nTotal: {passed}/{len(results)} passed")
    
    if failed > 0:
        sys.exit(1)
    
if __name__ == "__main__":
    main()
