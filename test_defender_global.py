"""
测试 DefenderGlobalPolicy 是否能正确导航到目标

场景：
- Attacker 不动（静止）
- Defender 使用 DefenderGlobalPolicy 导航
- 测试 protect 模式（导航到 Target）和 chase 模式（导航到 Attacker）
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from env import TADEnv
from rule_policies.defender_global import DefenderGlobalPolicy

def test_defender_global(skill_mode='protect', max_steps=500, verbose=True):
    """
    测试 DefenderGlobalPolicy
    
    Args:
        skill_mode: 'protect' (导航到Target) 或 'chase' (导航到Attacker)
        max_steps: 最大步数
        verbose: 是否打印详细信息
    """
    print(f"\n{'='*60}")
    print(f"Testing DefenderGlobalPolicy - Mode: {skill_mode}")
    print(f"{'='*60}")
    
    # 创建环境
    env = TADEnv(reward_mode=skill_mode)
    
    # 创建 Defender Global Policy
    policy = DefenderGlobalPolicy(
        env_width=env.width,
        env_height=env.height,
        defender_speed=env.defender_speed,
        defender_max_turn=6.0,
        skill_mode=skill_mode
    )
    
    # 重置环境
    obs, info = env.reset(seed=42)
    defender_obs, attacker_obs = obs
    policy.reset()
    
    # 获取初始状态
    priv_state = env.get_privileged_state()
    
    defender_start = np.array([
        priv_state['defender']['center_x'],
        priv_state['defender']['center_y']
    ])
    attacker_pos = np.array([
        priv_state['attacker']['center_x'],
        priv_state['attacker']['center_y']
    ])
    target_pos = np.array([
        priv_state['target']['center_x'],
        priv_state['target']['center_y']
    ])
    
    # 根据模式确定目标
    if skill_mode == 'protect':
        goal_pos = target_pos
        goal_name = "Target"
    else:
        goal_pos = attacker_pos
        goal_name = "Attacker"
    
    initial_dist = np.linalg.norm(defender_start - goal_pos)
    
    if verbose:
        print(f"\nInitial State:")
        print(f"  Defender: ({defender_start[0]:.1f}, {defender_start[1]:.1f})")
        print(f"  Attacker: ({attacker_pos[0]:.1f}, {attacker_pos[1]:.1f})")
        print(f"  Target:   ({target_pos[0]:.1f}, {target_pos[1]:.1f})")
        print(f"  Goal ({goal_name}): ({goal_pos[0]:.1f}, {goal_pos[1]:.1f})")
        print(f"  Initial distance to goal: {initial_dist:.1f}")
        print()
    
    # 运行模拟
    reached = False
    arrival_threshold = 20.0  # 到达阈值
    
    for step in range(max_steps):
        # 获取特权状态
        priv_state = env.get_privileged_state()
        
        # Defender 使用 global policy
        defender_action = policy.get_action(defender_obs, priv_state)
        
        # Attacker 不动（静止动作）
        attacker_action = np.array([0.0, -1.0], dtype=np.float32)  # 不转向，速度为0
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(defender_action, attacker_action)
        defender_obs, attacker_obs = obs
        
        # 计算当前距离
        priv_state = env.get_privileged_state()
        defender_pos = np.array([
            priv_state['defender']['center_x'],
            priv_state['defender']['center_y']
        ])
        
        # chase模式下attacker可能在动，更新目标位置
        if skill_mode == 'chase':
            goal_pos = np.array([
                priv_state['attacker']['center_x'],
                priv_state['attacker']['center_y']
            ])
        
        current_dist = np.linalg.norm(defender_pos - goal_pos)
        
        if verbose and step % 50 == 0:
            print(f"  Step {step:3d}: Defender at ({defender_pos[0]:.1f}, {defender_pos[1]:.1f}), "
                  f"dist to {goal_name}: {current_dist:.1f}")
        
        # 检查是否到达
        if current_dist < arrival_threshold:
            reached = True
            if verbose:
                print(f"\n  ✓ Reached {goal_name} at step {step}!")
                print(f"    Final distance: {current_dist:.1f}")
            break
        
        if terminated or truncated:
            if verbose:
                print(f"\n  Episode ended at step {step}")
                print(f"    terminated={terminated}, truncated={truncated}")
            break
    
    # 最终状态
    final_dist = current_dist
    
    if verbose:
        print(f"\nFinal State:")
        print(f"  Defender: ({defender_pos[0]:.1f}, {defender_pos[1]:.1f})")
        print(f"  Distance to {goal_name}: {final_dist:.1f}")
        print(f"  Distance reduced: {initial_dist - final_dist:.1f} ({(initial_dist - final_dist) / initial_dist * 100:.1f}%)")
        
        if reached:
            print(f"\n  ✓ SUCCESS: Defender reached {goal_name}!")
        else:
            print(f"\n  ✗ FAILED: Defender did not reach {goal_name} within {max_steps} steps")
    
    env.close()
    
    return reached, initial_dist, final_dist


def main():
    print("="*60)
    print("DefenderGlobalPolicy Navigation Test")
    print("="*60)
    
    # 测试 protect 模式（导航到 Target）
    success_protect, init_dist_p, final_dist_p = test_defender_global(
        skill_mode='protect', 
        max_steps=500,
        verbose=True
    )
    
    # 测试 chase 模式（导航到 Attacker）
    success_chase, init_dist_c, final_dist_c = test_defender_global(
        skill_mode='chase', 
        max_steps=500,
        verbose=True
    )
    
    # 总结
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"  Protect mode (to Target):   {'✓ SUCCESS' if success_protect else '✗ FAILED'}")
    print(f"  Chase mode (to Attacker):   {'✓ SUCCESS' if success_chase else '✗ FAILED'}")
    
    if success_protect and success_chase:
        print("\n  All tests passed! DefenderGlobalPolicy works correctly.")
    else:
        print("\n  Some tests failed. Check the policy implementation.")


if __name__ == '__main__':
    main()
