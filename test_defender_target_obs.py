#!/usr/bin/env python
"""
测试脚本：验证 defender 是否能正确看到 target

此脚本会：
1. 创建环境并重置
2. 打印 defender 观测中关于 target 的信息
3. 运行几步并验证 target 信息是否正确更新
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import math
from env import TADEnv
import map_config

def main():
    print("=" * 60)
    print("Defender 观测 Target 验证测试")
    print("=" * 60)
    
    # 创建环境
    env = TADEnv(reward_mode='protect')
    obs, _ = env.reset(seed=42)
    
    defender_obs, attacker_obs = obs
    
    print(f"\n环境尺寸: {env.width} x {env.height}")
    print(f"Target 位置: ({env.target['x']:.1f}, {env.target['y']:.1f})")
    print(f"Defender 位置: ({env.defender['x']:.1f}, {env.defender['y']:.1f})")
    print(f"Attacker 位置: ({env.attacker['x']:.1f}, {env.attacker['y']:.1f})")
    
    # 计算真实距离
    def_cx = env.defender['x'] + map_config.pixel_size * 0.5
    def_cy = env.defender['y'] + map_config.pixel_size * 0.5
    tgt_cx = env.target['x'] + map_config.pixel_size * 0.5
    tgt_cy = env.target['y'] + map_config.pixel_size * 0.5
    
    real_dist = math.hypot(tgt_cx - def_cx, tgt_cy - def_cy)
    real_angle = math.degrees(math.atan2(tgt_cy - def_cy, tgt_cx - def_cx))
    relative_angle = real_angle - env.defender['theta']
    
    # 规范化角度到 [-180, 180]
    while relative_angle > 180:
        relative_angle -= 360
    while relative_angle < -180:
        relative_angle += 360
    
    print(f"\n--- 真实值 ---")
    print(f"Defender -> Target 真实距离: {real_dist:.2f}")
    print(f"Defender -> Target 真实方位角: {relative_angle:.2f}°")
    
    print(f"\n--- Defender 观测 (72维) ---")
    print(f"obs[0] (attacker distance, normalized): {defender_obs[0]:.4f}")
    print(f"obs[1] (attacker bearing, normalized): {defender_obs[1]:.4f}")
    print(f"obs[2] (attacker fov_edge): {defender_obs[2]:.4f}")
    print(f"obs[3] (in_fov): {defender_obs[3]:.4f}")
    print(f"obs[4] (occluded): {defender_obs[4]:.4f}")
    print(f"obs[5] (unobserved_time): {defender_obs[5]:.4f}")
    print(f"obs[6:70] (radar, 64维): min={defender_obs[6:70].min():.4f}, max={defender_obs[6:70].max():.4f}")
    print(f"obs[70] (target distance, normalized): {defender_obs[70]:.4f}")
    print(f"obs[71] (target bearing, normalized): {defender_obs[71]:.4f}")
    
    # 反算 target 距离和方位
    map_diagonal = math.hypot(env.width, env.height)
    decoded_dist = ((defender_obs[70] + 1.0) / 2.0) * map_diagonal
    decoded_bearing = defender_obs[71] * 180.0
    
    print(f"\n--- 反算验证 ---")
    print(f"从 obs[70] 反算距离: {decoded_dist:.2f} (真实: {real_dist:.2f})")
    print(f"从 obs[71] 反算方位角: {decoded_bearing:.2f}° (真实: {relative_angle:.2f}°)")
    
    dist_error = abs(decoded_dist - real_dist)
    bearing_error = abs(decoded_bearing - relative_angle)
    
    print(f"\n--- 误差分析 ---")
    print(f"距离误差: {dist_error:.2f}")
    print(f"方位角误差: {bearing_error:.2f}°")
    
    if dist_error < 1.0 and bearing_error < 1.0:
        print("\n✅ Defender 可以正确看到 Target!")
    else:
        print("\n❌ Defender 观测 Target 存在问题!")
    
    # 测试几步移动
    print("\n" + "=" * 60)
    print("运行 5 步测试动态更新...")
    print("=" * 60)
    
    for step in range(5):
        # 随机动作
        action = np.array([0.5, 0.5])  # 向前走
        attacker_action = np.array([0.0, 0.5])
        
        obs, reward, terminated, truncated, info = env.step(action, attacker_action)
        defender_obs, attacker_obs = obs
        
        # 重新计算真实值
        def_cx = env.defender['x'] + map_config.pixel_size * 0.5
        def_cy = env.defender['y'] + map_config.pixel_size * 0.5
        tgt_cx = env.target['x'] + map_config.pixel_size * 0.5
        tgt_cy = env.target['y'] + map_config.pixel_size * 0.5
        
        real_dist = math.hypot(tgt_cx - def_cx, tgt_cy - def_cy)
        decoded_dist = ((defender_obs[70] + 1.0) / 2.0) * map_diagonal
        
        print(f"Step {step+1}: 真实距离={real_dist:.1f}, 观测距离={decoded_dist:.1f}, reward={reward:.3f}")
        
        if terminated or truncated:
            print(f"  Episode 结束: {info.get('reason', 'unknown')}")
            break
    
    print("\n测试完成!")

if __name__ == '__main__':
    main()
