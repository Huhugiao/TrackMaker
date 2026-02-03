#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
科学绘图脚本：用于绘制TensorBoard tfevents文件中的训练曲线
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
使用方法：
  1. 修改下方【用户配置区】的内容
  2. 运行脚本：python plot_tfevents.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                              ║
# ║                       ★★★  用 户 配 置 区  ★★★                            ║
# ║                                                                              ║
# ║                     直接修改下面的内容，然后运行脚本                           ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  1. 实验列表配置                                                              │
# │     格式: ("显示名称", "tfevents文件夹路径")                                   │
# │     路径支持相对路径(相对于本脚本)或绝对路径                                    │
# └──────────────────────────────────────────────────────────────────────────────┘
EXPERIMENTS = [
    # ↓↓↓ 在这里添加你的实验，格式：("图例名称", "文件夹路径") ↓↓↓
    
    # ("PPO_only", "models/baseline_01-30-22-19/summary"),
    # ("HRL", "models/hrl_01-30-17-06/summary"),
    ("Chase", "models/defender_chase_dense_02-02-11-00"),
    # ("Protect_stage1", "models/defender_protect_dense_01-28-11-28/protect_rl_01-28-11-28"),
    # ("Protect_stage2", "models/defender_protect2_dense_01-29-10-05"),
    
    # ↑↑↑ 添加更多实验只需复制上面一行并修改 ↑↑↑
]

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  2. 要绘制的指标                                                              │
# │     第一次运行时可以留空 []，脚本会打印出所有可用指标供你选择                    │
# └──────────────────────────────────────────────────────────────────────────────┘
METRICS = [
    # ↓↓↓ 在这里填入要绘制的指标名称 ↓↓↓
    
    "Train/Reward",
    # "Train/Episode_Length",
    # "Train/Win_Rate",
    # "Eval/Reward",
    # "Loss/Total",
    # "Loss/Policy",
    # "Loss/Value",

]

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  3. 绘图参数                                                                  │
# └──────────────────────────────────────────────────────────────────────────────┘
SMOOTH = 0.6                # 曲线平滑度 (0=不平滑, 1=最平滑, 推荐0.6)
SHOW_RAW = True             # 是否显示原始曲线（淡色背景线）
SHOW_STD = False            # 是否显示标准差区间

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  4. 横轴范围设置                                                              │
# │     设置为 None 表示自动范围，或设置具体数值限制范围                            │
# └──────────────────────────────────────────────────────────────────────────────┘
X_MIN = None                # 横轴最小值 (None=自动, 或填数字如 0)
X_MAX = 2e7                # 横轴最大值 (None=自动, 或填数字如 1e8)

# 示例：
# X_MIN = 0                 # 从0开始
# X_MAX = 5e7               # 到5000万步结束
# X_MIN = 1e7               # 从1000万步开始
# X_MAX = None              # 到数据结束

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  5. 输出设置                                                                  │
# └──────────────────────────────────────────────────────────────────────────────┘
OUTPUT_DIR = "plots"                    # 图片保存目录
OUTPUT_FILE = "training_curves.png"     # 输出文件名
DPI = 300                               # 图片分辨率

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │  6. 图表布局                                                                  │
# └──────────────────────────────────────────────────────────────────────────────┘
SUBPLOT_WIDTH = 8           # 每个子图宽度（英寸）
SUBPLOT_HEIGHT = 5          # 每个子图高度（英寸）
MAX_COLS = 3                # 每行最多几个子图


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           配 置 区 结 束                                      ║
# ║                        ↓↓↓ 以下代码无需修改 ↓↓↓                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# 配色方案
COLORS = [
    # '#1f77b4',  # 蓝
    # '#ff7f0e',  # 橙
    # '#2ca02c',  # 绿
    # '#d62728',  # 红
    '#9467bd',  # 紫
    # '#8c564b',  # 棕
    # '#e377c2',  # 粉
    # '#7f7f7f',  # 灰
    # '#bcbd22',  # 黄绿
    # '#17becf',  # 青
]


def setup_style():
    """设置绘图风格"""
    plt.style.use('seaborn-v0_8-whitegrid')
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'legend.fontsize': 11,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'lines.linewidth': 2,
        'figure.dpi': 150,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
    })


def load_data(log_dir):
    """加载tfevents数据"""
    ea = EventAccumulator(log_dir)
    ea.Reload()
    data = {}
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        data[tag] = (steps, values)
    return data


def smooth(values, weight):
    """指数移动平均平滑"""
    if weight <= 0:
        return values
    result = []
    last = values[0]
    for v in values:
        s = last * weight + (1 - weight) * v
        result.append(s)
        last = s
    return np.array(result)


def filter_by_range(steps, values, x_min, x_max):
    """根据横轴范围过滤数据"""
    mask = np.ones(len(steps), dtype=bool)
    if x_min is not None:
        mask &= (steps >= x_min)
    if x_max is not None:
        mask &= (steps <= x_max)
    return steps[mask], values[mask]


def plot_metric(ax, all_data, metric):
    """绑制单个指标"""
    for i, (name, data) in enumerate(all_data.items()):
        if metric not in data:
            continue
        
        steps, values = data[metric]
        
        # 过滤横轴范围
        steps, values = filter_by_range(steps, values, X_MIN, X_MAX)
        if len(steps) == 0:
            continue
        
        color = COLORS[i % len(COLORS)]
        
        # 原始曲线（淡色）
        if SHOW_RAW:
            ax.plot(steps, values, color=color, alpha=0.2, linewidth=0.8)
        
        # 平滑曲线
        smoothed = smooth(values, SMOOTH)
        ax.plot(steps, smoothed, color=color, linewidth=2, label=name)
        
        # 标准差区间
        std_metric = metric + '_Std'
        if SHOW_STD and std_metric in data:
            std_steps, std_vals = data[std_metric]
            std_steps, std_vals = filter_by_range(std_steps, std_vals, X_MIN, X_MAX)
            if len(std_steps) > 0:
                std_smooth = smooth(std_vals, SMOOTH)
                ax.fill_between(steps, smoothed - std_smooth, smoothed + std_smooth,
                              color=color, alpha=0.15)
    
    # 设置横轴范围
    if X_MIN is not None or X_MAX is not None:
        ax.set_xlim(X_MIN, X_MAX)
    
    ax.set_xlabel('Steps')
    ax.set_ylabel(metric.split('/')[-1])
    ax.set_title(metric.replace('/', ' / '))
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


def main():
    print("=" * 70)
    print("               TensorBoard 训练曲线绑图工具")
    print("=" * 70)
    
    setup_style()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # ========== 加载数据 ==========
    print("\n📂 加载实验数据...")
    all_data = {}
    all_metrics = set()
    
    for name, path in EXPERIMENTS:
        if not os.path.isabs(path):
            path = os.path.join(script_dir, path)
        
        try:
            data = load_data(path)
            if data:
                all_data[name] = data
                all_metrics.update(data.keys())
                print(f"   ✓ {name}: {len(data)} 个指标")
            else:
                print(f"   ✗ {name}: 无数据")
        except Exception as e:
            print(f"   ✗ {name}: {e}")
    
    if not all_data:
        print("\n❌ 错误：未能加载任何数据，请检查路径配置！")
        return
    
    # ========== 显示可用指标 ==========
    print("\n📋 可用指标列表（可复制到 METRICS 配置中）:")
    print("-" * 50)
    for m in sorted(all_metrics):
        print(f'    "{m}",')
    print("-" * 50)
    
    # 显示横轴范围设置
    if X_MIN is not None or X_MAX is not None:
        print(f"\n📏 横轴范围: {X_MIN if X_MIN else '自动'} ~ {X_MAX if X_MAX else '自动'}")
    
    # ========== 检查指标 ==========
    if not METRICS:
        print("\n💡 提示：METRICS 列表为空")
        print("   请从上方列表中选择指标，复制到脚本的 METRICS 配置中")
        return
    
    valid_metrics = []
    for m in METRICS:
        if any(m in d for d in all_data.values()):
            valid_metrics.append(m)
        else:
            print(f"   ⚠ 跳过不存在的指标: {m}")
    
    if not valid_metrics:
        print("\n❌ 错误：没有有效的指标！")
        return
    
    # ========== 绑图 ==========
    print(f"\n🎨 正在绑制 {len(valid_metrics)} 个指标...")
    
    n = len(valid_metrics)
    cols = min(MAX_COLS, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(SUBPLOT_WIDTH * cols, SUBPLOT_HEIGHT * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, metric in enumerate(valid_metrics):
        plot_metric(axes[i], all_data, metric)
        print(f"   ✓ {metric}")
    
    for i in range(n, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # ========== 保存 ==========
    out_dir = OUTPUT_DIR if os.path.isabs(OUTPUT_DIR) else os.path.join(script_dir, OUTPUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, OUTPUT_FILE)
    
    plt.savefig(out_path, dpi=DPI, bbox_inches='tight')
    print(f"\n💾 图片已保存: {out_path}")
    print("=" * 70)
    
    plt.show()


if __name__ == '__main__':
    main()
