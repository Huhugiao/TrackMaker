import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def Bacteria_Plot_Environment(par):
    # 初始化画布大小
    fig, ax = plt.subplots(figsize=(8, 8))
    # 设置坐标轴范围
    map_size = [0, 30, 0, 30]
    ax.set_xlim(par['map_size'][0], par['map_size'][1])
    ax.set_ylim(par['map_size'][2], par['map_size'][3])
    # 开启交互模式
    plt.ion()
    # 设置坐标轴范围，与 Matlab 中的 axis 函数功能类似
    plt.axis(par['map_size'])
    # 绘制目标点，使用红色的 'X' 标记，标记大小为 15
    target_plot = ax.plot(par['target_coordinates'][0], par['target_coordinates'][1], 'X', color='red', markersize=15)
    obstacles_plots = []
    # 绘制障碍物
    # 遍历所有障碍物的坐标
    for obs in par["obstacles_coordinates"]:
        circle = patches.Circle((obs[0], obs[1]), obs[2], edgecolor='r', facecolor='none')
        # 将圆添加到 Axes 对象上
        obstacles_plots.append(circle)
        ax.add_patch(circle)
    par["obstacles_plot"] = obstacles_plots
    par["target_plot"] = target_plot
    par['ax'] = ax
    par['fig'] = fig
    # 显示图形
    plt.show()
    plt.savefig("Env.jpg")
    return
