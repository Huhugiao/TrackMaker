import matplotlib.pyplot as plt
import numpy as np


def Bacteria_Plot_Robot(par):
    # 绘制机器人的位置，使用黑色的点标记，标记大小为 15
    robots_plot = par['ax'].plot(
        par["robot_coordinates"][0],
        par["robot_coordinates"][1],
        ".",
        color="blue",
        markersize=15,
    )
    par["robot_plots"] = robots_plot
    # 初始化细菌点的 x 和 y 坐标数组
    par["bx"] = np.zeros(par["bacteria_no"])
    par["by"] = np.zeros(par["bacteria_no"])

    # 计算每个细菌点的坐标
    for i in range(par["bacteria_no"]):
        # 根据机器人坐标、步长和细菌点角度计算细菌点的 x 坐标
        par["bx"][i] = par["robot_coordinates"][0] + (
            par["step_size"] * np.cos(np.radians(par["bacteria_angles"][i]))
        )
        # 根据机器人坐标、步长和细菌点角度计算细菌点的 y 坐标
        par["by"][i] = par["robot_coordinates"][1] + (
            par["step_size"] * np.sin(np.radians(par["bacteria_angles"][i]))
        )

    # 绘制细菌点，使用指定颜色的点标记
    # bacteria_plot = par['ax'].plot(par["bx"], par["by"], ".", color=[0.9290, 0.6940, 0.1250])

    return par
