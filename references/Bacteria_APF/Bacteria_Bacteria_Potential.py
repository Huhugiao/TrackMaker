import numpy as np


def Bacteria_Bacteria_Potential(par):
    # 初始化数组（使用 NumPy）
    par["J_ObstBT"] = np.zeros(par["bacteria_no"])
    par["J_GoalBT"] = np.zeros(par["bacteria_no"])
    par["BDTT"] = np.zeros(par["bacteria_no"])

    # 遍历所有细菌
    for i in range(par["bacteria_no"]):  # Python 索引从 0 开始
        # 计算细菌到目标的欧氏距离
        dx = par["bx"][i] - par["target_coordinates"][0]
        dy = par["bx"][i] - par["target_coordinates"][1]
        par["BDTT"][i] = np.sqrt(dx**2 + dy**2)

        # 遍历所有检测到的障碍物
        for j in range(par['det']):
            # 计算细菌到障碍物的欧氏距离
            obst_x = par["obstacles_coordinates_detected"][j, 0]
            obst_y = par["obstacles_coordinates_detected"][j, 1]
            dx_obst = par["bx"][i] - obst_x
            dy_obst = par["by"][i] - obst_y
            dist = np.sqrt(dx_obst**2 + dy_obst**2)

            # 根据距离计算势能
            if (
                dist >= par["potential_lower_distance_limit"]
                and dist <= par["potential_upper_distance_limit"]
            ):
                exponent = -par["obstacle_bacteria"][1] * (dx_obst**2 + dy_obst**2)
                pot_val = par["obstacle_bacteria"][0] * np.exp(exponent)
            elif dist < par["potential_lower_distance_limit"]:
                pot_val = np.inf
            else:
                pot_val = 0
            par["J_ObstBT"][i] += pot_val  # 累加障碍物势能

        # 计算目标势能
        exponent_goal = -par["target"][1] * (dx**2 + dy**2)
        par["J_GoalBT"][i] = -par["target"][0] * np.exp(exponent_goal)

        # 总势能 = 目标势能 + 障碍物势能
        par["J_BT"] = par["J_GoalBT"] + par["J_ObstBT"]  # 直接向量化操作（无需循环）

    return par
