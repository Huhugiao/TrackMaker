import numpy as np


def Bacteria_Robot_Potential(par):
    # 初始化障碍物对机器人的总势能为 0
    par["J_obstRT"] = 0
    # 遍历所有检测到的障碍物坐标
    for j in range(par["obstacles_coordinates_detected"].shape[0]):
        # 计算机器人与当前障碍物的距离
        dist = np.sqrt(
            (par["robot_coordinates"][0] - par["obstacles_coordinates_detected"][j, 0])
            ** 2
            + (
                par["robot_coordinates"][1]
                - par["obstacles_coordinates_detected"][j, 1]
            )
            ** 2
        )
        # 根据距离计算障碍物对机器人的势能
        if (
            dist >= par["potential_lower_distance_limit"]
            and dist <= par["potential_upper_distance_limit"]
        ):
            # 当距离在有效范围内，使用指数函数计算势能
            pot_val = par["obstacle_robot"][0] * np.exp(
                -par["obstacle_robot"][1]
                * (
                    (
                        par["robot_coordinates"][0]
                        - par["obstacles_coordinates_detected"][j, 0]
                    )
                    ** 2
                    + (
                        par["robot_coordinates"][1]
                        - par["obstacles_coordinates_detected"][j, 1]
                    )
                    ** 2
                )
            )
        elif dist < par["potential_lower_distance_limit"]:
            # 当距离小于下限，势能设为无穷大
            pot_val = np.inf
        else:
            # 当距离大于上限，势能设为 0
            pot_val = 0
        # 累加障碍物对机器人的势能
        par["J_obstRT"] += pot_val

    # 计算机器人到目标的距离
    par["RDTT"] = np.sqrt(
        (par["robot_coordinates"][0] - par["target_coordinates"][0]) ** 2
        + (par["robot_coordinates"][1] - par["target_coordinates"][1]) ** 2
    )

    # 计算目标对机器人的势能
    par["J_GoalRT"] = -par["target"][0] * np.exp(
        -par["target"][1]
        * (
            (par["robot_coordinates"][0] - par["target_coordinates"][0]) ** 2
            + (par["robot_coordinates"][1] - par["target_coordinates"][1]) ** 2
        )
    )

    # 计算机器人受到的总势能
    par["J_RT"] = par["J_GoalRT"] + par["J_obstRT"]

    return par
