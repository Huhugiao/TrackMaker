import numpy as np


def Bacteria_Initialize(obstacles_coordinates):
    # 初始化一个字典 par 用于存储所有参数
    par = {}
    # 设置环境的大小 [xmin, xmax, ymin, ymax]
    par['map_size'] = [0, 50, 0, 50]
    # 机器人的初始坐标 [x, y]
    par['robot_coordinates'] = [3, 3]
    # 障碍物的坐标，以二维数组形式存储
    par['obstacles_coordinates'] = obstacles_coordinates
    # 用于记录已检测到的障碍物坐标，初始为空数组
    par["obstacles_coordinates_detected"] = np.empty((len(obstacles_coordinates), 3))
    # print("origin", par["obstacles_coordinates_detected"])
    # 用于记录检测到的每个障碍物的最小距离，第一行是生成的障碍物的索引，第二行是记录到该障碍物的最小距离，初始为空数组
    par["detected_obstacles_distances"] = np.empty((2, len(obstacles_coordinates)))
    # 目标的坐标 [x, y]
    par['target_coordinates'] = [22, 22]
    # 传感器的探测范围
    par['sensor_range'] = 8
    # 细菌点的步长，为传感器范围的 0.05 倍
    par['step_size'] = 0.05 * par['sensor_range']
    # 障碍物势函数中 alpha 的取值范围
    par['alpha_o_range'] = np.arange(0.1, 50.1, 0.1)
    # 障碍物势函数中 mu 的取值范围
    par['mu_o_range'] = np.arange(1, 1001)
    # 用于存储到障碍物的平均距离，初始化为 0
    par['avg_r'] = np.zeros(1)
    # 用于存储到障碍物的距离总和，初始化为 0
    par['r_sum'] = np.zeros(1)
    # 存储最终用于机器人的障碍物势函数的 alpha 和 mu 值
    par['obstacle_robot'] = [1, 1000]
    # 存储最终用于选定细菌点的障碍物势函数的 alpha 和 mu 值
    par['obstacle_bacteria'] = [1, 1000]
    # 目标势函数的 [alpha, mu] 值
    par['target'] = [100000, 1]
    # 细菌点的数量
    par['bacteria_no'] = 60
    # 细菌点之间的角度间隔
    par['bacteria_degree'] = 360 / par['bacteria_no']
    # 细菌点的角度数组
    par['bacteria_angles'] = np.arange(par['bacteria_degree'], 360 + par['bacteria_degree'], par['bacteria_degree'])
    # 势函数计算的距离下限，低于此距离障碍物势为无穷大
    par['potential_lower_distance_limit'] = 0.4
    # 势函数计算的距离上限，高于此距离障碍物势为 0
    par['potential_upper_distance_limit'] = 4.5
    # 安全距离，若机器人到任何检测到的障碍物的距离低于此值，则终止运行（碰撞情况）
    par['safety_margin'] = 0.25
    # 找到合适细菌点时的提示信息
    par['confirm'] = 'Solution Found'
    # 未找到合适细菌点时的提示信息
    par['error'] = 'Solution Not Found'
    # 计算机器人到目标的距离
    par['RDTT'] = np.sqrt((par['robot_coordinates'][0] - par['target_coordinates'][0])**2 +
                          (par['robot_coordinates'][1] - par['target_coordinates'][1])**2)
    # 初始化势函数值
    par['J_obstRT'] = 0

    # 存储变量的初始化
    # 存储每次成功移动决策时势函数的误差
    par['err_J_sto_s'] = np.zeros(1)
    # 存储每次迭代中具有最小到目标距离的细菌点的势函数误差，用于分析和绘图
    par['err_J_sto'] = np.zeros(1)
    # 存储每次迭代中细菌点的最小势函数误差
    par['err_J_sto_f'] = np.zeros(1)
    # 存储每次迭代中机器人的总势函数值
    par['JT_sto'] = np.zeros(1)
    # 存储每次循环中具有最小到目标距离的细菌点的势函数值，用于分析和绘图
    par['JT_bacteria_sto'] = np.zeros(1)
    # 存储每次循环中细菌点的最小势函数误差
    par['JT_bacteria_sto_f'] = np.zeros(1)
    # 记录机器人移动的次数（成功迭代的次数）
    par['move_count'] = 0
    # 记录迭代的次数（执行循环的次数）
    par['loop'] = 0
    # 记录检测到的障碍物的数量
    par['det'] = 0

    return par
