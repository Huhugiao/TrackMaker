import numpy as np

def Bacteria_Calculate_Distances_Obstacles(par):
    # 遍历检测到的障碍物距离矩阵的每一列
    for i in range(par['det']):
        # 计算机器人与当前障碍物之间的欧几里得距离
        # par['robot_coordinates'] 是机器人的坐标，par['obstacles_coordinates'] 是障碍物的坐标
        # par['detected_obstacles_distances'][0, i] 表示当前障碍物在障碍物坐标矩阵中的索引
        distance = np.sqrt((par['robot_coordinates'][0] - par['obstacles_coordinates'][int(par['detected_obstacles_distances'][0, i]) - 1, 0])**2 +
                           (par['robot_coordinates'][1] - par['obstacles_coordinates'][int(par['detected_obstacles_distances'][0, i]) - 1, 1])**2)
        # 如果计算得到的距离小于之前记录的该障碍物的检测距离
        if distance < par['detected_obstacles_distances'][1, i]:
            # 则更新该障碍物的检测距离为新计算的距离
            par['detected_obstacles_distances'][1, i] = distance
    # 返回更新后的参数字典
    return par    