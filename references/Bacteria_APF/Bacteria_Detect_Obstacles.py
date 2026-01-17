import numpy as np

def Bacteria_Detect_Obstacles(par):
    # 遍历所有障碍物的坐标
    for j in range(par['obstacles_coordinates'].shape[0]):
        # 计算机器人与当前障碍物之间的欧几里得距离
        distance = np.sqrt((par['robot_coordinates'][0] - par['obstacles_coordinates'][j, 0])**2 +
                           (par['robot_coordinates'][1] - par['obstacles_coordinates'][j, 1])**2)
        # 检查当前障碍物是否在传感器范围内且未被检测到过
        if distance <= par['sensor_range'] and not any(np.all(par['obstacles_coordinates'][j] == row, axis=0) for row in par['obstacles_coordinates_detected']):
            # 如果满足条件，将该障碍物坐标添加到已检测到的障碍物坐标中
            par['obstacles_coordinates_detected'] = np.vstack((par['obstacles_coordinates_detected'], par['obstacles_coordinates'][j]))
            # print("detect", par["obstacles_coordinates_detected"])
            # 记录检测到的障碍物在障碍物坐标数组中的索引
            par['detected_obstacles_distances'][0, par['det']] = j
            # 记录检测到的障碍物与机器人的距离
            par['detected_obstacles_distances'][1, par['det']] = distance
            # 已检测到的障碍物数量加 1
            par['det'] = par['det'] + 1
    return par
