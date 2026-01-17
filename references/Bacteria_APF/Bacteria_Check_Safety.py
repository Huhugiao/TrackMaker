import numpy as np

def Bacteria_Check_Safety(par):
    # 初始化检查标志为 1，表示安全
    par['check'] = 1
    # 遍历检测到的障碍物坐标矩阵的每一行
    for i in range(par['obstacles_coordinates_detected'].shape[0]):
        # 计算机器人与当前障碍物之间的欧几里得距离
        # print("check",i,par["obstacles_coordinates_detected"][i])
        distance = np.sqrt((par['robot_coordinates'][0] - par['obstacles_coordinates_detected'][i, 0])**2 +
                           (par['robot_coordinates'][1] - par['obstacles_coordinates_detected'][i, 1])**2)
        # 如果计算得到的距离小于安全距离
        if distance < par["obstacles_coordinates_detected"][i, 2]:
            # 则将检查标志设为 0，表示不安全
            par['check'] = 0
            # 一旦发现不安全情况，跳出循环
            break
    return par
