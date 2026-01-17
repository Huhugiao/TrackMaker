import numpy as np


def Bacteria_Bacteria_Random_Selection(par):
    # 随机选择细菌索引（Python 索引从 0 开始）
    i = np.random.randint(0, par["bacteria_no"])  # 生成 0 到 bacteria_no-1 的整数

    # 检查势能是否非无穷大
    if not np.isinf(par['J_BT'][i]):
        # 更新机器人坐标并添加高斯噪声
        par["robot_coordinates"][0] = par["bx"][i] + 0.1 * np.random.randn()
        par["robot_coordinates"][1] = par["by"][i] + 0.1 * np.random.randn()
        par["check"] = 1  # 标记选择成功
    return par
