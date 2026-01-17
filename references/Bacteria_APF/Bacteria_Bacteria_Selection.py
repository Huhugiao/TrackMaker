import numpy as np


def Bacteria_Bacteria_Selection(par):
    # 初始化误差数组
    err_DTT = np.zeros(par["bacteria_no"])
    err_J = np.zeros(par["bacteria_no"])

    # 计算每个细菌的误差
    for i in range(par["bacteria_no"]):
        err_DTT[i] = par["BDTT"][i] - par["RDTT"]
        err_J[i] = par["J_BT"][i] - par["J_RT"]

    par["check"] = 0  # 初始标记未找到解

    # 遍历所有细菌尝试选择
    for _ in range(par["bacteria_no"]):
        mi = np.argmin(err_DTT)  # 找到当前最小距离误差的索引
        if err_J[mi] < 0:
            # 更新机器人坐标并添加噪声
            par["robot_coordinates"][0] = par["bx"][mi] + 0.1 * np.random.randn()
            par["robot_coordinates"][1] = par["by"][mi] + 1 * np.random.randn()
            par["check"] = 1  # 标记成功选择
            break
        else:
            err_DTT[mi] = np.inf  # 排除当前最小值，继续寻找下一个

    return par
