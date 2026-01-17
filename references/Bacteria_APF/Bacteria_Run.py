import time
import numpy as np
from Bacteria_Initialize import Bacteria_Initialize
from Bacteria_Plot_Environment import Bacteria_Plot_Environment
from Bacteria_Plot_Robot import Bacteria_Plot_Robot
from Bacteria_Check_Safety import Bacteria_Check_Safety
from Bacteria_Detect_Obstacles import Bacteria_Detect_Obstacles
from Bacteria_Calculate_Distances_Obstacles import Bacteria_Calculate_Distances_Obstacles
from Bacteria_Robot_Potential import Bacteria_Robot_Potential
from Bacteria_Bacteria_Potential import Bacteria_Bacteria_Potential
from Bacteria_Bacteria_Selection import Bacteria_Bacteria_Selection
from Bacteria_Bacteria_Random_Selection import Bacteria_Bacteria_Random_Selection
def generate_obstacles_coordinates(num_obstacles, map_size):
    """
    生成指定数量的障碍物坐标，这些障碍物将随机分布在指定的地图范围内。

    参数:
    num_obstacles (int): 要生成的障碍物数量。
    map_size (list): 地图的范围，格式为 [xmin, xmax, ymin, ymax]。

    返回:
    numpy.ndarray: 包含障碍物坐标的二维数组，每行代表一个障碍物的 [x, y] 坐标。
    """
    xmin, xmax, ymin, ymax = map_size
    # 随机生成障碍物的 x 坐标
    x_coords = np.random.uniform(xmin, xmax, num_obstacles)
    # 随机生成障碍物的 y 坐标
    y_coords = np.random.uniform(ymin, ymax, num_obstacles)
    obstacles_radius = np.random.uniform(1, 2, num_obstacles)
    # 将 x 和 y 坐标组合成二维数组
    obstacles_coordinates = np.column_stack((x_coords, y_coords, obstacles_radius))
    return obstacles_coordinates


def Bacteria_Run(obstacles_coordinates):
    # 调用 Bacteria_Initialize 函数初始化参数
    par = Bacteria_Initialize(obstacles_coordinates)
    # 绘制环境图
    Bacteria_Plot_Environment(par)
    # 记录开始时间
    start_time = time.time()
    # 当机器人到目标的距离大于 0.7 时，进入循环
    while par["RDTT"] > 0.7:
        # 绘制机器人和细菌点
        par = Bacteria_Plot_Robot(par)
        # 记录机器人移动次数
        par["move_count"] = par["move_count"] + 1
        # 检查机器人是否安全
        par = Bacteria_Check_Safety(par)
        # 如果不安全，跳出循环
        if par["check"] == 0:
            print("no safe")
            break
        # 检测障碍物
        par = Bacteria_Detect_Obstacles(par)
        # 计算机器人与障碍物的距离
        par = Bacteria_Calculate_Distances_Obstacles(par)
        # 计算机器人的势能
        par = Bacteria_Robot_Potential(par)
        # 计算细菌点的势能
        par = Bacteria_Bacteria_Potential(par)
        # 选择合适的细菌点
        par = Bacteria_Bacteria_Selection(par)
        # 记录当前时间
        elapsed_time = time.time() - start_time
        par["et"] = elapsed_time
        # 如果不安全，进行随机选择
        if par["check"] == 0:
            print("Collision  may occur!")
            par = Bacteria_Bacteria_Random_Selection(par)
        # 如果运行时间超过 90 秒，跳出循环
        if elapsed_time > 90:
            print("run time error")
            break
        par["fig"].canvas.draw()
        par["fig"].canvas.flush_events()
        # 暂停 0.1 秒
        time.sleep(0.1)

    # 如果机器人安全到达目标，记录总运行时间
    if par["check"] == 1:
        print(
            "The agent reach the target safely"
        )
        par["et2"] = time.time() - start_time
    # 计算机器人与障碍物的平均距离
    if par["detected_obstacles_distances"].size > 0:
        par["average_distance"] = np.mean(par["detected_obstacles_distances"][1, :])
    else:
        par["average_distance"] = 0

    return par


if __name__ == '__main__':
    # 定义地图范围
    map_size = [0, 50, 0, 50]
    # 定义要生成的障碍物数量
    num_obstacles = 10
    # 调用函数生成障碍物坐标
    obstacles_coords= generate_obstacles_coordinates(num_obstacles, map_size)
    # print("生成的障碍物坐标:")
    # print(obstacles_coords)
    Bacteria_Run(obstacles_coords)
