import numpy as np
from scipy.integrate import nquad

# 定义常量参数
l_y = 0.15
l_x = 0.3
freq = 1
n = 0.75
amp = 0.15

# 定义 f_x, f_y, f_t 和 dL_x
def f_x(x, y, t):
    return (amp * 2 * n * np.pi) / (l_x * l_y) * y * np.sin(-2 * np.pi * freq * t + (x - l_x) / l_x * 2 * n * np.pi)

def f_y(x, y, t):
    return (amp) / (l_y) * np.sin(-2 * np.pi * freq * t + (x - l_x) / l_x * 2 * n * np.pi)

def f_t(x, y, t):
    return (-amp * 2 * np.pi * freq) / (l_y) * y * np.cos(-2 * np.pi * freq * t + (x - l_x) / l_x * 2 * n * np.pi)

def dL_x(x, y, t):
    fx = f_x(x, y, t)
    fy = f_y(x, y, t)
    ft = f_t(x, y, t)
    return (ft**2 * np.sqrt(fx**2 + fy**2 + 1) * (-fx)) / np.sqrt(fx**2 + fy**2 + 1e-6)

# 定义积分的范围
ranges = [ (0, l_x), (0, l_y), (0, 1)]  # x, y, t 的积分范围

# 使用 nquad 进行数值积分
L_x, error = nquad(dL_x, ranges)

# 输出结果
print("L_x:", L_x)
