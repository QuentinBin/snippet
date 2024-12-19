import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 网格参数
L_x, L_y = 0.3, 0.15  # 空间范围（x, y方向）
Nx, Ny = 30, 15  # 网格分辨率（每个方向的网格数量）
n = 0.75  # 波数
freq = 1  # 频率（假设周期为5秒）
amp = 0.1  # 最大振幅（位于鳍条的边缘）
dt = 0.01  # 时间步长


# 创建空间网格
x = np.linspace(0, L_x, Nx)
y = np.linspace(0, L_y, Ny)
X, Y = np.meshgrid(x, y)

def generate_undulating_fin_mesh_right(L_x, L_y, t, resolution=[30,30], amp=1, freq=1, n = 1 ):
    x = np.linspace(0, L_x, resolution[0])
    y = np.linspace(0, L_y, resolution[1])
    xs, ys = np.meshgrid(x, y)
    amplitude = (ys) * amplitude
    zs = amplitude * np.sin((length - x)*k* np.pi / length - omega * t + phi)
    vertices = np.vstack([xs.ravel(), ys.ravel(), zs.ravel()]).T

    triangles = []
    for i in range(resolution[0] - 1):
        for j in range(resolution[1] - 1):
            idx1 = i * resolution[0] + j
            idx2 = idx1 + 1
            idx3 = idx1 + resolution[1]
            idx4 = idx3 + 1
            triangles.append([idx1, idx2, idx3])
            triangles.append([idx2, idx4, idx3])
    return vertices, np.array(triangles)
    
def generate_undulating_fin_mesh_right(length, width,t , resolution=[30,30], amplitude=1, freq=1, phi=0, k = 1 ):
    '''
    rotate axis: 
    '''
    omega = 2 * np.pi / freq
    x = np.linspace(0, length, resolution[0])
    y = np.linspace(-width, 0, resolution[1])
    xs, ys = np.meshgrid(x, y)
    amplitude = (-ys ) * amplitude
    zs = amplitude * np.sin(-(length - x)*k* np.pi / length - omega * t + phi)
    vertices = np.vstack([xs.ravel(), ys.ravel(), zs.ravel()]).T

    triangles = []
    for i in range(resolution[0] - 1):
        for j in range(resolution[1] - 1):
            idx1 = i * resolution[0] + j
            idx2 = idx1 + 1
            idx3 = idx1 + resolution[1]
            idx4 = idx3 + 1
            triangles.append([idx1, idx2, idx3])
            triangles.append([idx2, idx4, idx3])
    return vertices, np.array(triangles)

# 计算每个网格的位移（波动模型）
def displacement(X, Y, t):
    # 计算到鳍条的距离，并线性分布幅值
    dist_to_finn = (-Y  )  # 假设鳍条在y = Ly / 2处
    amplitude = (dist_to_finn ) * A  # 振幅线性分布
    Z = amplitude * np.sin(k * (Lx-X) - omega * t + phi)
    vertices = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    print("vertices shape:", vertices.shape)
    return vertices


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
