import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 网格参数
Lx, Ly = 10, 10  # 空间范围（x, y方向）
Nx, Ny = 30, 30  # 网格分辨率（每个方向的网格数量）
k = 2 * np.pi / Lx  # 波数
omega = 2 * np.pi / 5  # 频率（假设周期为5秒）
phi = 0  # 初始相位
A = 1  # 振幅
dt = 0.1  # 时间步长
T = 5  # 动画时长

# 创建空间网格
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# 计算每个网格的位移（波动模型）
def displacement(X, Y, t):
    return A * np.sin(k * X - omega * t + phi)

# 计算法向量（使用相邻顶点计算叉积）
def compute_normals(X, Y, Z):
    # 计算相邻网格点之间的向量
    dx = X[1:, 1:] - X[:-1, :-1]
    dy = Y[1:, 1:] - Y[:-1, :-1]
    dz = Z[1:, 1:] - Z[:-1, :-1]

    # 计算法向量：通过叉积得到每个三角形网格的法向量
    normal_x = dy * dz
    normal_y = dz * dx
    normal_z = dx * dy
    
    return normal_x, normal_y, normal_z

# 创建动画
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_zlim(-1, 1)

# 动画更新函数
def update(t):
    ax.clear()
    
    # 计算位移
    Z_disp = displacement(X, Y, t)
    
    # 计算新的Z坐标（加上位移）
    Z_new = Z_disp

    # 计算法向量
    normal_x, normal_y, normal_z = compute_normals(X, Y, Z_new)

    # 绘制波动的鳍表面
    ax.plot_surface(X, Y, Z_new, cmap='viridis', edgecolor='none', alpha=0.7)

    # 绘制法向量箭头（可选）
    # ax.quiver(X[::2, ::2], Y[::2, ::2], Z_new[::2, ::2], normal_x[::2, ::2], normal_y[::2, ::2], normal_z[::2, ::2], length=0.2, color='r')

    # 设置标题和坐标轴
    ax.set_title(f"Wave on Fin at Time t = {t:.2f} s")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# 创建动画
ani = FuncAnimation(fig, update, frames=np.arange(0, T, dt), interval=100)

# 显示动画
plt.show()
