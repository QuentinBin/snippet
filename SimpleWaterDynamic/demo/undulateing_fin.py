import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 网格参数
Lx, Ly = 10, 10  # 空间范围（x, y方向）
Nx, Ny = 30, 50  # 网格分辨率（每个方向的网格数量）
k = 1 * np.pi / Lx  # 波数
omega = 2 * np.pi / 5  # 频率（假设周期为5秒）
phi = 0  # 初始相位
A = 1  # 最大振幅（位于鳍条的边缘）
dt = 0.1  # 时间步长
T = 5  # 动画时长

# 创建空间网格
x = np.linspace(0, Lx, Nx)
y = np.linspace( -Ly,0, Ny)
X, Y = np.meshgrid(x, y)

def generate_undulating_fin_mesh_right(length, width,t , resolution=[30,30], amplitude=1, freq=1, phi=0, k = 1 ):
    '''
    rotate axis: 
    '''
    omega = 2 * np.pi / freq
    x = np.linspace(0, length, resolution[0])
    y = np.linspace(0, width, resolution[1])
    xs, ys = np.meshgrid(x, y)
    amplitude = (ys ) * amplitude
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

# 计算法向量（使用相邻顶点计算叉积）
def compute_normals(vertices):
    normals = []
    areas = []
    grids = []
    
    # 计算相邻网格点之间的向量
    for i in range(Nx - 1):
        for j in range(Ny - 1):
            # 获取相邻四个网格点的顶点
            idx1 = i * Nx + j
            idx2 = idx1 + 1
            idx3 = idx1 + Ny
            idx4 = idx3 + 1

            p1 = vertices[idx1]
            p2 = vertices[idx2]
            p3 = vertices[idx3]
            p4 = vertices[idx4]


            # 计算法向量：通过叉积得到每个三角形网格的法向量
            v1 = p2 - p1
            v2 = p3 - p1
            normal1 = np.cross(v1, v2)

            v1 = p3 - p2
            v2 = p4 - p2
            normal2 = np.cross(v1, v2)

            # 法向量归一化
            if np.linalg.norm(normal1) != 0:
                normal1 /= np.linalg.norm(normal1)
            if np.linalg.norm(normal2) != 0:
                normal2 /= np.linalg.norm(normal2)

            # 计算面积（每个三角形的面积）
            area1 = np.linalg.norm(np.cross(v1, v2)) / 2.0
            area2 = np.linalg.norm(np.cross(v1, v2)) / 2.0
            area = area1 + area2
            
            # 存储法向量和面积
            normals.append((normal1 + normal2) / 2)
            areas.append(area)
            grids.append([idx1, idx2, idx3, idx4])

    return normals, areas

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
    vertices = displacement(X, Y, t)
    

    # 计算法向量
    normals, areas = compute_normals(vertices)

    # 绘制波动的鳍表面
    ax.plot(vertices[:,0], vertices[:,1], vertices[:,2])

    # 设置标题和坐标轴
    ax.set_title(f"Wave on Fin at Time t = {t:.2f} s")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # # 输出部分法向量和面积
    # print(f"Sample Normals: {normals[:5]}")
    # print(f"Sample Areas: {areas[:5]}")

# 创建动画
ani = FuncAnimation(fig, update, frames=np.arange(0, T, dt), interval=100)

# 显示动画
plt.show()
