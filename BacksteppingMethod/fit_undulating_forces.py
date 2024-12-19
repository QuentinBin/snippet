import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def generate_undulating_fin_mesh_right(L_x, L_y, t, resolution=[30,30], amp=1, freq=1, n = 1 ,plot_flag=False):
    x = np.linspace(0, L_x, resolution[0]+1)
    y = np.linspace(0, L_y, resolution[1]+1)
    xs, ys = np.meshgrid(x, y)
    amplitude = (ys/L_y) * amp
    zs = amplitude * np.sin((-x+L_x)*n*2*np.pi / L_x - 2*np.pi*freq*t)
    vertices = np.vstack([xs.ravel(), ys.ravel(), zs.ravel()]).T

    triangles = []
    norms = []
    areas = []
    vels = []
    for i in range(resolution[0] - 1):
        for j in range(resolution[1] - 1):
            idx1 = j * resolution[0] + i
            idx2 = idx1 + 1
            idx3 = idx1 + resolution[0]
            idx4 = idx3 + 1
            p1 = vertices[idx1]
            p2 = vertices[idx2]
            p3 = vertices[idx3]
            p4 = vertices[idx4]

            # 计算三角面的 法向量、面积、定点序号、运动速度（垂直看）
            v1 = p2 - p1
            v2 = p3 - p1
            normal1 = np.cross(v1, v2)
            if np.linalg.norm(normal1) != 0:
                # 计算法向量
                normal1 /= np.linalg.norm(normal1)
                # 计算面积（每个三角形的面积）
                area1 = np.linalg.norm(np.cross(v1, v2)) / 2.0
                # 计算运动速度
                center = (p1+p2+p3)/3
                vel1 = -2*np.pi*freq*amp*center[1]/L_y * np.cos((L_x-center[0])*n*2*np.pi / L_x - 2*np.pi*freq*t)
                # 法向量朝向运动方向
                if normal1[2]*vel1 < 0:
                    normal1 = - normal1
                norms.append(normal1)
                triangles.append([idx1, idx2, idx3])
                areas.append(area1)
                vels.append(vel1)

            v1 = p3 - p2
            v2 = p4 - p2
            normal2 = np.cross(v1, v2)
            if np.linalg.norm(normal2) != 0:
                # 计算法向量
                normal2 /= np.linalg.norm(normal2)
                # 计算面积（每个三角形的面积）
                area2 = np.linalg.norm(np.cross(v1, v2)) / 2.0
                 # 计算运动速度
                center = (p2+p3+p4)/3
                vel2 = -2*np.pi*freq*amp*center[1]/L_y * np.cos((L_x-center[0])*n*2*np.pi / L_x - 2*np.pi*freq*t)
                # 法向量朝向运动方向
                if normal2[2]*vel2 < 0:
                    normal2 = - normal2
                areas.append(area2)
                norms.append(normal2)
                triangles.append([idx2, idx4, idx3])
                vels.append(vel2)
        
    # plot
    # 创建3D绘图
    if plot_flag is True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles=triangles, cmap='viridis', linewidth=0, antialiased=True)
        # 绘制法向量
        for i, normal in enumerate(norms):
            # 获取每个三角形的质心
            tri_vertices = vertices[triangles[i]]
            centroid = np.mean(tri_vertices, axis=0)  # 计算三角形的质心
            # 在质心位置绘制法向量
            ax.quiver(centroid[0], centroid[1], centroid[2], normal[0], normal[1], normal[2], length=0.05, color='r')

        # 设置标题和坐标轴标签
        ax.set_title("3D Surface with Normal Vectors")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

    return vertices, np.array(triangles), norms, areas, vels

# 网格参数
L_x, L_y = 0.3, 0.15  # 空间范围（x, y方向）
Nx, Ny = 30, 15  # 网格分辨率（每个方向的网格数量）
n = 1  # 波数

amps = np.linspace(0, 0.15, 301)
freqs = np.linspace(0, 2, 41)
amps_xs, freq_ys = np.meshgrid(amps, freqs)
lift_forces = np.zeros((amps_xs.shape[0]*amps_xs.shape[1], 3))
for i in range(amps_xs.shape[0]):
    for j in range(amps_xs.shape[1]):
        amp = amps_xs[i][j]
        freq = freq_ys[i][j]

        total_lift_force = np.zeros(3)

        fin_vers, fin_tris, fin_norms, fin_areas, fin_vels = generate_undulating_fin_mesh_right(L_x, L_y, 0, [Nx, Ny], amp, freq, n)
        for tri_id,tri in enumerate(fin_tris):
            tri_norm = fin_norms[tri_id]
            tri_area = fin_areas[tri_id]
            tri_vel = fin_vels[tri_id]
            tri_ver = fin_vers[tri]
            # 计算每个面上的lift and drag forces
            tri_lift = tri_vel**2 * tri_area * np.array([-tri_norm[0], -tri_norm[1], 0])
            total_lift_force += tri_lift
        lift_forces[i,:] = total_lift_force
