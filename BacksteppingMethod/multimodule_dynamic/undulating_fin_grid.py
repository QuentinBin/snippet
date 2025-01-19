import numpy as np
import sympy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def generate_undulating_fin_mesh_right(L_x, t, resolution=[30,30], freq=1, n = 1 , amp_angl=np.pi/6,plot_flag=False):
    ls = np.linspace(0, L_x, resolution[0])
    vertices = np.zeros([resolution[0], resolution[1], 3])
    amps = np.zeros([resolution[0], resolution[1]])
    for i in range(len(ls)):
        L_y = 0.01*(-10.0/225*(ls[i]*100-15)**2 + 20)
        rous = np.linspace(0, L_y, resolution[1])
        theta = amp_angl * np.sin((-ls[i]+L_x)*n*2*np.pi / L_x - 2*np.pi*freq*t)
        for j in range(len(rous)):
            vertices[i,j,:] =  np.array([ls[i], rous[j]*np.cos(theta), rous[j]*np.sin(theta)])
            amps[i,j] = rous[j]

    triangles = []
    norms = []
    areas = []
    vels = []
    for i in range(resolution[0] - 1):
        for j in range(resolution[1] - 1):
            p1 = vertices[i,j,:]
            p2 = vertices[i,j+1,:]
            p3 = vertices[i+1,j,:]
            p4 = vertices[i+1,j+1,:]

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
                amp = amps[i,j]
                vel1 = -np.pi/6 *2*np.pi*freq*amp*np.cos(np.pi/6 * np.sin((-ls[i]+L_x)*n*2*np.pi / L_x - 2*np.pi*freq*t))  * np.cos((-ls[i]+L_x)*n*2*np.pi / L_x - 2*np.pi*freq*t)
                # 法向量朝向运动方向
                if normal1[2]*vel1 < 0:
                    normal1 = - normal1
                norms.append(normal1)
                triangles.append([p1, p2, p3])
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
                amp = amps[i,j]
                vel2 = -np.pi/6 *2*np.pi*freq*amp*np.cos(np.pi/6 * np.sin((-ls[i]+L_x)*n*2*np.pi / L_x - 2*np.pi*freq*t))  * np.cos((-ls[i]+L_x)*n*2*np.pi / L_x - 2*np.pi*freq*t)
                # 法向量朝向运动方向
                if normal2[2]*vel2 < 0:
                    normal2 = - normal2
                areas.append(area2)
                norms.append(normal2)
                triangles.append([p2, p3, p4])
                vels.append(vel2)
        
    # plot
    # 创建3D绘图
    if plot_flag is True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot_trisurf(vertices[:,0], vertices[:,1], vertices[:,2], triangles=triangles, cmap='viridis', linewidth=0, antialiased=True)
        for j in range(resolution[1]):
            ax.plot(vertices[:,j,0], vertices[:,j,1], vertices[:,j,2])
        # 绘制法向量
        for i, normal in enumerate(norms):
            # 获取每个三角形的质心
            tri_vertices = np.array(triangles[i])
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
L_x = 0.3 # 空间范围（x方向）
Nx, Ny = 30, 20  # 网格分辨率（每个方向的网格数量）
n = 1  # 波数
freq = 1.2 # 频率（假设周期为5秒）
dt = 0.01  # 时间步长


drag_force_avg = 0

freqs = [0.1, 0.5, 0.6, 0.8, 1, 1.2]
amps = [10/180*np.pi, 20/180*np.pi, 30/180*np.pi]
lift_forces = []  # 用于存储升力数据
freq_list = []  # 用于存储频率数据

for amp in amps:
    amp_lift_forces = []  # 存储当前振幅下的升力数据
    for freq in freqs:

        total_lift_force = 0
        total_drag_force = np.zeros(3)

        fin_vers, fin_tris, fin_norms, fin_areas, fin_vels = generate_undulating_fin_mesh_right(L_x, 0, [Nx, Ny], freq, n, plot_flag=False, amp_angl=amp)
        for tri_id,tri in enumerate(fin_tris):
            tri_norm = fin_norms[tri_id]
            tri_area = fin_areas[tri_id]
            tri_vel = fin_vels[tri_id]
            # 计算每个面上的lift and drag forces
            tri_lift = tri_vel**2 * tri_area * np.array([-tri_norm[0], -tri_norm[1], 0]) * np.atan( ((tri_norm[1]/tri_norm[2])**2 +(tri_norm[0]/tri_norm[2])**2) )
            total_lift_force += tri_lift

        drag_force_avg = total_lift_force

        CL = 1.1652890937904505
        # CL = 3.8/2 /  drag_force_avg[0] / 500
        # drag_force_avg = 500 * CL * drag_force_avg
        # print(t, 'coe lift force:', CL)
        lift_force = 2 * 500 * CL * drag_force_avg
        amp_lift_forces.append(lift_force[0])  # 假设升力只考虑x方向
        freq_list.append(freq)
        # print('lift force:', 2 *500 * CL * drag_force_avg)
    lift_forces.append(amp_lift_forces)  # 将每个振幅下的升力数据添加到列表

# 绘制结果
for i, amp in enumerate(amps):
    plt.plot(freqs, lift_forces[i], "o-")#, label=f'Amp = {amp}', )  # 画出不同振幅下的升力与频率的关系

plt.xlabel('Frequency (Hz)')
plt.ylabel('Lift Force')
plt.legend(title='Amplitude')
plt.title('Lift Force vs Frequency for Different Amplitudes')
plt.grid(True)
plt.show()