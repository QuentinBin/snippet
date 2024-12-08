'''
Description: None
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-11-22 14:36:11
LastEditTime: 2024-12-09 02:32:45
'''
import numpy as np
from FluidDomain import FluidDomain, Assembly
from WaterObject import WaterObject
import tools
import logging
import matplotlib.pyplot as plt


# 构建鱼的几何
body_vertices, body_triangles = tools.generate_ellipsoid(0.4, 0.06, 0.09, resolution=100)
tail_vertices, tail_triangles = tools.generate_plate(0.2, 0.02, 0.08, resolution=30)

# 创建物体
body = WaterObject(vertices=body_vertices, triangles=body_triangles, velocity=[0, 0, 0])
tail = WaterObject(vertices=tail_vertices, triangles=tail_triangles, velocity=[0, 0, 0])
body.compute_inertia_matrix(tools.compute_ellipsoid_inertia(3, 0.4, 0.06, 0.09), 3)
tail.compute_inertia_matrix(tools.compute_plate_inertia(0.15, 0.2, 0.02, 0.08), 0.15)

# 创建装配体
assembly = Assembly()
assembly.add_object(body)
assembly.add_object(tail)
assembly.add_joint(0, 1, joint_position=[1, 0, 0], joint_axis=[0, 0, 1])

# 创建流体域
domain = FluidDomain(grid_resolution=[50, 50, 50], domain_size=[5, 5, 5])

# 模拟正弦摆动的鱼尾
dt = 0.01
# 创建一个图形和坐标轴对象
fig, ax = plt.subplots()

# 设置坐标轴范围
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
# 创建一个散点图对象，初始化为空
scatter, = ax.plot([], [])  # 'bo'表示蓝色圆点


position_xs = []
position_ys = []
for t in np.arange(0, 5, dt):
    # 更新鱼尾关节角度  # 振幅为 1 rad，频率为 1 Hz
    freq = 2
    joint_angle = 0.7 * np.sin(2 * np.pi *freq* t)
    d_joint_angle = 0.7 * np.sin(2 * np.pi *freq* (t+dt)) - 0.7 * np.sin(2 * np.pi *freq* t)
    omega = d_joint_angle / dt

    # TO DO
    assembly.update(dt, omega, joint_angle) # update joint velocitys

    assembly.update_boundary_conditions() # update boundary conditions 
    
    domain.solve_laplace(assembly)  # compute velocity potentials

    assembly.update_geometric_locomotion_velocity() # update geometric locmotion velocities
    
    position_xs.append(assembly.objects[0]._SE3[0,3]*100)
    position_ys.append(assembly.objects[0]._SE3[1,3]*100)
    scatter.set_data(position_xs, position_ys)  # 更新数据
    plt.draw()  # 绘制更新的图形
    plt.pause(0.1)  # 暂停一段时间，便于看到更新

    # 可视化结果
    # if t % (2 * dt) == 0:  # 每隔两步可视化一次
    #     print(f"Time: {t:.2f}")
        # domain.plot_potential()
        # domain.plot_potential_assembly(assembly)
# 让图形保持显示
plt.show()

