'''
Description: None
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-11-22 14:36:11
LastEditTime: 2024-11-25 13:34:09
'''
import numpy as np
from FluidDomain import FluidDomain, Assembly
from WaterObject import WaterObject
import tools


# 构建鱼的几何
body_vertices, body_triangles = tools.generate_ellipsoid(2, 1, 0.5, resolution=100)
tail_vertices, tail_triangles = tools.generate_plate(2, 0.1, 0.5, resolution=100)

# 创建物体
body = WaterObject(vertices=body_vertices, triangles=body_triangles, velocity=[-1, 0, 0])
tail = WaterObject(vertices=tail_vertices, triangles=tail_triangles, velocity=[0, 0, 0])

# 创建装配体
assembly = Assembly()
assembly.add_object(body)
assembly.add_object(tail)
assembly.add_joint(0, 1, joint_position=[2, 0, 0], joint_axis=[0, 0, 1])

# 创建流体域
domain = FluidDomain(grid_resolution=[200, 200, 200], domain_size=[10, 10, 10])

# 模拟正弦摆动的鱼尾
dt = 0.1
for t in np.arange(0, 2 * np.pi, dt):
    # 更新鱼尾关节角度  # 振幅为 1 rad，频率为 1 Hz
    joint_angle = 0.2 * np.sin(2 * np.pi * (t+dt)) - 0.2 * np.sin(2 * np.pi * t)
    omega = joint_angle / dt

    # 更新装配体
    assembly.update(dt, omega)

    # 更新速度势边界条件
    assembly.update_boundary_conditions()
    
    # 求解速度势
    domain.solve_laplace(assembly)
    
    # 可视化结果
    if t % (2 * dt) == 0:  # 每隔两步可视化一次
        print(f"Time: {t:.2f}")
        domain.plot_potential()
