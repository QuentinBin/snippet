'''
Description: Locomotion of Articulated Bodies in a Perfect Fluid 
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-11-21 11:27:27
LastEditTime: 2024-11-21 11:38:31
'''
import numpy as np
import matplotlib.pyplot as plt

# 参数设置
domain_size = 50  # 计算域的大小
iterations = 5000  # 迭代次数

# 初始化网格
phi = np.zeros((domain_size, domain_size))  # 速度势

# 边界条件
# 固体表面 (模拟一个简单圆形固体)
solid_radius = 10
solid_center = (domain_size // 2, domain_size // 2)
for i in range(domain_size):
    for j in range(domain_size):
        if (i - solid_center[0]) ** 2 + (j - solid_center[1]) ** 2 < solid_radius ** 2:
            phi[i, j] = 0  # 固体边界的速度势设置为固定值

# 流场边界条件 (例如远场速度为常数)
infinity_velocity = 1.0
phi[:, 0] = np.linspace(0, infinity_velocity * domain_size, domain_size)  # 左边界
phi[:, -1] = np.linspace(0, infinity_velocity * domain_size, domain_size)  # 右边界

# 拉普拉斯方程的数值求解 (有限差分方法)
for _ in range(iterations):
    phi_new = phi.copy()
    for i in range(1, domain_size - 1):
        for j in range(1, domain_size - 1):
            if (i - solid_center[0]) ** 2 + (j - solid_center[1]) ** 2 >= solid_radius ** 2:  # 跳过固体区域
                phi_new[i, j] = 0.25 * (phi[i + 1, j] + phi[i - 1, j] + phi[i, j + 1] + phi[i, j - 1])
    phi = phi_new

# 可视化结果
plt.figure(figsize=(8, 6))
plt.contourf(phi, levels=50, cmap="jet")
plt.colorbar(label="Velocity Potential (φ)")
plt.title("Velocity Potential Distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

