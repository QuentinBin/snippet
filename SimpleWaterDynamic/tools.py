'''
Description: None
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-11-24 21:44:50
LastEditTime: 2024-11-28 20:02:32
'''
import numpy as np
import matplotlib.pyplot as plt

# 椭球生成函数
def generate_ellipsoid(a, b, c, resolution=20):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))
    
    # 将 x, y, z 展平为顶点坐标
    vertices = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    
    # 生成三角形面片
    triangles = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx1 = i * resolution + j
            idx2 = idx1 + 1
            idx3 = idx1 + resolution
            idx4 = idx3 + 1
            triangles.append([idx1, idx2, idx3])
            triangles.append([idx2, idx4, idx3])
    return vertices, np.array(triangles)

# 薄板生成函数
def generate_plate(x_len, y_len, z_len, resolution=10):
    x = np.linspace( 0, x_len , resolution)
    y = np.linspace(0, y_len , resolution)
    z = np.array([-z_len / 2, z_len / 2])
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    vertices = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T
    
    # 面片连接（只考虑两面）
    triangles = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx1 = i * resolution + j
            idx2 = idx1 + 1
            idx3 = idx1 + resolution
            idx4 = idx3 + 1
            triangles.append([idx1, idx2, idx3])
            triangles.append([idx2, idx4, idx3])
    return vertices, np.array(triangles)



