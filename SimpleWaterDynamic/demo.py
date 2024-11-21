'''
Description: None
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-11-21 13:00:24
LastEditTime: 2024-11-21 13:02:57
'''

import numpy as np
import matplotlib.pyplot as plt

def initialize_domain(domain_size, boundary_value=0):
    """
    初始化速度势场域
    :param domain_size: 整体网格大小 (N, N)
    :param boundary_value: 初始边界值
    :return: 初始化的速度势场
    """
    phi = np.zeros((domain_size, domain_size))
    phi[:, :] = boundary_value
    return phi

def apply_boundary_conditions(phi, X, boundary_func, dx):
    """
    应用边界条件
    :param phi: 速度势场
    :param X: 空间向量场
    :param boundary_func: 边界值函数
    :param dx: 网格间距
    """
    n = phi.shape[0]
    for i in range(n):
        for j in range(n):
            # 定义边界条件
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                ni = np.array([i, j]) / np.linalg.norm([i, j])  # 法向量
                phi[i, j] = boundary_func(X, ni, i, j, dx)

def boundary_func_chi(X, ni, i, j, dx):
    """
    计算 χ_i 的边界值
    :param X: 空间矢量
    :param ni: 当前点的法向量
    :param i, j: 当前点索引
    :param dx: 网格间距
    """
    return np.cross(X, ni)

def boundary_func_phi(X, ni, i, j, dx):
    """
    计算 φ_i 的边界值
    :param ni: 当前点的法向量
    :param i, j: 当前点索引
    :param dx: 网格间距
    """
    return np.dot(ni, ni)

def solve_laplace(phi, max_iterations=1000, tolerance=1e-5):
    """
    用有限差分法求解拉普拉斯方程
    :param phi: 速度势场
    :param max_iterations: 最大迭代次数
    :param tolerance: 收敛容限
    """
    for _ in range(max_iterations):
        phi_new = phi.copy()
        # 更新内部网格点
        for i in range(1, phi.shape[0] - 1):
            for j in range(1, phi.shape[1] - 1):
                phi_new[i, j] = 0.25 * (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1])
        
        # 检查是否收敛
        if np.max(np.abs(phi_new - phi)) < tolerance:
            break
        phi = phi_new
    return phi

def visualize_field(phi, title="Velocity Potential Field"):
    """
    可视化速度势场
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(phi, cmap='viridis', origin='lower')
    plt.colorbar(label='Potential')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

# 初始化参数
domain_size = 50
dx = 1.0
X = np.array([1, 0])  # 常量向量场

# 初始化速度势场
phi_chi = initialize_domain(domain_size)
phi_phi = initialize_domain(domain_size)

# 应用边界条件
apply_boundary_conditions(phi_chi, X, boundary_func_chi, dx)
apply_boundary_conditions(phi_phi, X, boundary_func_phi, dx)

# 求解拉普拉斯方程
phi_chi_solution = solve_laplace(phi_chi)
phi_phi_solution = solve_laplace(phi_phi)

# 可视化结果
visualize_field(phi_chi_solution, title="Solution for χ_i")
visualize_field(phi_phi_solution, title="Solution for φ_i")
