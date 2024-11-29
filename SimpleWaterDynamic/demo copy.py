'''
Description: None
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-11-24 22:14:07
LastEditTime: 2024-11-29 21:15:19
'''
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

# 定义 ad^T 矩阵的函数
def adT_matrix(xi_alpha):
    """
    根据 xi_alpha 计算 ad^T 矩阵。
    xi_alpha: 列向量或数组
    返回: ad^T 矩阵 (numpy array)
    """
    # 示例：假设 xi_alpha 是 3x1 向量（如刚体角速度）
    return np.array([
        [0, -xi_alpha[2], xi_alpha[1]],
        [xi_alpha[2], 0, -xi_alpha[0]],
        [-xi_alpha[1], xi_alpha[0], 0]
    ])

# 定义外力项 Fξ 的函数
def external_force(t):
    """
    外力项 Fξ 作为时间的函数。
    t: 时间
    返回: 外力项向量
    """
    return np.array([np.sin(t), np.cos(t), 0])  # 示例外力

# 定义微分方程
def odesys(t, h_alpha, xi_alpha):
    """
    微分方程的右侧
    t: 时间
    h_alpha: 当前状态变量
    xi_alpha: 系统的速度或参数
    返回: dh_alpha/dt
    """
    adT = adT_matrix(xi_alpha)
    F_xi = external_force(t)
    return adT @ h_alpha + F_xi

# 初始条件
h_alpha_0 = np.array([1.0, 0.0, 0.0])  # 初始 h_alpha
xi_alpha = np.array([0.0, 0.0, 1.0])  # 假设 xi_alpha 为常值

# 时间范围
t_span = (0, 10)  # 时间范围 [0, 10]
t_eval = np.linspace(t_span[0], t_span[1], 500)  # 评估点

# 求解
solution = solve_ivp(
    fun=lambda t, h: odesys(t, h, xi_alpha),
    t_span=t_span,
    y0=h_alpha_0,
    t_eval=t_eval,
    method='RK45'  # Runge-Kutta 方法
)

# 提取结果
t = solution.t  # 时间
h_alpha = solution.y.T  # 状态向量 (每行为 h_alpha)

# 打印结果
for i, (time, h) in enumerate(zip(t, h_alpha)):
    if i % 50 == 0:  # 每隔若干步打印
        print(f"t = {time:.2f}, h_alpha = {h}")

# 可视化结果 (可选)
import matplotlib.pyplot as plt
plt.plot(t, h_alpha[:, 0], label=r'$h_{\alpha,1}$')
plt.plot(t, h_alpha[:, 1], label=r'$h_{\alpha,2}$')
plt.plot(t, h_alpha[:, 2], label=r'$h_{\alpha,3}$')
plt.xlabel('Time')
plt.ylabel(r'$h_\alpha$')
plt.legend()
plt.title('Solution of $\dot{h}_\\alpha = ad^T_{\\xi_\\alpha} h_\\alpha + F_\\xi$')
plt.grid()
plt.show()
