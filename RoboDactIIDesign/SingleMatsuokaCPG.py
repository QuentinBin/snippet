'''
Description: Single Matsuoka CPG with ReLU activation
Author: Bin Peng
Email: pb20020816@163.com
Date: 2025-07-07 14:03:30
LastEditTime: 2025-07-08 21:02:02
'''
import numpy as np
import matplotlib.pyplot as plt

# 模型参数
T_total = 15.0
dt = 0.001
steps = int(T_total / dt)
tau_r = 0.2
tau_a = 0.2
alpha = 2.5
beta = 2

# 初始化变量
x1, x2 = np.zeros(steps), np.zeros(steps)
v1, v2 = np.zeros(steps), np.zeros(steps)
y1, y2 = np.zeros(steps), np.zeros(steps)

x1[0] = 0.1
x2[0] = -0.1
v1[0] = 0.1
v2[0] = -0.1
# 模拟主循环
for i in range(1, steps):
    t_curr = i * dt

    # 🌀 动态改变输入频率和幅值
    if t_curr < 2.7:
        tau_r = 0.5
        tau_a = 0.5
        u_ext = 1
    else:
        tau_r = 0.5
        tau_a = 0.1
        u_ext = 3
    
    # 动态方程（Matsuoka 神经元核心）
    dx1 = (-x1[i-1] - beta * v1[i-1] - alpha * y2[i-1] + 0) / tau_r
    dx2 = (-x2[i-1] - beta * v2[i-1] - alpha * y1[i-1] + u_ext) / tau_r

    x1[i] = x1[i-1] + dx1 * dt
    x2[i] = x2[i-1] + dx2 * dt

    # ✅ 使用 ReLU 非线性激活（只允许神经元放电为正）
    y1[i] = max(0.0, x1[i])
    y2[i] = max(0.0, x2[i])

    dv1 = (-v1[i-1] + y1[i]) / tau_a
    dv2 = (-v2[i-1] + y2[i]) / tau_a

    v1[i] = v1[i-1] + dv1 * dt
    v2[i] = v2[i-1] + dv2 * dt

# 可视化
t = np.linspace(0, T_total, steps)
plt.plot(t, y1, label='Neuron 1 Output (Extensor)')
plt.plot(t, y2, label='Neuron 2 Output (Flexor)')
plt.plot(t, y1 - y2, label='Output Difference (Fin Signal)', linewidth=2)
plt.axvline(x=2.7, color='red', linestyle='--', label='Freq Switch')
plt.xlabel("Time (s)")
plt.ylabel("Output")
plt.title("Matsuoka CPG with ReLU: Alternating Neurons")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
