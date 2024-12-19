'''
Description: Failed
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-12-15 15:54:45
LastEditTime: 2024-12-15 22:52:42
'''
import sympy as sp
from sympy import exp, sin, cos
import numpy as np
# 定义符号变量
u, v, w, freq, amp, bias = sp.symbols('u, v, w, freq, amp, bias')
u_dot, v_dot, w_dot = sp.symbols('u_dot, v_dot, w_dot')
k_1, k_r = sp.symbols('k_1, k_r')

r, theta = sp.symbols('r, theta')


a2 = -0.1103 * u + 13.9448 * v + 12.8666 * w
a1 = -3.1762 * u + 0.0279 * v + -2.4948 * w
C = sp.Matrix([
            [0, 0 , -17*v-a2],
            [0, 0, 17*u+a1],
            [17*v+a2, -17*u-a1, 0]])
D = sp.Matrix([
            [0.5996, 2.4967, -1.9539],
            [-0.3623, 11.0042, 9.2471],
            [0.2295, -6.5004, -7.3168],])

M = sp.Matrix([
            [20.6222, -0.0279, 2.4647],
            [0.1103, 3.5012, -12.8056],
            [-0.1222, 8.2168, 9.2795],
        ])

Velocity_dot = sp.Matrix([u_dot, v_dot, w_dot])
Velocity = sp.Matrix([u, v, w])
thrust = (0.0514 * sp.exp(0.03102*(freq-0.4961)**2) - 1.7630 * sp.exp(-2.2080*(freq-1.8670)**2) \
                    - 0.8956 * sp.exp(-1.9130*(freq-1.0820)**2)) * (0.0085*amp**2-0.6171*amp+4.0280)

F_thrust =sp.Matrix([thrust*sp.cos(bias), thrust*sp.sin(bias), thrust*0.2*sp.sin(bias)])

eq1 = sp.Eq(M*Velocity_dot, -(C+D)*Velocity+F_thrust)
solution1 = sp.solve(eq1, [u_dot, v_dot, w_dot])
print("加速度的解析解:", solution1)

X = sp.Matrix([solution1[v_dot]*sin(theta) - solution1[u_dot]*cos(theta)+(k_1+k_r)*(-u*cos(theta)+v*sin(theta))+k_1*k_r*r+ \
            (u*sin(theta)+v*cos(theta))*(u/r*sin(theta)+v/r*cos(theta)+w)])
Y = sp.Matrix([0])
eq2 = sp.Eq(X, Y)
solution2 = sp.solve(eq2, [amp])
print(solution2)
