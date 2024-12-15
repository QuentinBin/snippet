'''
Description: None
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-12-15 15:54:45
LastEditTime: 2024-12-15 16:40:32
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

eq1 = sp.Eq(M*Velocity_dot, (C+D)*Velocity+F_thrust)
solution1 = sp.solve(eq1, [u_dot, v_dot, w_dot])
print("加速度的解析解:", solution1)

u_dot_solution= -5.37707803978557e-6*amp**2*exp(4.139732*freq)*exp(-1.913*freq**2)*sin(bias) - 3.92693750907772e-5*amp**2*exp(4.139732*freq)*exp(-1.913*freq**2)*cos(bias) - 4.51672554792251e-8*amp**2*exp(8.24467200000001*freq)*exp(-2.208*freq**2)*sin(bias) - 3.29861289739694e-7*amp**2*exp(8.24467200000001*freq)*exp(-2.208*freq**2)*cos(bias) + 2.91981104889054e-6*amp**2*exp(-0.030778044*freq)*exp(0.03102*freq**2)*sin(bias) + 2.13236918684652e-5*amp**2*exp(-0.030778044*freq)*exp(0.03102*freq**2)*cos(bias) + 0.000390375865688432*amp*exp(4.139732*freq)*exp(-1.913*freq**2)*sin(bias) + 0.00285095663159043*amp*exp(4.139732*freq)*exp(-1.913*freq**2)*cos(bias) + 3.27914274779174e-6*amp*exp(8.24467200000001*freq)*exp(-2.208*freq**2)*sin(bias) + 2.39479296351018e-5*amp*exp(8.24467200000001*freq)*exp(-2.208*freq**2)*cos(bias) - 0.000211978282149453*amp*exp(-0.030778044*freq)*exp(0.03102*freq**2)*sin(bias) - 0.00154810002965057*amp*exp(-0.030778044*freq)*exp(0.03102*freq**2)*cos(bias) + 0.000320915584148041*u**2 - 0.049813197789652*u*v + 0.0676334458466775*u*w + 0.0257605996283507*u + 8.11744768606559e-5*v**2 - 1.50588742735512*v*w + 0.219227221963883*v - 0.641199270454529*w**2 - 0.00664085937799122*w - 0.00254810239344191*exp(4.139732*freq)*exp(-1.913*freq**2)*sin(bias) - 0.0186090638665471*exp(4.139732*freq)*exp(-1.913*freq**2)*cos(bias) - 2.14039653023904e-5*exp(8.24467200000001*freq)*exp(-2.208*freq**2)*sin(bias) - 0.000156315444126057*exp(8.24467200000001*freq)*exp(-2.208*freq**2)*cos(bias) + 0.00138364692999189*exp(-0.030778044*freq)*exp(0.03102*freq**2)*sin(bias) + 0.0101049212760209*exp(-0.030778044*freq)*exp(0.03102*freq**2)*cos(bias)
v_dot_solution = -6.9731417528636e-5*amp**2*exp(4.139732*freq)*exp(-1.913*freq**2)*sin(bias) - 1.54360993357788e-7*amp**2*exp(4.139732*freq)*exp(-1.913*freq**2)*cos(bias) - 5.8574131287297e-7*amp**2*exp(8.24467200000001*freq)*exp(-2.208*freq**2)*sin(bias) - 1.29662660118212e-9*amp**2*exp(8.24467200000001*freq)*exp(-2.208*freq**2)*cos(bias) + 3.78649076409964e-5*amp**2*exp(-0.030778044*freq)*exp(0.03102*freq**2)*sin(bias) + 8.381967503335e-8*amp**2*exp(-0.030778044*freq)*exp(0.03102*freq**2)*cos(bias) + 0.00506250091257897*amp*exp(4.139732*freq)*exp(-1.913*freq**2)*sin(bias) + 1.12066081177754e-5*amp*exp(4.139732*freq)*exp(-1.913*freq**2)*cos(bias) + 4.25248193145776e-5*amp*exp(8.24467200000001*freq)*exp(-2.208*freq**2)*sin(bias) + 9.4135091245822e-8*amp*exp(8.24467200000001*freq)*exp(-2.208*freq**2)*cos(bias) - 0.00274899229473634*amp*exp(-0.030778044*freq)*exp(0.03102*freq**2)*sin(bias) - 6.08530840742121e-6*amp*exp(-0.030778044*freq)*exp(0.03102*freq**2)*cos(bias) - 0.0102554604339553*u**2 + 1.59187432538304*u*v + 2.12822714964243*u*w - 0.00297096603895536*u - 0.00259408292028425*v**2 + 0.227950392780884*v*w + 0.13790256438367*v - 0.170630402264182*w**2 - 0.0573024417009536*w - 0.0330444882123936*exp(4.139732*freq)*exp(-1.913*freq**2)*sin(bias) - 7.3148950734726e-5*exp(4.139732*freq)*exp(-1.913*freq**2)*cos(bias) - 0.000277572471559097*exp(8.24467200000001*freq)*exp(-2.208*freq**2)*sin(bias) - 6.14448464654304e-7*exp(8.24467200000001*freq)*exp(-2.208*freq**2)*cos(bias) + 0.0179435115268157*exp(-0.030778044*freq)*exp(0.03102*freq**2)*sin(bias) + 3.97206648275687e-5*exp(-0.030778044*freq)*exp(0.03102*freq**2)*cos(bias)
X = sp.Matrix([v_dot_solution*sin(theta) - u_dot_solution*cos(theta)+(k_1+k_r)*(-u*cos(theta)+v*sin(theta))+k_1*k_r*r+ \
            (u*sin(theta)+v*cos(theta))*(u/r*sin(theta)+v/r*cos(theta)+w)])
Y = sp.Matrix([0])
eq2 = sp.Eq(X, Y)
solution2 = sp.solve(eq2, [freq, amp, bias])
print(solution2)
