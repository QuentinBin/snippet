'''
Description: None
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-12-13 11:24:07
LastEditTime: 2024-12-15 16:04:28
'''
import numpy as np

class RobotFish:
    def __init__(self, init_vel=[0,0,0], init_pos=[0,0,0]):
        self._mass = 17 # Unit:kg
        self._position = np.array(init_pos, dtype=float)  # 位置 [x, y]
        self._velocity = np.array(init_vel, dtype=float)  # 速度 [vx, vy]

        self._M = np.array([
            [20.6222, -0.0279, 2.4647],
            [0.1103, 3.5012, -12.8056],
            [-0.1222, 8.2168, 9.2795],
        ])
        self._M_inv = np.linalg.inv(self._M)
        self._C = self._compute_C()
        self._D = np.array([
            [0.5996, 2.4967, -1.9539],
            [-0.3623, 11.0042, 9.2471],
            [0.2295, -6.5004, -7.3168],
        ])
        self._eta = 0 # thrust model
        self._k_r = 0  # 控制增益 k_r
        self._k_1 = 0  # 控制增益 k_1
        self._target = None  # 当前目标点
    
    def _compute_C(self):
        C_rb = np.array([
            [0, 0, -self._mass * self._velocity[1]],
            [0,0, self._mass * self._velocity[0]],
            [self._mass * self._velocity[1], -self._mass * self._velocity[0], 0]
        ])
        a2 = -0.1103 * self._velocity[0] + 13.9448 * self._velocity[1] + 12.8666 * self._velocity[2]
        a1 = -3.1762 * self._velocity[0] + 0.0279 * self._velocity[1] + -2.4948 * self._velocity[2]
        C_a = -np.array([
            [0, 0 , a2],
            [0, 0, -a1],
            [-a2, a1, 0],
        ])

        return C_rb + C_a
    
    def set_control_input(self, eta, freq, amp, bias):
        self._eta = eta
        if self._eta == 0:
            thrust = (0.0514 * np.exp(0.03102*(freq-0.4961)**2) - 1.7630 * np.exp(-2.2080*(freq-1.8670)**2) \
                    - 0.8956 * np.exp(-1.9130*(freq-1.0820)**2)) * (0.0085*amp**2-0.6171*amp+4.0280)
            F_thrust = np.array([thrust*np.cos(bias), thrust*np.sin(bias), thrust*0.2*np.sin(bias)])

        if self._eta == 1:
            thrust = (0.9262*freq**3-1.6480*freq**2+1.5960*freq+0.6419)*(0.0442*amp-1.5290) \
                    +(0.1016*freq**3+1.5580*freq**2-2.0650*freq+1.4700)*(-0.1958*amp+3.0170) \
                    +(0.0535*freq**3+0.7496*freq**2+1.3190*freq+0.3941)*(0.1191*amp-0.7917)
            F_thrust = np.array([thrust*2, 0, 0])

        if self._eta == 2:
            thrust = (0.9262*freq**3-1.6480*freq**2+1.5960*freq+0.6419)*(0.0442*amp-1.5290) \
                    +(0.1016*freq**3+1.5580*freq**2-2.0650*freq+1.4700)*(-0.1958*amp+3.0170) \
                    +(0.0535*freq**3+0.7496*freq**2+1.3190*freq+0.3941)*(0.1191*amp-0.7917)
            F_thrust = np.array([thrust, 0, np.sign(bias)*thrust*0.1])
        
        return F_thrust

    def set_target(self, distance_error, orientation_error):
        self._target = np.array([distance_error, orientation_error], dtype=float)

    def control_law(self):
        """
        李雅普诺夫控制律，用于计算下一步速度。
        """
        if self.target is None:
            raise ValueError("Target not set!")

        # 当前状态
        r = np.linalg.norm(self.target - self.position)  # 与目标的距离
        theta = np.arctan2(self.target[1] - self.position[1], self.target[0] - self.position[0])  # 方位角

        u = self._velocity[0]  # x方向速度
        v = self._velocity[1]  # y方向速度

        z1 = self._k_r * r - u * np.cos(theta) - v * np.sin(theta)  # 李雅普诺夫变量

        # 速度更新控制律
        control_force = -self.k_1 * z1
        tau = np.array([control_force * np.cos(theta), control_force * np.sin(theta)])

        return tau

    def update(self, dt, F_thrust):
        tau = self.control_law()
        # dynamic equation
        acceleration = -np.dot(self._M_inv, np.dot(self._C+self._D, self._velocity))+self._M_inv.dot(F_thrust)

        # update velocity and position
        self._velocity += acceleration * dt
        self._position += self._velocity * dt

    def has_converged(self, tol=1e-2):
        """
        判断是否已经收敛到目标点。
        :param tol: 收敛判定的距离阈值
        """
        return np.linalg.norm(self.target - self.position) < tol

# 示例使用
if __name__ == "__main__":
    # 初始化机器人鱼
    M = [[1.0, 0.0], [0.0, 1.0]]  # 质量矩阵
    C = [[0.0, -0.1], [0.1, 0.0]]  # 科氏力和离心力矩阵
    D = [[0.1, 0.0], [0.0, 0.1]]  # 阻尼矩阵

    fish = RobotFish(init_pos=[0, 0], init_vel=[0, 0], M=M, C=C, D=D, k_r=1.0, k_1=0.5)

    # 设置目标点
    fish.set_target([5, 5])

    # 模拟
    dt = 0.1
    for t in range(100):
        fish.update(dt)
        print(f"Time: {t*dt:.1f}s, Position: {fish.position}, Velocity: {fish.velocity}")

        if fish.has_converged():
            print("Converged to the target!")
            break
