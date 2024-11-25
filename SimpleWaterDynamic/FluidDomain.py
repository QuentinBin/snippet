'''
Description: None
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-11-21 19:51:57
LastEditTime: 2024-11-24 21:32:55
'''
import numpy as np
from WaterObject import WaterObject

class FluidDomain:
    def __init__(self, grid_resolution, domain_size):
        """
        初始化流体域
        :param grid_resolution: 格点分辨率，例如 [nx, ny, nz]
        :param domain_size: 域的物理大小，例如 [Lx, Ly, Lz]
        """
        self.grid_resolution = np.array(grid_resolution)
        self.domain_size = np.array(domain_size)

        # 构建网格
        x = np.linspace(-domain_size[0] / 2, domain_size[0] / 2, grid_resolution[0])
        y = np.linspace(-domain_size[1] / 2, domain_size[1] / 2, grid_resolution[1])
        z = np.linspace(-domain_size[2] / 2, domain_size[2] / 2, grid_resolution[2])
        self.grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)  # [nx, ny, nz, 3]

        # 初始化速度势
        self.potential = np.zeros(grid_resolution, dtype=float)

    def solve_laplace(self, assembly, tolerance=1e-4, max_iterations=1000):
        """
        用有限差分法求解拉普拉斯方程，更新流体域内的速度势
        :param assembly: Assembly 对象，包含所有物体及其边界条件
        :param tolerance: 收敛容限
        :param max_iterations: 最大迭代次数
        """
        # 初始猜测
        potential = self.potential.copy()

        # 离散化步长
        dx = self.domain_size[0] / (self.grid_resolution[0] - 1)
        dy = self.domain_size[1] / (self.grid_resolution[1] - 1)
        dz = self.domain_size[2] / (self.grid_resolution[2] - 1)

        # 迭代更新拉普拉斯方程
        for _ in range(max_iterations):
            new_potential = potential.copy()

            # 离散拉普拉斯算子
            new_potential[1:-1, 1:-1, 1:-1] = (
                (potential[:-2, 1:-1, 1:-1] + potential[2:, 1:-1, 1:-1]) / dx**2 +
                (potential[1:-1, :-2, 1:-1] + potential[1:-1, 2:, 1:-1]) / dy**2 +
                (potential[1:-1, 1:-1, :-2] + potential[1:-1, 1:-1, 2:]) / dz**2
            ) / (2 / dx**2 + 2 / dy**2 + 2 / dz**2)

            # 处理边界条件（流域内的物体）
            for obj in assembly.objects:
                for bc in obj.boundary_conditions:
                    # 插值到最近的网格点并施加边界条件
                    center = bc['boundary_position']
                    idx = np.round((center + self.domain_size / 2) * self.grid_resolution / self.domain_size).astype(int)
                    grad_chi = bc["grad_chi"]
                    grad_phi = bc["grad_phi"]
                    new_potential[tuple(idx)] = grad_chi + grad_phi

            # 检查收敛性
            if np.max(np.abs(new_potential - potential)) < tolerance:
                break

            potential = new_potential

        self.potential = potential

    def plot_potential(self):
        """
        可视化流体域内的速度势切片
        """
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(self.potential[:, :, self.grid_resolution[2] // 2], origin='lower', cmap='jet')
        plt.colorbar(label="Velocity Potential")
        plt.title("Velocity Potential at Mid-Z Plane")
        plt.show()
    

class Assembly:
    def __init__(self):
        self.objects = []  # 存储所有连接的 WaterObject
        self.joints = []   # 存储关节信息：每个关节连接两个物体

    def add_object(self, water_object):
        """添加单个物体到装配体中"""
        self.objects.append(water_object)

    def add_joint(self, obj1_idx, obj2_idx, joint_position=[0,0,0], joint_axis=[0,0,1]):
        """
        添加关节连接信息

        :param obj1_idx: 第一个物体的索引 (父物体)
        :param obj2_idx: 第二个物体的索引 (子物体)
        :param joint_axis: 关节在父物体局部坐标系中的旋转轴 (3,)
        :param joint_position: 关节在父物体局部坐标系中的位置 (3,)
        :param joint_angular_velocity: 关节的角速度 (float)
        """
        self.joints.append({
            "parent": obj1_idx,
            "child": obj2_idx,
            "axis": np.array(joint_axis, dtype=float),
            "position": np.array(joint_position, dtype=float)
        })

    def update(self, dt, omega):
        """
        更新装配体中所有物体的位置和方向
        :param dt: 时间步长
        """
        # # 初始化第一个物体的全局速度和角速度
        # if self.objects:
        #     self.objects[0].global_velocity = self.objects[0].velocity
        #     self.objects[0].global_omega = self.objects[0].omega

        for joint in self.joints:
            # 获取关节信息
            parent = self.objects[joint["parent"]]
            child = self.objects[joint["child"]]
            axis_local = joint["axis"]
            position_local = joint["position"]

            # 将关节的局部轴和位置转换到全局坐标系
            axis_global = parent._SE3[:3,:3].dot(axis_local)
            axis_global = axis_global / np.linalg.norm(axis_global)  # 归一化轴
            position_global = parent._SE3[:3,:3].dot(position_local) + parent._SE3[:3,3]

            # 计算关节旋转角度
            theta = omega * dt

            # 构造关节旋转矩阵 (Rodrigues公式)
            K = np.array([
                [0, -axis_global[2], axis_global[1]],
                [axis_global[2], 0, -axis_global[0]],
                [-axis_global[1], axis_global[0], 0]
            ])
            rotation_matrix = (
                np.eye(3) +
                np.sin(theta) * K +
                (1 - np.cos(theta)) * np.dot(K, K)
            )

            # 更新子物体的方向
            child._SE3[:3,:3] = rotation_matrix.dot(child._SE3[:3,:3])
            child._SE3[:3,3] = position_global
            child._se3[:3] = parent._se3[:3] + omega * axis_global
            child._se3[3:6] = parent._se3[3:6] + np.cross(parent._se3[:3], position_global-parent._SE3[:3,3])

        # 更新所有物体的平动 normals
        for obj in self.objects:
            obj._SE3[:3,3] += obj._se3[3:6] * dt
            obj._global_normals = np.dot(obj._normals, obj._SE3[:3,:3].T)

    def update_boundary_conditions(self):
        """
        更新所有物体的速度势边界条件
        """
        for obj in self.objects:
            obj.update_boundary_conditions()
        


    