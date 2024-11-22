'''
Description: None
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-11-21 19:51:57
LastEditTime: 2024-11-21 20:49:37
'''
import torch
import numpy

import torch

class FluidDomain():
    def __init__(self, nx, ny, nz, length):
        """
        初始化流体域
        :param nx, ny, nz: 网格数量
        :param length: 流体域的物理长度
        """
        self.nx, self.ny, self.nz = nx, ny, nz
        self.length = length
        self.grid = self.create_3d_grid(nx, ny, nz, length)
        self.phi = torch.zeros((nx, ny, nz), dtype=torch.float32)  # 速度势
        self.chi = torch.zeros((nx, ny, nz), dtype=torch.float32)  # 速度势旋转部分
        self.psi = torch.zeros((nx, ny, nz), dtype=torch.float32)  # 速度势平动部分

    def create_3d_grid(self, nx, ny, nz, length):
        """
        创建三维流体网格
        """
        x = torch.linspace(-length, length, nx)
        y = torch.linspace(-length, length, ny)
        z = torch.linspace(-length, length, nz)
        return torch.stack(torch.meshgrid(x, y, z), dim=-1)

    def apply_boundary_conditions(self, boundary_value=1.0):
        """
        应用边界条件
        """
        self.phi[0, :, :] = boundary_value
        self.phi[-1, :, :] = boundary_value
        self.phi[:, 0, :] = boundary_value
        self.phi[:, -1, :] = boundary_value
        self.phi[:, :, 0] = boundary_value
        self.phi[:, :, -1] = boundary_value

    def solve_laplace(self, iterations=500):
        """
        迭代求解拉普拉斯方程
        """
        for _ in range(iterations):
            phi_new = self.phi.clone()
            phi_new[1:-1, 1:-1, 1:-1] = (
                self.phi[:-2, 1:-1, 1:-1] + self.phi[2:, 1:-1, 1:-1] +
                self.phi[1:-1, :-2, 1:-1] + self.phi[1:-1, 2:, 1:-1] +
                self.phi[1:-1, 1:-1, :-2] + self.phi[1:-1, 1:-1, 2:]
            ) / 6.0
            self.phi = phi_new

    def get_phi(self):
        """
        返回速度势场
        """
        return self.phi
    
class WaterSystem:
    def __init__(self):
        """初始化包含多个物体和关节的水中系统。"""
        self.objects = []
        self.joints = []

    def add_object(self, obj):
        """添加物体到系统。"""
        self.objects.append(obj)

    def add_joint(self, joint):
        """添加关节到系统。"""
        self.joints.append(joint)

    def update(self, dt):
        """更新系统的状态，包括物体和关节。"""
        for joint in self.joints:
            joint.update(dt)
        for obj in self.objects:
            obj.update_position(dt)

    def boundary_conditions(self):
        """计算系统中所有物体的边界条件。"""
        conditions = []
        for obj in self.objects:
            conditions.append(obj.boundary_condition())
        return conditions
