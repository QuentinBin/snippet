'''
Description: None
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-11-21 19:51:57
LastEditTime: 2024-12-09 02:26:21
'''
import numpy as np
from WaterObject import WaterObject
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tools
import logging



class Assembly:
    def __init__(self):
        self.objects = []  # 存储所有连接的 WaterObject
        self.joints = []   # 存储关节信息：每个关节连接两个物体
        self.system_momentums = []
        self.system_momentum = np.zeros(6)

    def add_object(self, water_object):
        """Adding"""
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

    def update(self, dt, omega, theta):
        """
        更新装配体中所有物体的位置和方向
        :param dt: 时间步长
        """
        print("rotate angle:", theta)
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

            # 构造关节旋转矩阵 (Rodrigues公式)
            SE3_local = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0,0,1]
            ])
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
            child._SE3[:3,:3] = parent._SE3[:3,:3].dot(SE3_local)
            child._SE3[:3,3] = position_global
            child._se3[:3] = parent._se3[:3] + omega * axis_global
            child._se3[3:6] = parent._se3[3:6] + np.cross(parent._se3[:3], position_global-parent._SE3[:3,3])
            

            global_to_bodyi = np.linalg.inv(child._SE3)
            adjoint_matrix = tools.adjoint_matrix(global_to_bodyi)
            child._se3_fixed = np.dot(adjoint_matrix, child._se3)
            child._se3_local = child._se3_fixed - np.dot(adjoint_matrix, self.objects[0]._se3)

        # 更新所有物体的平动 normals
        for idx, obj in enumerate(self.objects):
            obj._SE3[:3,3] += obj._se3[3:6] * dt
            print("object idx:", idx)
            print("object position:", obj._SE3[:3,3])
            print("object velocity:", obj._se3[3:6])
            obj._global_normals = np.dot(obj._normals, obj._SE3[:3,:3].T)
        

    def update_boundary_conditions(self):
        """
        更新所有物体的速度势边界条件
        """
        for obj in self.objects:
            obj.update_boundary_conditions()

    def _compute_added_inertia_matrix(self, obj_idx, obj_jdx,  fluid_density=1):
        """
        计算附加惯性矩阵 M^f_ij。
        
        返回：
            M_f_ij: np.ndarray
                附加惯性矩阵 \( 6 \times 6 \)。
        """
        # 处理边界条件（流域内的物体）
        Theta_ij_chichi = np.zeros((3,3))
        Theta_ij_chipsi = np.zeros((3,3))
        Theta_ij_psichi = np.zeros((3,3))
        Theta_ij_psipsi = np.zeros((3,3))
        rho = fluid_density

        if obj_idx == obj_jdx:
            for bc in self.objects[obj_idx].boundary_conditions: 
                chi = bc['chi']
                psi = bc['psi']
                d_chi = bc['d_chi']
                d_psi = bc['d_psi']
                area = bc['area']
                if(chi is not None and d_chi is not None and d_psi is not None and psi is not None):
                    Theta_ij_chichi += rho * np.dot(chi.reshape(3,1), d_chi.reshape(3,1).T) * area
                    Theta_ij_psipsi += rho * np.dot(psi.reshape(3,1), d_psi.reshape(3,1).T) * area
                    Theta_ij_chipsi += 0.5 * rho * (np.dot(chi.reshape(3,1), d_psi.reshape(3,1).T) + np.dot(d_chi.reshape(3,1), psi.reshape(3,1).T)) * area
                    Theta_ij_psichi += 0.5 * rho * (np.dot(psi.reshape(3,1), d_chi.reshape(3,1).T) + np.dot(d_psi.reshape(3,1), chi.reshape(3,1).T)) * area


        # 组合矩阵
        M_f_ij = np.block([
            [Theta_ij_chichi, Theta_ij_chipsi],
            [Theta_ij_psichi, Theta_ij_psipsi]
        ])

        return M_f_ij
        
    def _compute_I_matrix(self, obj_idx, obj_jdx):
        I_matrix = np.zeros((6,6))

        if obj_idx == obj_jdx:
            I_matrix = self._compute_added_inertia_matrix(obj_idx, obj_jdx, 1) + self.objects[obj_idx]._inertia_matrix

        return I_matrix
    
    # def _compute_system_momentum(self):
    #     system_momentum = np.zeros(6)
    #     system_momentum += np.dot(self._compute_I_matrix(0,0), self.objects[0]._se3)
    #     obj_num = len(self.objects)
    #     for i in range(1,obj_num):
    #         I_matrix_ii = self._comput_I_matrix(i,i)
    #         adjoint_matrix = tools.adjoint_matrix(np.linalg.inv(self.objects[i]._SE3))
    #         system_momentum += np.dot(adjoint_matrix.T, I_matrix_ii).dot(self.objects[i]._se3)

    #     return system_momentum
    
    def update_geometric_locomotion_velocity(self):
        I_loc_matrix = np.zeros((6,6))
        shape_momentum = np.zeros(6)
        I_loc_matrix += self._compute_I_matrix(0,0) #base I matrix
        obj_num = len(self.objects)
        for i in range(1,obj_num):
            base_to_bodyi = np.dot(np.linalg.inv(self.objects[i]._SE3), self.objects[0]._SE3)
            adjoint_matrix = tools.adjoint_matrix(base_to_bodyi)
            I_matrix_ii = self._compute_I_matrix(i,i)
            I_loc_matrix += np.dot(adjoint_matrix.T, I_matrix_ii).dot(adjoint_matrix)
            # print(self.objects[i]._se3_local)
            shape_momentum += np.dot(adjoint_matrix.T, I_matrix_ii).dot(self.objects[i]._se3_local)
        
        print("shape_momentum:", shape_momentum)
        print("I_loc_matrix_inv: \n", np.linalg.inv(I_loc_matrix))
        self.objects[0]._se3_fixed = -np.linalg.inv(I_loc_matrix).dot(shape_momentum)
        self.objects[0]._se3 = np.dot(tools.adjoint_matrix(self.objects[0]._SE3), self.objects[0]._se3_fixed)
        return self.objects[0]._se3
    
    def update_total_locomotion_velocity(self, F_ext, dt): # TO DO
        adjoint_dual_matrix = tools.se3_adjoint_dual_matrix(self.objects[0]._se3)
        self.system_momentum += (np.dot(adjoint_dual_matrix, self.system_momentum) + F_ext) * dt
        self.system_momentums.append(self.system_momentum)

        I_loc_matrix = np.zeros((6,6))
        shape_momentum = np.zeros(6)
        I_loc_matrix += self._compute_I_matrix(0,0) #base I matrix
        obj_num = len(self.objects)
        for i in range(1,obj_num):
            adjoint_matrix = tools.adjoint_matrix(np.linalg.inv(self.objects[i]._SE3))
            I_matrix_ii = self._compute_I_matrix(i,i)
            I_loc_matrix += np.dot(adjoint_matrix.T, I_matrix_ii).dot(adjoint_matrix)
            # print(self.objects[i]._se3_local)
            shape_momentum += np.dot(adjoint_matrix.T, I_matrix_ii).dot(self.objects[i]._se3_local)
        
        self.objects[0]._se3 = np.linalg.inv(I_loc_matrix).dot(self.system_momentum-shape_momentum)
        return self.objects[0]._se3



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
        print('grid:',self.grid.shape)
        # 初始化速度势
        self.potential_chi = np.zeros((self.grid_resolution[0],self.grid_resolution[1],self.grid_resolution[2],3), dtype=float)
        self.potential_psi = np.zeros((self.grid_resolution[0],self.grid_resolution[1],self.grid_resolution[2],3), dtype=float)
        self.potential_phi = np.zeros(grid_resolution, dtype=float)

    def solve_laplace(self, assembly, tolerance=1e-4, max_iterations=1000):
        """
        用有限差分法求解拉普拉斯方程，更新流体域内的速度势
        :param assembly: Assembly 对象，包含所有物体及其边界条件
        :param tolerance: 收敛容限
        :param max_iterations: 最大迭代次数
        """
        # 初始猜测
        potential_chi = self.potential_chi.copy()
        potential_psi = self.potential_psi.copy()
        potential_phi = self.potential_phi.copy()

        # 离散化步长
        dx = self.domain_size[0] / (self.grid_resolution[0] - 1)
        dy = self.domain_size[1] / (self.grid_resolution[1] - 1)
        dz = self.domain_size[2] / (self.grid_resolution[2] - 1)

        # 迭代更新拉普拉斯方程
        for _ in range(max_iterations):
            new_potential_chi = potential_chi.copy()
            new_potential_psi = potential_psi.copy()
            new_potential_phi = potential_phi.copy()

            # 离散拉普拉斯算子
            new_potential_chi[1:-1, 1:-1, 1:-1, :] = (
                (potential_chi[:-2, 1:-1, 1:-1, :] + potential_chi[2:, 1:-1, 1:-1, :]) / dx**2 +
                (potential_chi[1:-1, :-2, 1:-1, :] + potential_chi[1:-1, 2:, 1:-1, :]) / dy**2 +
                (potential_chi[1:-1, 1:-1, :-2, :] + potential_chi[1:-1, 1:-1, 2:, :]) / dz**2
            ) / (2 / dx**2 + 2 / dy**2 + 2 / dz**2)
            new_potential_psi[1:-1, 1:-1, 1:-1, :] = (
                (potential_psi[:-2, 1:-1, 1:-1, :] + potential_psi[2:, 1:-1, 1:-1, :]) / dx**2 +
                (potential_psi[1:-1, :-2, 1:-1, :] + potential_psi[1:-1, 2:, 1:-1, :]) / dy**2 +
                (potential_psi[1:-1, 1:-1, :-2, :] + potential_psi[1:-1, 1:-1, 2:, :]) / dz**2
            ) / (2 / dx**2 + 2 / dy**2 + 2 / dz**2)

            # 处理边界条件（流域内的物体）
            for obj in assembly.objects:
                for bc in obj.boundary_conditions:
                    # 插值到最近的网格点并施加边界条件
                    boundary_position = bc['boundary_position']
                    if(not True in np.isnan(bc['boundary_normal'])):
                        boundary_neighbor_position = bc['boundary_position'] + bc['boundary_normal'] *4* self.domain_size[0] / self.grid_resolution[0]
                        boundary_idx = np.round((boundary_position + self.domain_size / 2 - assembly.objects[0]._SE3[3,:3]) * self.grid_resolution / self.domain_size).astype(int)
                        boundary_neighbor_idx = np.round((boundary_neighbor_position + self.domain_size / 2 - assembly.objects[0]._SE3[3,:3]) * self.grid_resolution / self.domain_size).astype(int)
                        # print(boundary_idx,boundary_neighbor_idx)
                        grad_chi = bc["grad_chi"]
                        grad_psi = bc["grad_psi"]
                        d_chi = bc["d_chi"]
                        d_psi = bc["d_psi"]
                        new_potential_phi[tuple(boundary_neighbor_idx)] = new_potential_phi[tuple(boundary_idx)] + grad_chi + grad_psi 
                        new_potential_chi[boundary_neighbor_idx[0],boundary_neighbor_idx[1],boundary_neighbor_idx[2],:] = new_potential_chi[boundary_idx[0],boundary_idx[1],boundary_idx[2],:] + d_chi
                        new_potential_psi[boundary_neighbor_idx[0],boundary_neighbor_idx[1],boundary_neighbor_idx[2],:] = new_potential_psi[boundary_idx[0],boundary_idx[1],boundary_idx[2],:] + d_psi

                        # 更新boundary表面的速度势
                        bc["chi"] = new_potential_chi[boundary_idx[0],boundary_idx[1],boundary_idx[2],:]
                        bc["psi"] = new_potential_psi[boundary_idx[0],boundary_idx[1],boundary_idx[2],:]
                        bc['phi'] = new_potential_phi[tuple(boundary_idx)]
            # 检查收敛性
            if np.max(np.abs(new_potential_phi - potential_phi)) < tolerance:
                break

            potential_chi = new_potential_chi
            potential_psi = new_potential_psi
            potential_phi = new_potential_phi

        self.potential_chi = potential_chi
        self.potential_psi = potential_psi
        self.potential_phi = potential_phi

    

    def plot_potential(self):
        """
        可视化流体域内的速度势切片
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.potential_phi[:, :, self.grid_resolution[2] // 2], origin='lower', cmap='jet')
        plt.colorbar(label="Velocity Potential")
        plt.title("Velocity Potential at Mid-Z Plane")
        plt.show()

    def plot_potential_assembly(self, assembly):
        boundary_positions_xs = []
        boundary_positions_ys = []
        boundary_positions_zs = []
        potentials = []
        for obj in assembly.objects:
                for bc in obj.boundary_conditions:
                    boundary_position = bc['boundary_position']
                    boundary_idx = np.round((boundary_position + self.domain_size / 2) * self.grid_resolution / self.domain_size).astype(int)
                    boundary_positions_xs.append(boundary_idx[0]) 
                    boundary_positions_ys.append(boundary_idx[1]) 
                    boundary_positions_zs.append(boundary_idx[2]) 
                    potentials.append(self.potential_phi[tuple(boundary_idx)])

        # 绘制散点图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 创建散点图，使用速度势作为颜色
        scatter = ax.scatter(boundary_positions_xs, boundary_positions_ys, boundary_positions_zs, c=potentials, cmap='viridis')

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Velocity Potential')

        # 设置轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
       
        ax.set_xlim(15, 40)
        ax.set_ylim(15, 40)
        ax.set_zlim(15, 40)
        ax.set_box_aspect([1, 1, 1])  # 保证三个轴的比例一致
        # 显示图形
        plt.show()
        
