import numpy as np
import math

class WaterObject:
    def __init__(self, center=None, vertices=None, triangles=None, velocity=None, omega=None):
        """
        Frame: LCS 
        
        :param shape: String('ellipsoid', 'plate', etc.) 
        :param id,parent_id: int
        :param karwg: a=2.0, b=1.5, c=1.0 // l=1
        """
        # Initialize object's info  
        self._center = np.array([0,0,0]) if center is None else center
        # Initialize surface points
        self._vertices = vertices if vertices is not None else np.zeros((0, 3), dtype=np.float32)
        self._triangles = triangles if triangles is not None else np.zeros((0, 3), dtype=np.int32)
        self._normals = self._calculate_normals()
        self._global_normals = self._normals
        self._triangles_area = self._compute_triangles_area_3d()
        # Initialize lie group and lie algebra in {WCS}
        self._SE3 = np.eye(4)
        self._se3 = np.zeros(6)
        self._se3_matrix = np.zeros((4,4))
        self._TransformMatrix_parent2link = np.eye(4)
        # Initialize rotating info
        self._omega = np.array([0,0,0], dtype=np.float32)

        if velocity is not None:
            self._se3[3:6] = np.array(velocity)

    def _calculate_normals(self):
        """
        :return: 校正后的法向量 (Mx3 numpy 数组)。
        """
        # 取三角面的顶点
        v0 = self._vertices[self._triangles[:, 0]]
        v1 = self._vertices[self._triangles[:, 1]]
        v2 = self._vertices[self._triangles[:, 2]]

        # 计算未校正的法向量
        normals = np.cross(v1 - v0, v2 - v0)  # 三角面法向量
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms  # 归一化

        # 确定三角面的中心点
        face_centers = np.array((v0 + v1 + v2) / 3.0)

        # 计算从物体中心指向三角面中心的向量
        center_to_face = face_centers - self._center
        center_to_face /= np.linalg.norm(center_to_face, axis=1, keepdims=True)  # 归一化

        # 校正法向量方向：检查是否与中心到面中心的方向一致
        dot_products = np.sum(normals * center_to_face, axis=1)
        normals[dot_products < 0] *= -1  # 若法向量指向内侧，翻转方向

        return normals
    
    def _compute_triangles_area_3d(self):
        """
        Calculate the areas of multiple triangles in 3D space.

        Args:
            points1, points2, points3 (np.ndarray): Each of shape (N, 3), representing
                                                    the coordinates of the three vertices
                                                    of N triangles.

        Returns:
            np.ndarray: Shape (N,), the areas of the triangles.
        """
        # Calculate edge vectors
        # 取三角面的顶点
        points1 = self._vertices[self._triangles[:, 0]]
        points2 = self._vertices[self._triangles[:, 1]]
        points3 = self._vertices[self._triangles[:, 2]]

        v1 = points2 - points1  # Vector from points1 to points2
        v2 = points3 - points1  # Vector from points1 to points3
        
        # Compute cross products
        cross_products = np.cross(v1, v2)  # Shape (N, 3)
        
        # Compute the magnitudes of the cross products
        cross_magnitudes = np.linalg.norm(cross_products, axis=1)  # Shape (N,)
        
        # Compute triangle areas
        areas = 0.5 * cross_magnitudes
        
        return areas

    def set_center(self, center):
        self._center = center
    
    def set_rotate_value(self, rad, omega):
        self._TransformMatrix_parent2link[:3,0] = np.array([math.cos(rad), -math.sin(rad), 0])
        self._TransformMatrix_parent2link[:3,0] = np.array([math.sin(rad), math.cos(rad), 0])
        self._omega[:,0] = np.array([0, 0, omega])

    def _calculate_rotation_potential(self, points):
        """
        Frame: BSC
        计算旋转流势 χ_i 的几何贡献，不包含角速度。\\

        :param points: 边界点的坐标 (Nx3 的张量)。
        :return: 旋转流势的几何部分 (Nx3 的张量)。
        """
        normals = self._calculate_normals(points)
        relative_positions = points
        rotation_contribution = np.cross(relative_positions, normals, dim=-1)  # r × n
        return rotation_contribution

    def _calculate_translation_potential(self, points):
        """
        计算平动流势 ψ_i 的几何贡献，不包含速度。
        :param points: 边界点的坐标 (Nx3 的张量)。
        :return: 平动流势的几何部分 (Nx1 的张量)。
        """
        normals = self._calculate_normals(points)
        return normals  # 平动部分只与法向量相关

    def calculate_total_potential(self, points):
        """
        计算完整流势 φ，包括速度和角速度。
        :param points: 边界点的坐标 (Nx3 的张量)。
        :return: 总流势梯度 (Nx1 的张量), 几何旋转贡献(NX3), 几何平动贡献(NX3)
        """
        # 几何部分
        rotation_geom = self._calculate_rotation_potential(points)  # 几何旋转贡献
        translation_geom = self._calculate_translation_potential(points)  # 几何平动贡献

        # 添加速度和角速度的权重
        rotation_effect = np.sum(rotation_geom * self.omega, dim=-1, keepdim=True)
        translation_effect = np.sum(translation_geom * self.velocity, dim=-1, keepdim=True)

        # 合并平动和旋转的贡献
        total_potential = rotation_effect + translation_effect
        return total_potential, rotation_geom, translation_geom


    def update_boundary_conditions(self):
        """
        更新边界条件:
        ∂χi/∂n = (ω × r) ⋅ n
        ∂φi/∂n = v ⋅ n
        """
        self.boundary_conditions = []  # 重置边界条件

        # 遍历每个三角面
        for idth, triangle in enumerate(self._triangles):
            vertices = self._vertices[triangle]  # 三角形顶点
            boundary_center = vertices.mean(axis=0)
            global_normal = self._global_normals[idth]

            # 旋转引起的速度势 χi
            r = self._SE3[:3,:3].dot(boundary_center) # 相对于中心的向量 [需要再全局坐标系下表示]
            boundary_position = self._SE3[:3,:3].dot(boundary_center) + self._SE3[:3,3]
            grad_chi = np.dot(np.cross(self._se3[:3], r), global_normal)
            d_chi = np.cross(r, global_normal)
            # 平动引起的速度势 φi
            grad_psi = np.dot(self._se3[3:6], global_normal)
            d_psi = global_normal

            # 存储边界条件
            self.boundary_conditions.append({
                "boundary_position": boundary_position,
                "boundary_normal": global_normal,
                "grad_chi": grad_chi,
                "grad_psi": grad_psi,
                "d_chi": d_chi,
                "d_psi": d_psi,
                "area": self._triangles_area[idth]
                "chi": None,
                "psi": None,
            })