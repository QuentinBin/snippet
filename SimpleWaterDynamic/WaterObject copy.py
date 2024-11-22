import numpy as np

class WaterObject:
    def __init__(self, shape, **kwargs):
        """
        初始化水中物体。
        :param shape: 物体的形状类型 ('ellipsoid', 'plate', etc.)
        :param kwargs: 形状参数及运动参数。
        """
        self.shape = shape
        self.params = kwargs
        self.center = np.array(kwargs.get("center", [0.0, 0.0, 0.0]), dtype=np.float64)
        self.velocity = np.array(kwargs.get("velocity", [0.0, 0.0, 0.0]), dtype=np.float64)
        self.omega = np.array(kwargs.get("omega", [0.0, 0.0, 0.0]), dtype=np.float64)
        self.orientation = np.eye(3, dtype=np.float32)  # 初始方向矩阵

    def set_motion(self, velocity, omega):
        """设置物体的速度和角速度。"""
        self.velocity = np.array(velocity, dtype=np.float64)
        self.omega = np.array(omega, dtype=np.float64)

    def update_position(self, dt):
        """根据速度更新位置。"""
        self.center += self.velocity * dt

    def update_pose(self, joint_rotation):
        """
        更新物体的旋转矩阵和位置。
        :param joint_rotation: 关节旋转矩阵 (3x3 的 numpy 数组)。
        :param dt: 时间步长 (float)。
        """
        self.orientation = joint_rotation @ self.orientation  # 更新方向

    
    def _calculate_normals(self, points):
        """
        Frame: BCS
        计算边界点的法向量。
        :param points: 边界点的坐标 (Nx3 的张量)。
        :return: 法向量 (Nx3 的张量)。
        """
        if self.shape == "ellipsoid":
            a, b, c = self.params["a"], self.params["b"], self.params["c"]
            normals = np.stack([
                2 * (points[:, 0] - self.center[0]) / a**2,
                2 * (points[:, 1] - self.center[1]) / b**2,
                2 * (points[:, 2] - self.center[2]) / c**2
            ], dim=-1)
        elif self.shape == "plate":
            normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float32).repeat(points.shape[0], 1)
        else:
            raise ValueError(f"Shape '{self.shape}' not supported for normals.")
        return normals / np.linalg.norm(normals, dim=-1, keepdim=True)  # 归一化

    def _calculate_rotation_potential(self, points):
        """
        Frame: BSC
        计算旋转流势 χ_i 的几何贡献，不包含角速度。
        :param points: 边界点的坐标 (Nx3 的张量)。
        :return: 旋转流势的几何部分 (Nx3 的张量)。
        """
        normals = self._calculate_normals(points)
        relative_positions = points - self.center
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


# 示例用法
if __name__ == "__main__":
    # 定义两个物体
    parent = WaterObject(shape="ellipsoid", a=2.0, b=1.5, c=1.0, center=[0.0, 0.0, 0.0], velocity=[1.0, 0.0, 0.0])
    child = WaterObject(shape="ellipsoid", a=1.0, b=0.8, c=0.5, center=[2.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0])

    # 定义一个旋转关节
    joint = RotationalJoint(parent, child, axis=[0, 0, 1], initial_angle=0.0, angular_velocity=np.pi / 4)

    # 创建系统
    system = WaterSystem()
    system.add_object(parent)
    system.add_object(child)
    system.add_joint(joint)

    # 模拟系统
    dt = 0.1  # 时间步长
    for t in range(10):  # 模拟 10 个时间步
        system.update(dt)
        print(f"Time {t*dt:.1f} s: Child Center = {child.center}")
