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

    def set_motion(self, velocity, omega):
        """设置物体的速度和角速度。"""
        self.velocity = np.array(velocity, dtype=np.float64)
        self.omega = np.array(omega, dtype=np.float64)

    def update_position(self, dt):
        """根据速度更新位置。"""
        self.center += self.velocity * dt

    def boundary_condition(self, points):
        """
        计算物体边界上的法向速度。
        :param points: 边界上的点(Nx3 的张量，每行是一个点坐标）。
        :return: 边界条件 (Nx1 的张量)。
        """
        if self.shape == "ellipsoid":
            return self._ellipsoid_boundary_condition(points)
        elif self.shape == "plate":
            return self._plate_boundary_condition(points)
        else:
            raise ValueError(f"Shape '{self.shape}' is not supported.")
    
    def _ellipsoid_boundary_condition(self, points):
        """
        椭球的边界条件计算。
        椭球定义为 (x/a)^2 + (y/b)^2 + (z/c)^2 = 1。
        :param points: 边界上的点(Nx3 的张量，每行是一个点坐标）。
        :return: 边界条件 (Nx1 的张量)。
        """
        a, b, c = self.params["a"], self.params["b"], self.params["c"]

        # 法向量计算：梯度 (2x/a^2, 2y/b^2, 2z/c^2)
        gradients = np.stack([
            2 * (points[:, 0] - self.center[0]) / a**2,
            2 * (points[:, 1] - self.center[1]) / b**2,
            2 * (points[:, 2] - self.center[2]) / c**2
        ], dim=-1)

        # 归一化法向量
        normals = gradients / np.linalg.norm(gradients, dim=-1, keepdim=True)

        # 计算总速度：v + ω × r
        relative_positions = points - self.center  # 每个点相对中心的偏移
        total_velocity = self.velocity + np.cross(self.omega, relative_positions, dim=-1)

        # 法向速度：dot(total_velocity, normals)
        normal_velocity = (total_velocity * normals).sum(dim=-1, keepdim=True)

        return normal_velocity

    def _plate_boundary_condition(self, points):
        """
        平板的边界条件计算。
        平板定义为平面 z = 0。
        :param points: 边界上的点（Nx3 的张量，每行是一个点坐标）。
        :return: 边界条件 (Nx1 的张量)。
        """
        # 平板法向量是固定的
        normals = np.tensor([[0.0, 0.0, 1.0]], dtype=np.float32).repeat(points.shape[0], 1)

        # 计算总速度：v + ω × r
        relative_positions = points - self.center  # 每个点相对中心的偏移
        total_velocity = self.velocity + np.cross(self.omega, relative_positions, dim=-1)

        # 法向速度：dot(total_velocity, normals)
        normal_velocity = (total_velocity * normals).sum(dim=-1, keepdim=True)

        return normal_velocity


class RotationalJoint:
    def __init__(self, parent, child, axis, initial_angle=0.0, angular_velocity=0.0):
        """
        初始化旋转关节。
        :param parent: 父物体 (WaterObject)
        :param child: 子物体 (WaterObject)
        :param axis: 旋转轴 [x, y, z]
        :param initial_angle: 初始角度 (rad)
        :param angular_velocity: 初始角速度 (rad/s)
        """
        self.parent = parent
        self.child = child
        self.axis = np.array(axis, dtype=np.float64)
        self.axis /= np.linalg.norm(self.axis)  # 归一化旋转轴
        self.angle = initial_angle
        self.angular_velocity = angular_velocity

    def update(self, dt):
        """更新关节角度，并调整子物体的姿态和运动。"""
        # 更新角度
        self.angle += self.angular_velocity * dt

        # Rodrigues 公式计算旋转矩阵
        c = np.cos(self.angle)
        s = np.sin(self.angle)
        v = 1 - c
        x, y, z = self.axis
        R = np.array([
            [c + x**2 * v, x * y * v - z * s, x * z * v + y * s],
            [y * x * v + z * s, c + y**2 * v, y * z * v - x * s],
            [z * x * v - y * s, z * y * v + x * s, c + z**2 * v]
        ])

        # 计算子物体的位置
        relative_position = self.child.center - self.parent.center
        self.child.center = self.parent.center + R @ relative_position

        # 更新子物体速度和角速度
        self.child.velocity = self.parent.velocity
        self.child.omega = self.parent.omega + self.angular_velocity * self.axis


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
