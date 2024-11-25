import numpy as np
from FluidDomain import FluidDomain, Assembly
from WaterObject import WaterObject
import tools

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# 视频设置
frames = 200  # 动画帧数
dt = 0.1      # 每帧时间步
output_dir = "./output_videos"
os.makedirs(output_dir, exist_ok=True)


# 构建鱼的几何
body_vertices, body_triangles = tools.generate_ellipsoid(2, 1, 0.5)
tail_vertices, tail_triangles = tools.generate_plate(0.5, 2, 0.1)
# 创建物体
body = WaterObject(vertices=body_vertices, triangles=body_triangles, velocity=[1, 0, 0])
tail = WaterObject(vertices=tail_vertices, triangles=tail_triangles, velocity=[0, 0, 0])
# 创建装配体
assembly = Assembly()
assembly.add_object(body)
assembly.add_object(tail)
assembly.add_joint(0, 1, joint_position=[2, 0, 0], joint_axis=[0, 0, 1])
# 创建流体域
domain = FluidDomain(grid_resolution=[50, 50, 50], domain_size=[10, 10, 10])

# 初始化 Matplotlib
fig_traj = plt.figure(figsize=(6, 6))
ax_traj = fig_traj.add_subplot(111, projection="3d")
fig_potential = plt.figure(figsize=(6, 6))
ax_potential = fig_potential.add_subplot(111)

# 轨迹存储
trajectory = []

def update_trajectory(frame):
    """更新轨迹图"""
    ax_traj.clear()
    
    # 按正弦摆动
    joint_angle = 1 * np.sin(2 * np.pi * (dt)*(frame+1)) - 0.2 * np.sin(2 * np.pi *dt* frame)
    omega = joint_angle / (dt)
    # 更新装配体
    assembly.update(dt, omega)
    # 更新速度势边界条件
    assembly.update_boundary_conditions()

    position = assembly.objects[0]._SE3[:3,3]
    trajectory.append(position)

    # 绘制轨迹
    trajectory_array = np.array(trajectory)
    ax_traj.plot(trajectory_array[:, 0], trajectory_array[:, 1], trajectory_array[:, 2], 'r-', label="Trajectory")
    ax_traj.set_xlim(-10, 10)
    ax_traj.set_ylim(-10, 10)
    ax_traj.set_zlim(-10, 10)
    ax_traj.set_title("Fish Motion and Trajectory")
    ax_traj.set_xlabel("X")
    ax_traj.set_ylabel("Y")
    ax_traj.set_zlabel("Z")

    return ax_traj

def update_potential(frame):
    """更新速度势图"""
    ax_potential.clear()
    
    # 按正弦摆动
    joint_angle = 1 * np.sin(2 * np.pi * (dt)*(frame+1)) - 0.2 * np.sin(2 * np.pi *dt* frame)
    omega = joint_angle / (dt)
    # 更新装配体
    assembly.update(dt, omega)
    # 更新速度势边界条件
    assembly.update_boundary_conditions()

    # 绘制速度势
    potential = domain.potential[:, :, domain.grid_resolution[2] // 2]  # 中间层
    im = ax_potential.imshow(potential.T, extent=(-5, 5, -5, 5), origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax_potential)
    ax_potential.set_title("Velocity Potential")
    ax_potential.set_xlabel("X")
    ax_potential.set_ylabel("Y")
    return ax_potential

# 保存轨迹动画帧
def save_trajectory_frames():
    for frame in range(frames):
        update_trajectory(frame)
        plt.savefig(f"{output_dir}/trajectory_{frame:03d}.png")
        print(f"Saved trajectory frame {frame+1}/{frames}")

# 保存速度势动画帧
def save_potential_frames():
    for frame in range(frames):
        update_potential(frame)
        plt.savefig(f"{output_dir}/potential_{frame:03d}.png")
        print(f"Saved potential frame {frame+1}/{frames}")

# 调用 FFmpeg 生成视频
def create_video(input_pattern, output_filename, fps=30):
    os.system(f"ffmpeg -r {fps} -i {input_pattern} -vcodec libx264 -crf 25 -pix_fmt yuv420p {output_filename}")

# 保存帧和生成视频
print("Generating trajectory frames...")
save_trajectory_frames()
print("Generating trajectory video...")
create_video(f"{output_dir}/trajectory_%03d.png", f"{output_dir}/trajectory.mp4")

print("Generating potential frames...")
save_potential_frames()
print("Generating potential video...")
create_video(f"{output_dir}/potential_%03d.png", f"{output_dir}/potential.mp4")

print("Videos are saved in", output_dir)
