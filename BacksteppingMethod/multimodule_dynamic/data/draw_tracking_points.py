import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split

from sklearn.cluster import DBSCAN
from scipy.optimize import least_squares

# 拟合圆的函数
def fit_circle(x, y):
	"""拟合圆并计算曲率半径"""
	def residuals(params):
		x0, y0, R = params
		return np.sqrt((x - x0)**2 + (y - y0)**2) - R

	# 初始猜测值（圆心在路径中心附近，半径取平均距离）
	x0_guess = np.mean(x)
	y0_guess = np.mean(y)
	R_guess = np.mean(np.sqrt((x - x0_guess)**2 + (y - y0_guess)**2))
	
	# 使用最小二乘法拟合圆
	result = least_squares(residuals, [x0_guess, y0_guess, R_guess])
	 # 拟合圆的圆心坐标和半径
	x0, y0, radius = result.x
	
	# 计算每个点的角度（相对于圆心）
	angles = np.arctan2(y - y0, x - x0)
	
	# 计算角度差（弧长）
	angle_diff = np.diff(angles)  # 计算连续点之间的角度差
	arc_length = np.sum(np.abs(angle_diff)) * radius  # 弧长 = 半径 * 角度差之和
	return radius, arc_length  # 返回曲率半径和弧长

# 计算曲率、运动时间和夹角的函数
def calculate_curvature_and_angles(head_points, timestamps, stability_threshold=0.01):
	# 朝向方向
	head_dirs = np.diff(head_points, axis=0)  # 计算连续点之间的差值，代表朝向变化
	head_dirs = np.vstack([head_dirs, head_dirs[-1]])  # 补充最后一个方向，使得与head_points长度一致
	positions = head_points
	
	# 计算每50个连续点的运动曲率半径、时间、夹角
	window_size = 75
	window_interval = 5
	num_windows = (len(head_points)-window_size-1) // window_interval
	curvatures = []
	omegas = []
	angles = []
	velocity_fowards = []
	for i in range(num_windows):
		start_idx = i * window_interval
		end_idx = i * window_interval + window_size

		# 获取当前窗口的位置数据
		x = positions[start_idx:end_idx, 0]
		y = positions[start_idx:end_idx, 1]
		
		# 拟合曲率半径和弧长
		radius, arc_length = fit_circle(x, y)
		curvatures.append(radius)

		# 计算角速度（omega），即弧长除以半径和时间差
		omega = arc_length / radius / (timestamps[end_idx-1] - timestamps[start_idx])
		omegas.append(omega)
	
		# 计算夹角：朝向方向和切线方向的夹角
		# 计算路径切向方向：相邻两点之间的方向
		tangent_dirs = positions[start_idx + 1:end_idx + 1] - positions[start_idx:end_idx]  # 相邻点的差值
		tangent_angles = np.arctan2(tangent_dirs[:, 1], tangent_dirs[:, 0])  # 计算切线方向的角度
		
		# 计算朝向方向的角度
		head_angles = np.arctan2(head_dirs[start_idx:end_idx, 1], head_dirs[start_idx:end_idx, 0])  # 头部方向角度
		
		# 计算夹角：朝向方向和路径切线的夹角
		angle_differences = np.abs(tangent_angles - head_angles)  # 计算角度差
		angle_differences = np.minimum(angle_differences, 2 * np.pi - angle_differences)  # 夹角是小于180度的

		# 计算平均夹角
		angles.append(np.mean(angle_differences))
		velocity_fowards.append(omega * radius * np.cos(np.mean(angle_differences)))
	# 查找稳定阶段：通过检查曲率、角速度和夹角的变化来判断稳定阶段
	stable_velocity_fowards = []
	stable_curvatures = []
	stable_omegas = []
	stable_angles = []
	
	# 计算稳定阶段（即变化幅度小于某个阈值的段）
	for i in range(1, len(curvatures)):
		if abs(velocity_fowards[i] - velocity_fowards[i-1]) < stability_threshold :
			stable_velocity_fowards.append(velocity_fowards[i])
			stable_curvatures.append(curvatures[i])
			stable_omegas.append(omegas[i])
			stable_angles.append(angles[i])

	# 提取稳定阶段的平均值
	avg_stable_velocity_fowards = get_stable_value(np.array(stable_velocity_fowards), threshold=0.01, method="mean") if get_stable_value(np.array(stable_velocity_fowards), threshold=0.01, method="mean") else None
	avg_stable_curvatures = np.mean(stable_curvatures) if stable_curvatures else None
	avg_stable_omegas = np.mean(stable_omegas) if stable_omegas else None
	avg_stable_angles = np.mean(stable_angles) if stable_angles else None
	
	return avg_stable_velocity_fowards, avg_stable_curvatures, avg_stable_omegas, avg_stable_angles, stable_velocity_fowards, stable_curvatures, stable_omegas, stable_angles, velocity_fowards

def get_stable_value(data, threshold=0.01, method="mean"):
	"""
	提取数组的最终稳定值
	
	参数：
	data (numpy.ndarray): 输入的数据数组
	threshold (float): 变化阈值，用于判断数据是否稳定，默认0.01
	method (str): 提取稳定值的方法，"mean"为计算均值，"last"为取最后一个值，默认"mean"
	
	返回：
	stable_value (float): 最终稳定值
	"""
	# 计算数据的差分
	diff_data = np.abs(np.diff(data))  # 计算相邻元素的差值（绝对值）

	# 找出变化小于阈值的位置，表示系统已稳定
	stable_indices = np.where(diff_data < threshold)[0]  # 获取变化小于阈值的位置索引

	# 如果找到了稳定阶段
	if len(stable_indices) > 0:
		last_stable_index = stable_indices[-1]  # 最后一个稳定点的索引
		stable_segment = data[last_stable_index:]  # 提取稳定阶段的所有数据
		
		if method == "mean":
			stable_value = np.mean(stable_segment)  # 计算稳定阶段的平均值
		elif method == "last":
			stable_value = stable_segment[-1]  # 取稳定阶段的最后一个值
		else:
			raise ValueError("Invalid method! Use 'mean' or 'last'.")
	else:
		# 如果没有找到明显的稳定阶段，直接返回最后一个值
		stable_value = data[-1]
	
	return stable_value


folder_path = "D:\work\科研项目\snippet\BacksteppingMethod\multimodule_dynamic\data"#"./data"

# 使用glob获取所有csv文件路径
Tail_freq_list = [0, 5, 10, 15, 20]
Tail_biases_list = [0, 5, 10, 15, 20]
CPG_freq_list = [0, 1, 5, 10, 12]

fig, axes = plt.subplots(5, 10)
fig.tight_layout(pad=5.0)  # 调整子图之间的间距
# 定义坐标轴的最大最小值，以确保所有图的尺度一致
all_x = []
all_y = []

for tail_freq_id, tail_freq in enumerate(Tail_freq_list):
	plot_id = 0
	for  tail_bias_id,tail_bias in enumerate(Tail_biases_list):
		for  cpg_freq_id, cpg_freq in enumerate(CPG_freq_list):
			csv_file = glob.glob(folder_path + "/" + str(tail_freq) + '-' + str(cpg_freq) + '-' + str(tail_bias)+".csv")
			if len(csv_file) != 0:
				print(csv_file)
				df = pd.read_csv(csv_file[0], header=None, names=['timestamp', 'state0','state1','state2', 'head_x', 'head_y', 'tail_x', 'tail_y'])
				df = df[((df['head_x'].diff().abs()>=1) |
						(df['head_y'].diff().abs()>=1)) ]
				
				df['head_x'] = df['head_x'] / 515 * 2
				df['head_y'] = df['head_y'] / 515 * 2
				df['tail_x'] = df['tail_x'] / 515 * 2
				df['tail_y'] = df['tail_y'] / 515 * 2

				# 示例路径点数据（二维点集，包含噪点）
				path_points_x = np.array(df['head_x'])
				path_points_y = np.array(df['head_y'])
				head_points = np.column_stack((path_points_x, path_points_y))
				timestamps = np.array(df['timestamp'])
				robot_points = (head_points) # + tail_points)/2
				
				all_x.extend(path_points_x)
				all_y.extend(path_points_y)
				# 设置稀疏采样间隔
				sample_interval = int(len(timestamps)/100)  

				ax = axes[tail_freq_id, plot_id]
				plot_id += 1
				# 遍历采样后的点
				for i in range(0, len(head_points), sample_interval):
					head = head_points[i]
					# tail = tail_points[i]
					
					# 计算机器人的朝向向量
					direction = head  
					pos = head 
					# 画出机器人的位置（箭头的起点）
					# ax.quiver(tail[0], tail[1], direction[0], direction[1], angles='xy', scale_units='xy', scale=1, color='blue', headwidth=2)
					
					# 画出路径
					if i > 0:
						prev_pos = robot_points[i - sample_interval]
						ax.plot([prev_pos[0], pos[0]], [prev_pos[1], pos[1]], color='gray', linestyle='--', alpha=0.5)
				
				# 调用函数并打印结果
				avg_stable_velocity_fowards, avg_stable_curvatures, avg_stable_omegas, avg_stable_angles, stable_velocity_fowards, stable_curvatures, stable_omegas, stable_angles, velocity_fowards = calculate_curvature_and_angles(head_points, timestamps)

				# # 设置图形参数
				ax.set_aspect('equal', 'box')
				ax.set_xlabel('X Position')
				ax.set_ylabel('Y Position')
				# # 设置y轴的方向反转
				# ax.gca().invert_yaxis()
				ax.axis('equal')
				ax.grid(True)
				ax.set_title("tail_freq:"+ str(tail_freq)+"||cpg_freq:" +  str(cpg_freq)+ "||tail_bias:"+str(tail_bias))
				
				# 设置坐标轴的范围（所有子图使用相同范围）
				x_min, x_max = min(all_x), max(all_x)
				y_min, y_max = min(all_y), max(all_y)
				ax.set_xlim(x_min, x_max)  # 设置x轴范围一致
				ax.set_ylim(y_min, y_max)  # 设置y轴范围一致
		
								
				# 输出结果并在图上标注
				ax.text(0.05, 0.95, f"Avg Vel: {avg_stable_velocity_fowards:.2f}", transform=ax.transAxes, fontsize=3, color='blue', verticalalignment='top')
				ax.text(0.05, 0.90, f"Avg Curvature: {avg_stable_curvatures:.2f}", transform=ax.transAxes, fontsize=3, color='blue', verticalalignment='top')
				ax.text(0.05, 0.85, f"Avg Omega: {avg_stable_omegas:.2f}", transform=ax.transAxes, fontsize=3, color='blue', verticalalignment='top')
				ax.text(0.05, 0.80, f"Avg Angle: {avg_stable_angles:.2f}", transform=ax.transAxes, fontsize=3, color='blue', verticalalignment='top')

				# # 输出结果
				# # print(velocity_fowards)
				# print("avg_stable_velocity_fowards:", avg_stable_velocity_fowards)
				# print("avg_stable_curvatures:", avg_stable_curvatures)
				# print("avg_stable_omegas:", avg_stable_omegas)
				# print("avg_stable_angles:", avg_stable_angles)
# 显示图形
plt.show()