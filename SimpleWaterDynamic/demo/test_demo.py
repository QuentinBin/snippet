'''
Description: None
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-12-04 15:55:12
LastEditTime: 2024-12-04 18:20:24
'''
import matplotlib.pyplot as plt
import numpy as np

# 假设你有一个坐标数组，表示物体的位置
# 这里只是一个简单的例子，模拟物体的运动
x = np.linspace(0, 10, 100)  # x坐标
y = np.sin(x)  # y坐标

# 创建一个图形和坐标轴对象
fig, ax = plt.subplots()

# 设置坐标轴范围
ax.set_xlim(0, 10)
ax.set_ylim(-1.5, 1.5)

# 创建一个散点图对象，初始化为空
scatter, = ax.plot([], [], 'bo')  # 'bo'表示蓝色圆点

# 更新函数，每次调用时更新图像
def update(frame):
    scatter.set_data(x[:frame], y[:frame])  # 更新数据
    return scatter,

# 动画效果
from matplotlib.animation import FuncAnimation
ani = FuncAnimation(fig, update, frames=len(x), interval=100, blit=True)

# 显示动画
plt.show()
