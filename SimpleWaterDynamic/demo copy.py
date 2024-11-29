'''
Description: None
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-11-28 22:50:19
LastEditTime: 2024-11-29 12:16:17
'''
import numpy as np

from WaterObject import WaterObject
import tools

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

a = np.array([1, 2, 3]).reshape(3,1)
b = np.array([4, 5, 6]).reshape(3,1).T
print(np.eye(6).dot(np.ones(6)))