import numpy as np

from WaterObject import WaterObject
import tools

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

a = np.array([1, 2, 3]).reshape(3,1)
b = np.array([4, 5, 6]).reshape(3,1).T
print(a.dot(b))