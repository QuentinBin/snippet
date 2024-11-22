import numpy as np
A = np.array([[1, 0], [0, 1]])
B = np.array([[0, -1], [1, 0]])
C = A @ B  # 等价于 np.dot(A, B)
print(np.zeros(3))  # 输出 [[0 -1], [1  0]]