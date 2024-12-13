import numpy as np

M_matrix = np.array(
    [
        [20.622, -0.0279, 2.4674],
        [0.1103, 3.5012, -12.8056],
        [-0.1222, 8.2168, 9.2725],
    ]
)

print(np.linalg.inv(M_matrix))