'''
Description: None
Author: Bin Peng
Email: pb20020816@163.com
Date: 2024-11-24 21:44:50
LastEditTime: 2024-11-29 00:14:20
'''
import numpy as np
import matplotlib.pyplot as plt

# 椭球生成函数
def generate_ellipsoid(a, b, c, resolution=20):
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))
    
    # 将 x, y, z 展平为顶点坐标
    vertices = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    
    # 生成三角形面片
    triangles = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx1 = i * resolution + j
            idx2 = idx1 + 1
            idx3 = idx1 + resolution
            idx4 = idx3 + 1
            triangles.append([idx1, idx2, idx3])
            triangles.append([idx2, idx4, idx3])
    return vertices, np.array(triangles)

# 薄板生成函数
def generate_plate(x_len, y_len, z_len, resolution=10):
    x = np.linspace( 0, x_len , resolution)
    y = np.linspace(0, y_len , resolution)
    z = np.array([-z_len / 2, z_len / 2])
    xv, yv, zv = np.meshgrid(x, y, z, indexing="ij")
    vertices = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T
    
    # 面片连接（只考虑两面）
    triangles = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            idx1 = i * resolution + j
            idx2 = idx1 + 1
            idx3 = idx1 + resolution
            idx4 = idx3 + 1
            triangles.append([idx1, idx2, idx3])
            triangles.append([idx2, idx4, idx3])
    return vertices, np.array(triangles)

def compute_ellipsoid_inertia(mass, a, b, c):
    """
    Compute the inertia tensor for a uniform ellipsoid.
    
    Args:
        mass (float): Total mass of the ellipsoid.
        a (float): Semi-axis along x-axis.
        b (float): Semi-axis along y-axis.
        c (float): Semi-axis along z-axis.
    
    Returns:
        np.ndarray: 3x3 inertia tensor matrix.
    """
    I_xx = (1/5) * mass * (b**2 + c**2)
    I_yy = (1/5) * mass * (a**2 + c**2)
    I_zz = (1/5) * mass * (a**2 + b**2)
    return np.diag([I_xx, I_yy, I_zz])

def compute_plate_inertia(mass, length, width, thickness):
    """
    Compute the inertia tensor for a uniform rectangular plate.
    
    Args:
        mass (float): Total mass of the plate.
        length (float): Length of the plate (x-axis).
        width (float): Width of the plate (y-axis).
        thickness (float): Thickness of the plate (z-axis).
    
    Returns:
        np.ndarray: 3x3 inertia tensor matrix.
    """
    I_xx = (1/12) * mass * (width**2 + thickness**2)
    I_yy = (1/12) * mass * (length**2 + thickness**2)
    I_zz = (1/12) * mass * (length**2 + width**2)
    return np.diag([I_xx, I_yy, I_zz])

def skew_symmetric(v):
    """
    Construct the skew-symmetric matrix (cross-product matrix) for a vector v.
    
    Args:
        v (np.ndarray): A 3x1 vector.
    
    Returns:
        np.ndarray: The 3x3 skew-symmetric matrix.
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def adjoint_matrix(g):
    """
    Compute the adjoint matrix (Ad) of a SE(3) element g.
    
    Args:
        g (np.ndarray): A 4x4 matrix representing an element of SE(3), 
                         where g = [ R | p; 0 | 1]
                         R is a 3x3 rotation matrix, p is a 3x1 translation vector.
    
    Returns:
        np.ndarray: The 6x6 Adjoint matrix.
    """
    # Extract R (3x3) and p (3x1) from the 4x4 matrix g
    R = g[:3, :3]
    p = g[:3, 3]
    
    # Construct the skew-symmetric matrix S(p)
    S_p = skew_symmetric(p)
    
    # Construct the adjoint matrix Ad(g)
    Ad_g = np.block([[R, np.zeros((3, 3))],
                     [S_p @ R, R]])
    
    return Ad_g


