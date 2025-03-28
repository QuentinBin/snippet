'''
Description: None
Author: Bin Peng
Email: pb20020816@163.com
Date: 2025-03-20 20:17:55
LastEditTime: 2025-03-20 20:18:00
'''
import numpy as np
import trimesh
import random
import collada
import noise

def generate_terrain(size, scale, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    x = np.linspace(-scale, scale, size)
    y = np.linspace(-scale, scale, size)
    X, Y = np.meshgrid(x, y)
    
    height = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            height[i, j] = noise.pnoise2(i / 10.0, j / 10.0, octaves=6, persistence=0.5, lacunarity=2.0, repeatx=size, repeaty=size, base=seed) * 20
    
    vertices = []
    faces = []
    
    for i in range(size - 1):
        for j in range(size - 1):
            v1 = (x[i], y[j], height[i, j])
            v2 = (x[i + 1], y[j], height[i + 1, j])
            v3 = (x[i], y[j + 1], height[i, j + 1])
            v4 = (x[i + 1], y[j + 1], height[i + 1, j + 1])
            
            idx1, idx2, idx3 = len(vertices), len(vertices) + 1, len(vertices) + 2
            idx4 = len(vertices) + 3
            
            vertices.extend([v1, v2, v3, v4])
            
            faces.append((idx1, idx2, idx3))
            faces.append((idx2, idx4, idx3))
    
    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)

def export_to_dae(vertices, faces, filename="terrain.dae"):
    mesh = collada.Collada()
    
    vert_src = collada.source.FloatSource("verts-array", vertices.flatten(), ('X', 'Y', 'Z'))
    geom = collada.geometry.Geometry(mesh, "geometry0", "mygeom", [vert_src])
    
    indices = faces.flatten()
    input_list = collada.source.InputList()
    input_list.addInput(0, 'VERTEX', "#verts-array")
    
    polylist = geom.createTriangleSet(indices, input_list, "materialref")
    geom.primitives.append(polylist)
    
    mesh.geometries.append(geom)
    
    scene = collada.scene.Scene("myscene", [collada.scene.GeometryNode(geom)])
    mesh.scenes.append(scene)
    mesh.scene = scene
    
    mesh.write(filename)
    print(f"Exported terrain to {filename}")

# 生成地形并导出
t_size = 50  # 网格大小
t_scale = 50  # 地形范围
t_seed = 42   # 随机种子

vertices, faces = generate_terrain(t_size, t_scale, t_seed)
export_to_dae(vertices, faces)
