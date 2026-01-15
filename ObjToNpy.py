from typing import Union
import trimesh
import numpy as np
from enum import Enum
import math
import transforms3d as tfs

def read_mesh(mesh_path, save_path="data/mesh_triangles.npy"):
    mesh = trimesh.load_mesh(mesh_path) #load or load_mesh
    assert type(mesh) in [trimesh.scene.scene.Scene,trimesh.base.Trimesh],"Incorrect mesh type. please check it!"
    
    if type(mesh) == trimesh.scene.scene.Scene:
        all_meshes = [geom for geom in mesh.geometry.values()]
        mesh = trimesh.util.concatenate(all_meshes)
        
    vertices = mesh.vertices
    faces = mesh.faces
    triangles = vertices[faces]
    triangles = np.array(triangles, dtype='f4') 

    # 保存三角形网格数据到npy文件
    np.save(save_path, triangles)
    return triangles

if __name__ == '__main__':
    read_mesh("/home/xjy/code/location_Map/ObjectLocation/data/map/JiuLongLake_1223.obj")