import trimesh
from tqdm import tqdm
import numpy as np
import mesh_raycast
import threading
import pickle
import os 

class MultiMesh:
    def __init__(self, multi_num, downSampled_scale, split_scale, block_num=1, overlap_x_scale=0.1, overlap_y_scale=0.1, default_height=500):
        '''
        @abstract: 初始化multiMesh类
        @param multi_num: int, 多层网格数，例如2表示有两层网格
        @param downSampled_scale: list, 下采样比例, 0-1之间的浮点数, 例如[0.5, 0.1]表示第一层下采样50%, 第二层下采样10%
        @param split_scale: list, 切分比例, 例如[[10,10],[5,5]]表示第一层切分网格为10*10块，第二层切分网格为5*5块
        @param block_num: int, 查找失败是，周围网格查找的范围
        @param overlap_x_scale: float, x方向重叠比例
        @param overlap_y_scale: float, y方向重叠比例
        @param default_height: float, 默认高度
        '''
        self.multi_num = multi_num # [2]
        self.downSampled_scale = downSampled_scale # [0.5, 0.1]
        self.split_scale = split_scale # [[10,10],[5,5]]
        self.overlap_x_scale = overlap_x_scale
        self.overlap_y_scale = overlap_y_scale
        self.over_block = block_num
        self.default_height = 500
        # self.thread_num = 8
        self.multi_meshes = None

    def read_mesh(self, mesh_path):
        '''
        @abstract: 读取obj文件，返回三角网格对象、三角网格顶点坐标、顶点坐标、面片索引、地图边界
        @param mesh_path: str, obj文件路径
        @return mesh: trimesh.base.Trimesh, 三角网格对象
        @return triangles: np.array, 三角网格顶点坐标
        @return vertices: np.array, 顶点坐标
        @return faces: np.array, 面片索引
        @return bbox: np.array, 地图边界
        '''
        assert mesh_path.endswith('.obj'), "Incorrect file format. Please check it!"
        assert os.path.exists(mesh_path), "File not found. Please check it!"
        mesh = trimesh.load_mesh(mesh_path)
        
        assert type(mesh) in [trimesh.scene.scene.Scene,trimesh.base.Trimesh],"Incorrect mesh type. please check it!"
        if type(mesh) == trimesh.scene.scene.Scene:
            all_meshes = [geom for geom in mesh.geometry.values()]
            mesh = trimesh.util.concatenate(all_meshes)
        vertices = mesh.vertices
        faces = mesh.faces
        triangles = vertices[faces]
        triangles = np.array(triangles, dtype='f4')
        bbox = mesh.bounds
        return mesh, triangles, vertices, faces, bbox
    
    def Mesh_scene2base(self, mesh_path, output_path):
        '''
        @abstract: 可以将Scene对象转换为标准的Trimesh对象，并将其写入单个文件
        @param mesh_path: str, obj文件路径
        @param output_path: str, 输出文件路径
        '''
        scene_mesh = trimesh.load_mesh(mesh_path)
        assert isinstance(scene_mesh, trimesh.scene.scene.Scene), "Incorrect mesh type. Please check it!"
        combined_mesh = trimesh.Trimesh()
        for submesh in scene_mesh.geometry.values():
            combined_mesh = trimesh.util.concatenate([combined_mesh, submesh])
        combined_mesh.export(output_path)
        
    def get_split_mesh(self, vertices, faces, target_x_min, target_x_max, target_y_min, target_y_max):
        # 筛选在指定范围内的顶点索引
        indices = np.where(
            (vertices[:, 0] >= target_x_min) & (vertices[:, 0] <= target_x_max) &
            (vertices[:, 1] >= target_y_min) & (vertices[:, 1] <= target_y_max)
        )[0]
        if len(indices) == 0:
        # 如果没有在指定范围内的顶点，则返回空数组
            return np.array([])

        # 根据筛选的顶点索引提取指定范围内的网格
        target_vertices = vertices[indices]
        # 更新面片的索引以匹配筛选后的顶点
        # 构建筛选后的面片数据
        target_faces = []
        for face in faces:
            if all(vertex_index in indices for vertex_index in face): # 只要有一个顶点在范围内，就保留这个面片
                target_faces.append([np.where(indices == vertex_index)[0][0] if vertex_index in indices else -1 for vertex_index in face])
        target_faces = [face for face in target_faces if -1 not in face]
        triangles = target_vertices[target_faces]
        triangles = np.array(triangles, dtype='f4')
        # target_mesh = trimesh.Trimesh(vertices=target_vertices, faces=target_faces)
        # return target_mesh
        return triangles

    def create_split_meshes(self, vertices, faces,bbox, split_x_num, split_y_num, overlap_x_scale, overlap_y_scale):
        '''
        @abstract: 将网格分割为多个网格
        @param vertices: np.array, 顶点坐标
        @param faces: np.array, 面片索引
        @param bbox: np.array, 地图边界
        @param split_x_num: int, x方向切分数
        @param split_y_num: int, y方向切分数
        @param overlap_x_scale: float, x方向重叠比例
        @param overlap_y_scale: float, y方向重叠比例
        @return meshes: list, 分割后的网格 (i, j): {'triangles': target_triangles,'bbox': [target_x_min, target_x_max, target_y_min, target_y_max]}
        '''
        x_min, y_min, z_min = bbox[0]
        x_max, y_max, z_max = bbox[1]
        x_step = (x_max - x_min) / split_x_num
        y_step = (y_max - y_min) / split_y_num
        overlap_x = x_step * overlap_x_scale
        overlap_y = y_step * overlap_y_scale
        meshes = {}
        for i in tqdm(range(split_x_num), desc="Processing rows"):
            for j in tqdm(range(split_y_num), desc="Processing columns", leave=False):
                target_x_min = max(x_min + i * x_step - overlap_x, x_min)
                target_x_max = min(x_min + (i+1) * x_step + overlap_x, x_max)
                target_y_min = max(y_min + j * y_step - overlap_y, y_min)
                target_y_max = min(y_min + (j+1) * y_step + overlap_y, y_max)
                # target_mesh = get_split_mesh(vertices, faces, target_x_min, target_x_max, target_y_min, target_y_max)
                target_triangle = self.get_split_mesh(vertices, faces, target_x_min, target_x_max, target_y_min, target_y_max)
                meshes[(i, j)] = {'triangles': np.array(target_triangle, dtype='f4'),'bbox': [target_x_min, target_x_max, target_y_min, target_y_max]}
                # print('target triangle shape:',target_triangle.shape, type(target_triangle), type(target_triangle[0][0][0]))
                # print('target bbox:',[target_x_min, target_x_max, target_y_min, target_y_max])
        return meshes

    def multi_threaded_construct_split_mesh(self, meshes, vertices, faces, overlap_x, overlap_y, i_js, x_ys):
        for i in range(len(i_js)):
            i, j = i_js[i]
            target_x_min, target_x_max, target_y_min, target_y_max = x_ys[i]
            print('constructing splitMesh:', i, j)
            target_triangle = self.get_split_mesh(vertices, faces, target_x_min, target_x_max, target_y_min, target_y_max)
            meshes[(i, j)] = {'triangles': np.array(target_triangle, dtype='f4'), 'bbox': [target_x_min, target_x_max, target_y_min, target_y_max]}

    def multi_threaded_construct_meshes(self,vertices, faces, bbox, split_x_num, split_y_num, overlap_x_scale, overlap_y_scale, num_threads=4):
        x_min, y_min, z_min = bbox[0]
        x_max, y_max, z_max = bbox[1]
        x_step = (x_max - x_min) / split_x_num
        y_step = (y_max - y_min) / split_y_num
        overlap_x = x_step * overlap_x_scale
        overlap_y = y_step * overlap_y_scale
        meshes = {}
        i_j = []
        x_y = []
        for i in range(split_x_num):
            for j in range(split_y_num):
                i_j.append((i, j))
                target_x_min = max(x_min + i * x_step - overlap_x, x_min)
                target_x_max = min(x_min + (i+1) * x_step + overlap_x, x_max)
                target_y_min = max(y_min + j * y_step - overlap_y, y_min)
                target_y_max = min(y_min + (j+1) * y_step + overlap_y, y_max)
                x_y.append((target_x_min, target_x_max, target_y_min, target_y_max))

        chunk_size = len(i_j) // num_threads
        i_js = []
        x_ys = []
        for i in range(0, num_threads-1):
            i_js.append(i_j[i * chunk_size: (i + 1) * chunk_size])
            x_ys.append(x_y[i * chunk_size: (i + 1) * chunk_size])
        i_js.append(i_j[(num_threads-1) * chunk_size:])
        x_ys.append(x_y[(num_threads-1) * chunk_size:])

        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=self.multi_threaded_construct_split_mesh, args=(meshes, vertices, faces, overlap_x, overlap_y, i_js[i], x_ys[i]))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return meshes

    def construct_multi_meshes(self, raw_mesh_path):
        '''
        @abstract: 构建多层三角网格
        @return multi_meshes: list, 多层三角网格
        '''
        multi_meshes = []
        now_mesh, now_triangles, now_vertices, now_faces, raw_bbox = self.read_mesh(raw_mesh_path)
        raw_face_num = now_faces.shape[0]
        for i in range(self.multi_num):
            print('constructing multiMesh:',i)
            now_vertices, now_faces, raw_bbox = now_mesh.vertices, now_mesh.faces, now_mesh.bounds
            split_x_num, split_y_num = self.split_scale[i]
            split_meshes = self.create_split_meshes(now_vertices, now_faces, raw_bbox, split_x_num, split_y_num, self.overlap_x_scale, self.overlap_y_scale)
            # 在你的代码中调用多线程构建函数
            # split_meshes = self.multi_threaded_construct_meshes(now_vertices, now_faces, raw_bbox, split_x_num, split_y_num, self.overlap_x_scale, self.overlap_y_scale, num_threads=self.thread_num)
            multi_meshes.append(split_meshes)
            downSampled_scale = self.downSampled_scale[i]
            now_mesh =  now_mesh.simplify_quadric_decimation(int(raw_face_num*downSampled_scale))
        # print('now_mesh:',now_mesh.vertices.shape,now_mesh.faces.shape)
        multi_meshes.append(np.array(now_mesh.vertices[now_mesh.faces], dtype='f4'))
        return multi_meshes
           
    def write_multi_mesh_to_file(self,raw_mesh_path, multi_mesh_path):
        """
        将multi_mesh写入本地文件
        :param multi_mesh: multi_mesh对象
        :param file_path: 本地文件路径
        """
        self.multi_meshes = self.construct_multi_meshes(raw_mesh_path)
        with open(multi_mesh_path, 'wb') as f:
            pickle.dump(self.multi_meshes, f)


    def load_multi_mesh_from_file(self,multi_mesh_path):
        """
        从本地文件中读取并恢复multi_mesh
        :param file_path: 本地文件路径
        :return: multi_mesh对象
        """
        with open(multi_mesh_path, 'rb') as f:
            self.multi_meshes = pickle.load(f)

    def get_blocks(self, start, t, index, x, y, meshes):
        # 找到匹配的网格
        matching_block = None
        for key, value in meshes.items():
            x_min, x_max, y_min, y_max = value['bbox']
            if x_min <= x <= x_max and y_min <= y <= y_max:
                matching_block = value['triangles']
                x_index, y_index = key
                break
        if matching_block is None: # 没找到返回result=空列表
            return []
        else:
            # time1 = time.time()
            # print('xi matching block shape:',matching_block.shape, type(matching_block), type(matching_block[0][0][0]))
            result = mesh_raycast.raycast(start, t, matching_block)
            # time2 = time.time()
            # print('xi raycast time:',(time2-time1))
            if len(result) != 0: # 找到交点
                result = min(result, key=lambda x: x['distance'])['point']
            else:
                # 寻找周围的网格 TODO 
                split_x_num, split_y_num = self.split_scale[index]
                for i in range(max(0, x_index - self.over_block), min(split_x_num, x_index + self.over_block+1)):
                    for j in range(max(0, y_index - self.over_block), min(split_y_num, y_index + self.over_block+1)):
                        
                        if i == x_index and j == y_index:
                            continue
                        matching_block = meshes[(i, j)]['triangles']
                        # if np.array_equal(empty_array, np.array([])):
                        #     continue
                        # time1 = time.time()
                        # print('xi matching block shape:',matching_block.shape, type(matching_block), type(matching_block[0][0][0]))
                        result = mesh_raycast.raycast(start, t, matching_block)
                        # time2 = time.time()
                        # print('xi raycast time:',(time2-time1))
                        if len(result) != 0:
                            result = min(result, key=lambda x: x['distance'])['point']
                            return result
                        else:
                            continue

        return result

    def ray_cast(self, start, t):
        result = []
        # 最粗糙mesh
        # time1 = time.time()
        # print('cu block shape:',self.multi_meshes[-1].shape, type(self.multi_meshes[-1]), type(self.multi_meshes[-1][0][0][0]))
        result = mesh_raycast.raycast(start, t, self.multi_meshes[-1])
        if len(result) == 0:  # TODO
            return result
        else:
            result = min(result, key=lambda x: x['distance'])['point']
        # time2 = time.time()
        # print('cu raycast time:',(time2-time1))
        
        for i in range(self.multi_num-1, -1, -1): 
            result_i= self.get_blocks(start, t,i, result[0], result[1], self.multi_meshes[i])
            if(len(result_i) == 0):
                break
            else:
                result = result_i
        return result

