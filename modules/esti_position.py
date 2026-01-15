from framework import Package
from framework import Location
import mesh_raycast
from tools import get_ray, read_mesh, set_K, set_distortion_coeffs, set_camera_pose, UWConvert
from tools.multi_mesh import MultiMesh
import numpy as np
from multiprocessing import Queue
import time


class EstiPosition(Location):
    # enable  true:根据相机位姿定位目标点 false:根据道通uav_obj_pose重定位目标点
    def __init__(self, is_multi_map=False, mesh_path=None, default_height=60, order="rzyx", enable=True, max_queue_length=None, 
                 multi_num=2, downSampled_scale=[0.2, 0.1], split_scale=[[10, 10], [5, 5]], block_num=1, overlap_x_scale=0.1, overlap_y_scale=0.1):
        super().__init__("EstiPosition", max_queue_length)

        self.is_multi_map = is_multi_map
        self.order = order
        self.default_height = default_height
        self.enable = enable
        if is_multi_map:
            self.mesh = MultiMesh(multi_num, downSampled_scale, split_scale, block_num, overlap_x_scale, overlap_y_scale, default_height)
            self.mesh.load_multi_mesh_from_file(mesh_path)
        else:
            self.mesh = read_mesh(mesh_path)  # mesh地图



    def get_point(self, data: Package):
        _, K_inv = set_K(data.camera_K)
        D = set_distortion_coeffs(data.camera_distortion)

        # TODO: 可能R逆 是
        R, t, _ = set_camera_pose(data.camera_pose, order=self.order)
        ray = -get_ray(data.get_center_point(), K_inv, D, R)


        if self.is_multi_map:
            result = self.mesh.ray_cast( t.flatten(), ray)
            if len(result) == 0:  # TODO
                l = (self.default_height - t) / -ray[2]
                # 计算交点坐标
                inter_point = t - l * ray.reshape(3, 1)
                return inter_point.flatten().tolist()
            return result
        else:
            result = mesh_raycast.raycast(t.flatten(), ray, self.mesh)
            if len(result) == 0:  # TODO
                l = (self.default_height - t) / -ray[2]
                # 计算交点坐标
                inter_point = t - l * ray.reshape(3, 1)
                return inter_point.flatten().tolist()
            else:
                return min(result, key=lambda x: x['distance'])[
                    'point']

    def get_point_form_uav_object_point(self, data: Package):
        p_camera = np.array(data.camera_pose[3:]).reshape(3, 1)
        p_obj = np.array([data.uav_utm]).reshape(3, 1)
        ray = p_obj - p_camera
        ray = ray / np.linalg.norm(ray)

        if self.is_multi_map:
            result = self.mesh.ray_cast(p_camera.flatten(), ray)
            if len(result) == 0:  # TODO
                l = (self.default_height - p_camera) / -ray[2]
                # 计算交点坐标
                inter_point = p_camera - l * ray.reshape(3, 1)
                return inter_point.flatten().tolist()
            return result

        else:
            result = mesh_raycast.raycast(p_camera.flatten(), ray, self.mesh)
            if len(result) == 0:  # TODO
                l = (self.default_height - p_camera) / -ray[2]
                # 计算交点坐标
                inter_point = p_camera - l * ray.reshape(3, 1)
                return inter_point.flatten().tolist()
            else:
                return min(result, key=lambda x: x['distance'])[
                    'point']


    def process(self, data: Package):
        # data.location = self.get_point(
        #     data) if self.enable else self.get_point_form_uav_object_point(data)
        # if data.location[2] == self.default_height:
        #     data.location = data.uav_utm
        # if abs(data.location[0]) > 1000:
        #     print("data.camera_pose:",data.camera_pose[3:])
        #     print("data.uav_utm:",data.uav_utm[:3])
        #     print("data.location:",data.location[:3])
        #     print("data.time:",data.time)
        data.location = data.uav_utm

    def run_by_process(self, q_in: Queue, q_out: Queue):
        while True:
            if q_in.empty():
                continue
            data = q_in.get()

            if data == "exit":
                q_out.put("exit")
                break
            self.process(data)

            while q_out.full():
                time.sleep(0.1)
                
            q_out.put(data)
        print(f"{self.name} exit")
