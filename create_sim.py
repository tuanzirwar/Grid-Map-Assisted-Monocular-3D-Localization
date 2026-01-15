import json
import random
from tools.utils import PointType, SimulationCamera, SimulationObject
import argparse
import copy
from tqdm import tqdm

data_json = {
    "Time": None,
    "Fly_id": None,
    "Camera_id": None,
    "Camera": {
    },
    "obj_cnt": None,
    "objs": [],
}
camera_json = {
    "Yaw": 0.0,
    "Pitch": 0.0,
    "Roll": 0.0,
    "longitude": 0,
    "latitude": 0,
    "height": 0,
    "intrinsic_params": {
        "focal_length": {
            "value": 0, "unit": "mm"
        },
        "zoom": {
            "is_zoom_supported": False,
            "img_size": [],
            "sensor_size": []
        },
        "distortion_correction_params": {
            "distortion": [0, 0, 0, 0, 0]
        }
    }
}
obj_json = {
    "uid": None,  # for evaluation
    "tracker_id": None,
    "cls_id": None,
    "bbox": [],
    "loc": None,
    "obj_img": None
}

unity_obj = {
    "id": None,
    "point": []
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        '--mesh_path', default='data/map/JiuLongLake_1223.obj', type=str, help='Path of map mesh')
    parser.add_argument('--cameras_cfg', default="data/cameras_2.json", type=str,
                        help='Configuration file for camera parameters')
    parser.add_argument('--objects', default="simulated_data/objects.json", type=str,
                        help='Real points used for generating simulated data.')
    parser.add_argument('--nogen_unity',  action='store_true',
                        help='Whether to generate unity data,default: yes')
    parser.add_argument('--nogen_sim',  action='store_true',
                        help='Whether to generate sim data, default: yes')
    parser.add_argument('--unity_data', default='simulated_data/unity_data.json',
                        type=str, help='Path of unity_data')
    parser.add_argument('--sim_data', default='simulated_data/simulate_data.json',
                        type=str, help='Path of sim_data')
    args = parser.parse_args()

    unity_data = {
        "data": [],
        "camera_params": []
    }

    sim_data = {"data": []}

    # 根据配置文件解析相机参数
    with open(args.cameras_cfg, "r") as f:
        camera_param = json.load(f)

    K = camera_param["K"]  # 内参
    distortion = camera_param["D"]  # 畸变参数
    frame_rate = camera_param["fps"]  # 帧率
    shape = camera_param["shape"]  # 图像大小
    euler_type = camera_param["euler_type"]  # 欧拉角类型

    cameras = []
    for ext_param in camera_param["ext_param"]:
        pose = [ext_param["yaw"], ext_param["pitch"], ext_param["roll"],
                ext_param["x"], ext_param["y"], ext_param["z"]]
        cameras.append(SimulationCamera(
            pose, K, distortion, shape, mesh_path=args.mesh_path, euler=euler_type))

        # unity data  default: szxy
        u_camera = {
            "pose": ext_param,
        }
        u_camera["horizontal"], u_camera["vertical"] = cameras[-1].get_fov_angle()
        unity_data["camera_params"].append(u_camera)

    # 根据配置文件解析目标参数
    with open(args.objects, "r") as f:
        obj_data = json.load(f)

    duration = obj_data["duration"]

    objs = []
    for obj in obj_data["objects"]:
        objs.append(SimulationObject(
            obj, len(cameras), max_age=20, uid=obj["uid"]))

    start_timestamp = 1701482850000  # unix 时间戳 2023-12-02 10:07:30.000 ms 起始时间

    for i in tqdm(range(duration*frame_rate)):
        for uav_id, camera in enumerate(cameras):
            sim_package = copy.deepcopy(data_json)
            sim_package["Time"] = start_timestamp + \
                i*(1000/frame_rate) + random.randint(-10, 10)
            sim_package["Fly_id"] = uav_id
            sim_package["Camera_id"] = uav_id
            pose = camera.pose
            c_json = copy.deepcopy(camera_json)
            c_json["Yaw"] = pose[0]
            c_json["Pitch"] = pose[1]
            c_json["Roll"] = pose[2]
            c_json["longitude"] = pose[3]
            c_json["latitude"] = pose[4]
            c_json["height"] = pose[5]
            c_json["intrinsic_params"]["focal_length"]["value"] = camera_param["focal_length"]
            c_json["intrinsic_params"]["zoom"]["img_size"] = camera_param["shape"]
            c_json["intrinsic_params"]["zoom"]["sensor_size"] = camera_param["sensor_size"]
            c_json["intrinsic_params"]["distortion_correction_params"]["distortion"] = camera_param["D"]
            sim_package["Camera"] = c_json

            for idx in range(len(objs)):
                if objs[idx].age[uav_id] > objs[idx].max_age:
                    # 超过最大age，表示跟丢，重置tracker_id,重置age
                    objs[idx].tracker_id[uav_id] = -1
                    objs[idx].reset_age(uav_id)

                sim_obj_data = copy.deepcopy(obj_json)
                status, data = camera.get_bbox_result(
                    objs[idx].next_point(), objs[idx].get_bbox_size())

                if data[1]:
                    unity_d = copy.deepcopy(unity_obj)
                    unity_d["id"] = objs[idx].uid
                    unity_d["point"] = data[1]
                    unity_data["data"].append(unity_d)

                if status == PointType.ValidPoint:
                    if objs[idx].tracker_id[uav_id] == -1:
                        # 新目标, 赋值tracker_id
                        objs[idx].tracker_id[uav_id] = camera.get_max_id()
                        cameras[uav_id].max_id += 1

                    sim_obj_data["tracker_id"] = objs[idx].tracker_id[uav_id]
                    sim_obj_data["cls_id"] = objs[idx].get_clsid()
                    sim_obj_data["bbox"] = data[0]
                    sim_obj_data["loc"] = data[1]
                    sim_obj_data["uid"] = objs[idx].uid

                    sim_package["objs"].append(sim_obj_data)
                    objs[idx].reset_age(uav_id)  # 重置age
                else:
                    objs[idx].age[uav_id] += 1

            sim_package["obj_cnt"] = len(sim_package["objs"])
            if not sim_package["obj_cnt"] == 0:
                sim_data["data"].append(sim_package)

    if not args.nogen_unity:
        with open(args.unity_data, 'w') as outfile:
            json_data = json.dumps(unity_data)
            outfile.write(json_data)

    if not args.nogen_sim:
        with open(args.sim_data, 'w') as outfile:
            json_data = json.dumps(sim_data)
            outfile.write(json_data)
