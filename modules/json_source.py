from framework import Package
from framework import Source
import json


class JsonSource(Source):
    def __init__(self, json_file,bbox_type="cxcywh"):
        super().__init__("json_source")
        self.bbox_type = bbox_type
        self.json_file = json_file
        self.data = self.read_json(self.json_file)

    def read_json(self, json_file):
        with open(json_file, "r") as f:
            json_data = json.load(f)
        return json_data["data"]

    def close(self):
        # 对齐操作
        pass

    def parse_c_pose(self, c_pose: dict):
        pose = []
        pose.append(c_pose["Yaw"])
        pose.append(c_pose["Pitch"])
        pose.append(c_pose["Roll"])
        pose.append(c_pose["longitude"])
        pose.append(c_pose["latitude"])
        pose.append(c_pose["height"])
        return pose

    def parse_K_D(self, params: dict):
        focal_len = params["focal_length"]["value"]
        img_w, img_h = params["zoom"]["img_size"]
        sensor_w, sensor_h = params["zoom"]["sensor_size"]
        fx = focal_len * img_w / sensor_w
        fy = focal_len * img_h / sensor_h
        cx = img_w / 2
        cy = img_h / 2
        return [fx, fy, cx, cy], params["distortion_correction_params"]["distortion"]

    def process(self, packages: list[Package]):
        if len(self.data) == 0:
            return False
        objs = self.data.pop(0)
        bbox = Package()
        bbox.bbox_type = self.bbox_type
        for obj in objs["objs"]:
            bbox.time = objs["Time"]
            bbox.uav_id = "QUHJ_6387"+str(objs["Fly_id"])
            bbox.camera_id = objs["Camera_id"]
            bbox.camera_pose = self.parse_c_pose(objs["Camera"])
            bbox.camera_K, bbox.camera_distortion = self.parse_K_D(
                objs["Camera"]["intrinsic_params"])
            bbox.Bbox = obj["bbox"]
            bbox.uid = obj["uid"]  # for evaluation
            bbox.class_id = obj["cls_id"]
            bbox.tracker_id = obj["tracker_id"]
            bbox.uav_pos = obj["loc"]
            packages.append(bbox.copy())

        return True
