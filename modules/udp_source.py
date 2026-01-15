from framework import Package
from framework import Source
import socket
import sys


class UDPSource(Source):
    def __init__(self, ip="127.0.0.1", port=8888):
        super().__init__("udp_source")
        self.addr = (ip, port)
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(self.addr)
        except:
            self.sock.close()
            print(f"\033[91mCan't open socket at {self.addr}") 
            sys.exit(-1)
            
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

    def close(self):
        self.sock.close()

    def process(self, packages: list[Package]):
        try:
            data, address = self.sock.recvfrom(1024)
            # 处理接收到的数据
        except socket.timeout as e:
            return
        except socket.error as e:
            print(e)
            raise
        try:
            objs = eval(data.decode("utf-8"))
            if isinstance(objs, dict):
                raise TypeError
        except:
            print(f"can't convert {data.decode('utf-8')} to dict,plase check")
            return False

        for obj in objs["objs"]:
            bbox = Package()
            bbox.time = objs["Time"]
            bbox.uav_id = objs["Fly_id"]
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
