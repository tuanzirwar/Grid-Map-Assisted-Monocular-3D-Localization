from framework import Package
from framework import Tracker

from tracker import Sort
import numpy as np

from multiprocessing import Queue
import time


def cal_iou(box1, box2):
    """
    :param box1: = [x1, y1, w1, h1]
    :param box2: = [x2, y2, w2, h2]
    :return: 
    """
    xmin1, ymin1, xmax1, ymax1 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    xmin2, ymin2, xmax2, ymax2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
    # 计算每个矩形的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # b1的面积
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # b2的面积
 
    # 计算相交矩形
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
 
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    a1 = w * h  # C∩G的面积
    a2 = s1 + s2 - a1
    iou = a1 / a2 #iou = a1/ (s1 + s2 - a1)
    return iou

def packages_nms(packages:list[Package], thresh):

    class_map= {}
    aimed = [1] * len(packages)
    for idx,p in enumerate(packages):
        if p.class_id not in class_map:
           class_map[p.class_id ]=[idx]
        else:
            class_map[p.class_id].append(idx)

    for k,v in class_map.items():
        for i in range(len(v)):
            for j in range(i+1,len(v)):
                if cal_iou(packages[v[i]].Bbox,packages[v[j]].Bbox)>thresh and aimed[v[i]]==1:
                    aimed[v[j]]= -1

    r_data = []
    for idx,mask in enumerate(aimed):
        if mask==-1:
            continue
        r_data.append(packages[idx])   
                  
    return r_data

class MOTracker(Tracker):
    def __init__(self, max_age=10, min_hits=3, distance_threshold=4, max_queue_length=None):
        super().__init__("MOTracker", max_queue_length)
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.tracker = {}

    def new_tracker(self):
        return Sort(max_age=self.max_age, min_hits=self.min_hits, distance_threshold=self.distance_threshold)

    def process(self, data: list[Package]):

        point_map = {}
        for package in data:
            if package.uav_id not in self.tracker.keys():
                self.tracker[package.uav_id] = self.new_tracker()
            if package.uav_id not in point_map.keys():
                point_map[package.uav_id] = []
            point_map[package.uav_id].append(package)

        for uav_id in point_map.keys():
            point_map[uav_id] = packages_nms(point_map[uav_id],0.3)
            
        for uav_id, packages in point_map.items():
            points = []
            for p in packages:
                points.append([*p.uav_utm, p.class_id])
            points = np.array(points).reshape(-1, 4)
            ret = self.tracker[uav_id].update(points)
            for i, t in enumerate(ret):
                point_map[uav_id][i].tracker_id = t[4]

        return_data = []
        for _, v in point_map.items():
            return_data.extend(v)
        return return_data

    def run_by_process(self, q_in: Queue, q_out: Queue):
        while True:
            if q_in.empty():
                continue
            data = q_in.get()
            if data == "exit":
                q_out.put("exit")
                break
            if len(data) == 0:
                continue

            packages = self.process(data)

            q_out.put(packages)
        print(f"{self.name} exit")
