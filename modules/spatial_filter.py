from framework import Package, TimePriorityQueue
from framework import Filter
import numpy as np
from multiprocessing import Queue
import time

# global溯源表循环队列
class CircularQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.size = 0
        self.front = 0
        self.rear = -1
        self.max_global_id = -1

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == self.capacity

    def enqueue(self, item):
        if self.is_full():
            # 如果队列满了，删除最旧的元素
            self.dequeue()

        self.rear = (self.rear + 1) % self.capacity
        self.queue[self.rear] = item
        self.size += 1

    def dequeue(self):
        if self.is_empty():
            return None

        item = self.queue[self.front]
        self.queue[self.front] = None
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return item

    def display(self):  # TODO 需优化
        print("Current Queue: ", end="")
        index = self.front
        for _ in range(self.size):
            print()
            print(index, self.queue[index])
            index = (index + 1) % self.capacity
        print()

    # 获得最新数据
    def get_last_element(self):
        if self.is_empty():
            return None
        return self.queue[self.rear]
    # 获得第i新的数据

    def get_last_element_i(self, i):
        if self.is_empty() or i >= self.size or i < 0:
            return None
        index = (self.rear - i) % self.capacity
        return self.queue[index]


class SpatialFilter(Filter):
    def __init__(self, time_slice, distance_threshold=None, max_map=None, max_queue_length=None):
        super().__init__("SpatialFilter", time_slice, max_queue_length)
        self.global_history = self.create_history(max_map)
        self.distance_threshold = distance_threshold  # 超参
        self.process_queue = TimePriorityQueue()

    def create_history(self, max_number):
        global_history = CircularQueue(max_number)
        global_history.enqueue({})  # TODO 这个代码history为null时报错，所以先enqueue
        return global_history

    # 按照class_id划为列表，每个列表再按照uav_id分行
    def classify_classid_uav(self, packages: list[Package]):
        # 按照class_id拆分列表
        from collections import defaultdict
        group_cls_id = defaultdict(list)
        for package in packages:
            group_cls_id[package.class_id].append(package)
        class_list = list(group_cls_id.values())

        # 每个类别对应的列表用uav_id拆分
        for i in range(len(class_list)):
            group_uav_id = defaultdict(list)
            for package in class_list[i]:
                group_uav_id[package.uav_id].append(package)
            class_list[i] = list(group_uav_id.values())

        return class_list

    # 空间滤波1:赋值local_id(将不同相机间距离相近的点视为同一空间点，使用相同local_id)
    def Spatial_filter1(self, distance_threshold, detections_list, local_id=0):
        '''
        注意:相机内的local_id不同，两个相机间同个local_id最多有两个点
        输入:distance_threshold, detections_list (距离阈值，观测数据)
        输出:具有local_id的detections_list
        '''
        # 讲各个相机的观测数据进行local_id，如果相机间观测的位置很接近，认为同一目标。
        for i in range(len(detections_list)):  # uav_id
            list_i = detections_list[i]
            for j in range(i+1, len(detections_list)):  # uav_id+1
                list_j = detections_list[j]
                # 创建两个列表的距离矩阵，并初始化为最大值
                matrix_distance = np.full(
                    (len(list_i), len(list_j)), float('inf'))
                # 将距离小于阈值的赋值即可
                for index_i, child_list_i in enumerate(list_i):
                    if child_list_i.local_id is None:
                        child_list_i.local_id = local_id
                        local_id = local_id+1
                    # 找到与child_list_i距离最小的下标（list_j中）
                    for index_j, child_list_j in enumerate(list_j):
                        distance = np.linalg.norm(
                            np.array(child_list_i.location) - np.array(child_list_j.location))
                        if not (child_list_i.local_id is not None and child_list_j.local_id is not None):
                            if distance < distance_threshold:  # 如果距离小：更新距离和下标
                                matrix_distance[index_i, index_j] = distance

                # 找到矩阵中的最小值及其索引,找到共视点，更新local_id
                for i in range(min(len(list_i), len(list_j))):
                    min_index = np.argmin(matrix_distance)
                    # (a,b)说明list_i的第a个与list_j的第b个距离最近
                    min_index_2d = np.unravel_index(
                        min_index, matrix_distance.shape)
                    if (matrix_distance[min_index_2d]) > 1000.:
                        break
                    list_j[min_index_2d[1]
                           ].local_id = list_i[min_index_2d[0]].local_id
                    matrix_distance[min_index_2d[0], :] = float('inf')
                    matrix_distance[:, min_index_2d[1]] = float('inf')

        # 更新最后一个uav列表的local_id
        for child_list_i in detections_list[-1]:
            if child_list_i.local_id is None:
                child_list_i.local_id = local_id
                local_id = local_id+1

        # for i in range(len(detections_list)):
        #     for j in range(len(detections_list[i])):
        #         detect = detections_list[i][j]
        #         print(f"Time: {detect.time},uav_id:{detect.uav_id}, Track ID: {detect.track_id}, local_id:{detect.local_id},  uv:{detect.boundingbox}, point:{detect.pose_location}")
        #     print()

        return detections_list, local_id

    # 空间滤波2:根据local_id更新平均距离，同local_id会按照uav_id排序
    '''
    输入:detections_list (观测数据)
    输出:更新距离后的detections_list
    '''

    def Spatial_filter2(self, class_list):
        from collections import defaultdict
        # 拉成一维
        detections_list = []
        for i in range(len(class_list)):
            detections_list = detections_list + \
                [item for sublist in class_list[i] for item in sublist]
        # 这里与普通的字典不同，这里与原数据引用的是同一块，会同时改变
        grouped_detections = defaultdict(list)
        # 使用 defaultdict 初始化一个字典，键为 local_id，值为包含相同 local_id 的元素的列表
        for detect in detections_list:
            grouped_detections[detect.local_id].append(detect)

        # 更新距离: 遍历列表，累加相同 local_id 的坐标和计数,并求解平均距离
        for local_id, group in grouped_detections.items():
            # print(f"Local ID: {local_id}")
            sum_pose_location = np.array([0., 0., 0.])
            for detect in group:
                sum_pose_location += np.array(detect.location).reshape(3,)
            aver_pose_location = sum_pose_location / len(group)
            for detect in group:
                detect.location = aver_pose_location.tolist()
            group = sorted(group, key=lambda x: x.uav_id)

        #     for detect in group:
        #         print(f"Time: {detect.time}, uav_id:{detect.uav_id},global_id:{detect.global_id}, Track ID: {detect.tracker_id}, local_id:{detect.local_id}, point:{detect.location}")
        # for detect in detections_list:
        #     print(f"Time: {detect.time}, uav_id:{detect.uav_id},global_id:{detect.global_id}, Track ID: {detect.track_id}, local_id:{detect.local_id},  uv:{detect.boundingbox}, point:{detect.pose_location}")
        return grouped_detections

    # 追溯global
    '''
    输入:grouped_detections, global_queue(空间过滤后的detection, global溯源表)
    输出:grouped_detections, global_queue(更新global后的detection, 更新后global溯源表)
    '''

    def find_global(self, grouped_detections, global_queue):
        my_dict = {}
        # 遍历新检测数据
        for local_id, group in grouped_detections.items():
            found = False
            # 遍历溯源表
            for i in range(0, global_queue.size):
                last_track = global_queue.get_last_element_i(i)

                for detect in group:  # 遍历同一个local_id
                    key = f"{detect.uav_id}_{detect.tracker_id}"
                    if key in last_track:  # 找到了
                        found = True
                        for new_detect in group:
                            # 更新global_id
                            new_detect.global_id = last_track[key]
                            # 更新global溯源表
                            new_key = f"{new_detect.uav_id}_{new_detect.tracker_id}"
                            my_dict[new_key] = last_track[key]
                        break  # 结束detect in group
                if found:  # 结束溯源表循环
                    break

            # 如果遍历完所有溯源表没找到
            if (not found):
                global_queue.max_global_id = global_queue.max_global_id + 1
                for detect in group:
                    detect.global_id = global_queue.max_global_id  # 更新global_id
                    # 更新global溯源表
                    new_key = f"{detect.uav_id}_{detect.tracker_id}"
                    my_dict[new_key] = detect.global_id

        global_queue.enqueue(my_dict)
        detections_list = []
        for group in grouped_detections.values():
            detections_list.extend(group)
        return detections_list, global_queue

    def process(self, packages: list[Package]):
        # 拆解list，便于后续操作
        class_list = self.classify_classid_uav(packages)

        # 赋值local_id
        local_id = 0
        for i in range(len(class_list)):
            if len(class_list[i]) != 0:
                class_list[i], local_id = self.Spatial_filter1(
                    self.distance_threshold, class_list[i], local_id=local_id)

        # 空间滤波2:根据local_id更新平均距离，同local_id会按照track_id排序
        group_list = self.Spatial_filter2(class_list)

        # global_id溯源
        return_data, self.global_history = self.find_global(
            group_list, self.global_history)
        # TODO: 滤波
        return return_data

    def run_by_process(self, q_in: Queue, q_out: Queue):
        while True:
            if q_in.empty():
                continue
            data = q_in.get()
            if data == "exit":
                q_out.put("exit")
                break
            self.process_queue.push(data)
            if self.process_queue.is_empty() or self.process_queue.delta_time() < self.time_slice + 1:
                continue
            packages = self.process_queue.get_time_slice(self.time_slice)
            out_packages = self.process(packages)
            while q_out.full():
                time.sleep(0.1)
            q_out.put(out_packages[:])
        print(f"{self.name} exit")
