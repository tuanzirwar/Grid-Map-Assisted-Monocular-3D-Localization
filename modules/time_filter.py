from framework import Package
from framework import PreProcess
from multiprocessing import Queue
from framework import TimePriorityQueue
import time


class TimeFilter(PreProcess):
    def __init__(self, time_slice, max_queue_length=None):
        super().__init__("TimeFilter", time_slice, max_queue_length)
        self.process_queue = TimePriorityQueue()
        self.image_map = {}

    def process(self, data: list[Package]):
        return_data = []  # 需要返回的列表
        data_map = {}  # 用于存储数据的字典，按无人机id分类
        
        #1.  存link 和  id map
        #2. 降采样
        #2.1  找时间戳相同的包，max（len（these）） 
        #3. find 1.map 填包, 已处理的删除
        #   

        for package in reversed(data):
            map_key = f"{package.uav_id}_{package.tracker_id}"
            if map_key not in data_map.keys():
                data_map[map_key] = package
            else:
                if data_map[map_key].time < package.time:
                    if data_map[map_key].obj_img != None:
                        package.obj_img = data_map[map_key].obj_img
                    data_map[map_key] = package

        for package in data_map.values():
            return_data.append(package)

        return return_data

    def run_by_process(self, q_in: Queue, q_out: Queue):
        while True:
            if q_in.empty():
                continue
            data = q_in.get()
            if data == "exit":
                q_out.put("exit")
                break
            for package in data:
                self.process_queue.push(package)

            if self.process_queue.is_empty() or self.process_queue.delta_time() < self.time_slice + 1:
                continue

            packages = self.process_queue.get_time_slice(self.time_slice)

            out_packages = self.process(packages)

            for p in out_packages:
                while q_out.full():
                    time.sleep(0.1)
                q_out.put(p)
                
        print(f"{self.name} exit")
        return 
