from framework import Package,TimePriorityQueue
from framework import Sink
from tools import UWConvert

import time
from datetime import datetime
import copy
from multiprocessing import Queue

class PrintSink(Sink):
    def __init__(self, time_freq=5, offset=None):
        super().__init__("print_sink")
        self.convert = None
        self.process_queue = TimePriorityQueue()
        if offset:
            self.convert = UWConvert(offset)
        self.buffer = []
        self.last_time = int(time.time() * 1000)
        self.time_interval = int(1000.0 / time_freq)  # ms

    def close(self):
        # 对齐操作
        pass

    def process(self, data: list[Package]):

        send_data = {}
        send_data["timestamp"] = data[0].time
        send_data["obj_cnt"] = len(data)
        send_data["objs"] = []

        print(f"convert start {time.time()}:")
        for obj in data:
            if self.convert:
                
                obj.location = self.convert.U2W(obj.location)
            send_data["objs"].append({"id": obj.global_id, "cls": obj.class_id,
                                      "gis": obj.location, "bbox": obj.norm_Bbox,
                                      "obj_img": f"http://192.168.31.210:9002/detect/{obj.obj_img}.jpg" if obj.obj_img else "null"})
        print(f"convert finished {time.time()}:")
        self.buffer.append(send_data)

        now_time = int(time.time() * 1000)
        if now_time - self.last_time >= self.time_interval:
            print(
                f'\033[92m{datetime.fromtimestamp(now_time/1000.).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}\033[0m->', end="")
            print(self.buffer[0])
            if len(self.buffer) > 1:
                print(len(self.buffer))
                self.buffer.pop()
            self.last_time = now_time

    def run(self):
        while True:
            self.input_lock.acquire()
            if self.input_queue.is_empty():
                self.input_lock.release()
                while len(self.buffer) != 0:
                    now_time = int(time.time() * 1000)
                    if now_time - self.last_time >= self.time_interval:
                        print(
                            f'\033[92m{datetime.fromtimestamp(now_time/1000.).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}\033[0m->', end="")
                        print(self.buffer.pop())
                        self.last_time = now_time
                continue

            package = self.input_queue.pop()
            self.input_lock.release()

            self.process(package)

    def run_by_process(self, q_in: Queue, q_out: Queue):
        while True:
            # print("q_in size:", q_in.qsize())
            if q_in.empty():
                continue
            data = q_in.get()
            if data == "exit":
                break
            if len(data) == 0:
                continue
            for package in data:
                self.process_queue.push(package)
            
            if self.process_queue.is_empty() or self.process_queue.delta_time() < 1:
                continue
            packages = self.process_queue.get_time_slice(0)
            self.process(packages)
        print(f"{self.name} exit")