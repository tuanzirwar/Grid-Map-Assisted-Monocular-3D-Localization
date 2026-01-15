from multiprocessing import Process, Queue
import time
import os
from framework import SpatialFilter
from modules import MQTTSource


offset = "data/map/JiuLongLake_v1223/offset.txt"
bbox_type = "xywh"
broker_url = "127.0.0.1"
port = 1883
client_id = "sub_camera_param"
qos = 2
topic_uav_sn = "thing/product/sn"
timeout = 30




def source(queue: Queue):
    mqtt = MQTTSource(offset, bbox_type, broker_url, port,
                  client_id, qos, topic_uav_sn, timeout)
    while True:
        data = mqtt.process()
        if data != []:
            queue.put(data)

def process(queue: Queue):
    while True:
        data = queue.get()
        for i in data:
            print(i.time)
            time.sleep(0.1)
            
def sink(queue: Queue):
    while True:
        data = queue.get()
        for i in data:
            print(i.time)

if __name__ == "__main__":
    queue = Queue(10000)
    
    source_process = Process(target=source, args=(queue,))
    sink_process = Process(target=sink, args=(queue,))
    source_process.start()
    sink_process.start()

    source_process.join()
    print("All processes completed")
