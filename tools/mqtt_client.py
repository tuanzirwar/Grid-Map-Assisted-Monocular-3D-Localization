from paho.mqtt import client
from .record_log import Log
import time
import json
from data import UAV_LISTS

class MqttClient:
    def __init__(self,
                #  broker_url="192.168.31.158",
                 broker_url ="127.0.0.1",
                 port=1883,
                 client_id='sub_camera_param',
                 qos=2,
                 topic_uav_sn="thing/product/sn",
                 timeout=30,
                 log: Log = Log("",enable=False)):
    
        self.broker_url = broker_url
        self.client_id = client_id
        self.port = port
        self.topic_uav_sn = topic_uav_sn

        self.qos = qos

        self.timeout = timeout   # second

        self.topic_map = {}  # key: sn  value: topic

        self.buffer = []

        self.log: Log = log

        # 创建MQTT客户端实例
        self.client = client.Client(self.client_id)
        self.client.on_connect = self.on_connect
        self.client.connect(self.broker_url, self.port)
        # 设置回调函数，用于处理消息接收事件
        self.client.on_message = self.on_message
        # 开始循环订阅
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc):
        age = 0
        while rc != 0:
            print("%dth connect error, return code %d\n", age, rc)
            age += 1
            if age > self.timeout:
                raise TimeoutError("Max retries exceeded")

        print("Connected to MQTT Broker!\n")
        # 订阅话题接受
        self.client.subscribe(self.topic_uav_sn, qos=self.qos)
        for topic in self.topic_map.values():
            self.client.subscribe(topic, qos=self.qos)
            # log
            print(f"MQTT: resubscribe {topic} success!")

        sn = "TH7923461372"
        topic = f"thing/product/{sn}/state"
        self.client.subscribe(topic, qos=self.qos)
        self.topic_map[sn] = topic

    def on_message(self, client, userdata, msg):
        if msg.topic == self.topic_uav_sn:
            json_data = eval(msg.payload.decode())
            for i in json_data:
                sn_from_mqtt = i["gatewaySn"]
                if sn_from_mqtt not in self.topic_map:
                    topic = f"thing/product/{sn_from_mqtt}/state"
                    self.topic_map[sn_from_mqtt] = topic
                    self.client.subscribe(topic, qos=self.qos)
                    # log
                    print(f"MQTT: subscribe {sn_from_mqtt} success!")

        elif msg.topic in self.topic_map.values():
            payload =msg.payload.decode()
            self.log.record(payload)
            try:
                data = json.loads(payload)
                self.buffer.append(data)
            except:
                print("PAYLOAD ERROR")

    def publish(self, topic, payload):
        self.client.publish(topic, payload, qos=self.qos)

    def start(self):
        times_count = 0
        while times_count < self.timeout:
            try:
                self.client.loop_start()
                return True
            except:
                print(f"MQTT: reconnect {times_count} times!")
                self.client.reconnect()
                times_count += 1
            time.sleep(1)
        raise TimeoutError("Max retries exceeded")

    def get_data(self):
        while len(self.buffer) == 0:
            time.sleep(0.1)
        return self.buffer.pop(0)

    def close(self):
        self.client.loop_stop()
        self.client.disconnect()
        print("MQTT: connection close success!")
        self.log.close_record()
        print("MQTT: log close success!")
        self.buffer = ["exit"]
