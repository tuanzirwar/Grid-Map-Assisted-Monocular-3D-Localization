from tools import MqttClient, Log
import signal
import json


LOG_ROOT = "./log"

broker_url = "192.168.31.38"
port = 1883
client_id = 'record_camera_param'
qos = 2
topic_uav_sn = "uav/product/syn_device_sn"
timeout = 30


my_log = Log(LOG_ROOT, enable=True, eveytime=True)


def print_writeonce(msg):
    print(msg)

def print_writevertime(msg):
    try:
        msg_dict = json.loads(msg)
        time = msg_dict["timestamp"]
        data = msg_dict["data"]
        print(time, data["obj_cnt"])
    except:
        print(msg)


my_log.log_show(print_writevertime)

# 创建MQTT客户端实例,不记录日志
client = MqttClient(broker_url=broker_url,
                    port=port,
                    client_id=client_id,
                    qos=qos,
                    topic_uav_sn=topic_uav_sn,
                    timeout=timeout,
                    log=my_log)

client.start()


def signal_handler(signum, frame):
    print(
        "Ctrl+C detected. Closing socket. Exiting...")
    client.close()

    exit()


signal.signal(signal.SIGINT, signal_handler)

while True:
    client.get_data()
