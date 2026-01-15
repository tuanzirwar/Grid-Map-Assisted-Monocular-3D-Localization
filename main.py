import yaml
import sys
import os

from framework.pipeline import Pipeline
import signal

from modules import JsonSource, UDPSource, MQTTSource, MQTTLogSource
from modules import EstiPosition
from modules import MOTracker
from modules import TimeFilter
from modules import SpatialFilter
from modules import UnitySink, PrintSink, HttpSink


if __name__ == "__main__":
    # 读取YAML配置文件
    defualt_cfg = "config.yaml"
    try:
        if os.path.exists(sys.argv[1]):
            defualt_cfg = sys.argv[1]
        else:
            raise FileNotFoundError
    except:
        print(f"---Using default config {defualt_cfg}---")

    with open(defualt_cfg, 'r', encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)

    pipelines = []
    for stage in config["pipeline"].values():
        module = eval(stage["name"])
        pipelines.append([
            module(*stage["args"].values()) if stage["args"] else module() for _ in range(stage["parallel"])])

    pipe = Pipeline(pipelines)
    pipe.run()

    def signal_handler(signum, frame):
        print(
            "Ctrl+C detected. Closing socket. Exiting...")
        pipelines[0].close()
        pipelines[-1].close()
        exit()

    signal.signal(signal.SIGINT, signal_handler)
