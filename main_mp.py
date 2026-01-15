import yaml
import sys
import os
from keyboard import add_hotkey

from framework.pipeline_mp import Pipeline_mp


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

    pipe = Pipeline_mp(pipelines)

    # add_hotkey("esc", pipe.stop)
    # add_hotkey("q", pipe.stop)
    # add_hotkey(config["global"]["exit_key"], pipe.stop)

    pipe.run()
