
from tools.multi_mesh import MultiMesh
import yaml
import time

obj_map = "data/map/yumen/Tile_geo.obj"
pkl_map = "data/map/yumen/Tile_geo.pkl"

defualt_cfg = "config.yaml"
with open(defualt_cfg, 'r', encoding="utf-8") as yaml_file:
    config = yaml.safe_load(yaml_file)
multi_map_pramas = config['pipeline']['stage4']['args']


time1 = time.time()
multi_mesh = MultiMesh(multi_num=multi_map_pramas['multi_num'],
                       downSampled_scale=multi_map_pramas['downSampled_scale'],
                       split_scale=multi_map_pramas['split_scale'],
                       block_num=multi_map_pramas['block_num'],
                       overlap_x_scale=multi_map_pramas['overlap_x_scale'],
                       overlap_y_scale=multi_map_pramas['overlap_y_scale'],
                       default_height=multi_map_pramas['default_height'])
time2 = time.time() 
print('create class time:',(time2-time1))
time1 = time.time()
multi_mesh.write_multi_mesh_to_file(raw_mesh_path=obj_map, multi_mesh_path=pkl_map)
time2 = time.time()
print('write time:',(time2-time1))
time1 = time.time()
multi_mesh.load_multi_mesh_from_file(pkl_map)
time2 = time.time()
print('load time:',(time2-time1))