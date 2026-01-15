# 使用说明
## 1. 文件介绍
```bash
.
├── analysis_log.py   # 日志分析工具
├── config.yaml    # 配置文件
├── create_gt.py 
├── create_sim.py
├── data         # 固定数据文件夹，不建议修改
├── framework
├── main.py     # 主程序
├── modules
├── README.md    # 说明文档
├── record_mqtt.py
├── requirements.txt
├── simulated_data
├── tools   # 工具文件夹
├── tracker  
└── 使用说明.md
```
## 2. 运行步骤
### 2.1 生成gt数据
```bash
python create_gt.py 
```
### 2.2 生成模拟数据
```bash
python create_sim.py
```
### 2.3 运行目标定位程序
```bash
mosquitto
python mqtt_publish.py
python main.py
```


The main structure of the codes (such as ex1new_lazer.py) is identical. Different scenarios can be simulated by switching the input files map.json and plane.json. Noise added to the ProjectionMatrix is used to simulate intrinsic parameter perturbations, while variations in screen_x and screen_y are used to model detection errors. In addition, choosing either midpoints or rightpoint in
"calculate_and_plot_3d_rmse(midpoints, gt)"allows the simulation of LiDAR-based versus stereo-based reconstruction results.