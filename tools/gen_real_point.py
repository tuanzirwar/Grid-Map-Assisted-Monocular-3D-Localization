import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from .utils import SimulationCamera, SimulationObject, PointType, read_mesh, get_real_point
from .gen_simulat_obj import get_attr
import random
from matplotlib.animation import FuncAnimation
import json


def plot_camera_fov(ax, t1, points, color):
    """
    在三维坐标轴上绘制相机的视场范围。

    参数:
    ax:matplotlib 三维坐标轴对象
    t1:相机位置的三维坐标,形如 [x, y, z]
    points:视场范围的四个点的三维坐标,形如 [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], [x4, y4, z4]]
    color:绘制的颜色

    """
    # 绘制四个点
    scope = np.array([*points, points[0]], dtype=np.float32).reshape(5, 3)
    ax.scatter(t1[0], t1[1], t1[2], c=color, marker='o')
    # 依次连接四个点
    for i in range(4):
        ax.plot([t1[0], scope[i][0]], [t1[1], scope[i][1]], [
                t1[2], scope[i][2]], c=color, linestyle='--')

    # 依次连接四个点
    ax.plot(scope[:, 0], scope[:, 1], scope[:, 2], c=color)

def plot_line(ax, line, color):
    line = np.array(line, dtype=np.float32)
    x = line[:, 0]
    y = line[:, 1]
    z = line[:, 2]
    ax.plot(x, y, z, c=color)


def generate_n_colors(n, type="hex"):
    """
    生成n种尽可能不相似的颜色。

    Args:
        n (int): 要生成的颜色数量。
        color_type (str): 颜色表示类型,可选值为"hex"、"rgb"或"normal rgb"。默认为"hex"。

    Returns:
        list: 包含n种不相似颜色的列表。
    """
    assert type in ["hex", "rgb",
                    "normal rgb"], "type must be hex, rgb, normal rgb"
    colors = []
    for _ in range(n):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)

        # 确保生成的颜色与已有的颜色差别较大
        while any(abs(r - color[0]) < 50 and abs(g - color[1]) < 50 and abs(b - color[2]) < 50 for color in colors):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
        colors.append((r, g, b))

    for i in range(len(colors)):
        if type == "hex":
            colors[i] = '#%02x%02x%02x' % colors[i]
        elif type == "rgb":
            colors[i] = (colors[i][0] / 255, colors[i][1] /
                         255, colors[i][2] / 255)
        elif type == "normal rgb":
            colors[i] = (colors[i][0] / 255.0, colors[i][1] /
                         255.0, colors[i][2] / 255.0)
    return colors


def gen_real_point(args):
    # 根据配置文件解析相机参数
    with open(args.cameras_cfg, "r") as f:
        camera_param = json.load(f)

    K = camera_param["K"]  # 内参
    distortion = camera_param["D"]  # 畸变参数
    frame_rate = camera_param["fps"]  # 帧率
    shape = camera_param["shape"]  # 图像大小
    euler_type = camera_param["euler_type"]  # 欧拉角类型

    cameras = []
    for ext_param in camera_param["ext_param"]:
        pose = [ext_param["yaw"], ext_param["pitch"], ext_param["roll"],
                ext_param["x"], ext_param["y"], ext_param["z"]]
        cameras.append(SimulationCamera(
            pose, K, distortion, shape, euler=euler_type))

    # 根据配置文件解析目标参数
    with open(args.objs_cfg, "r") as f:
        objs_param = json.load(f)

    objs = []
    objs_attr = {"objects": []}
    for idx, ponit in enumerate(objs_param["points"]):
        attr = get_attr(ponit["start_point"],
                        ponit["end_point"], frame_rate * args.duration)
        objs.append(SimulationObject(attr, uid=idx))
        attr["uid"] = idx
        objs_attr["objects"].append(attr)

    with open(args.obj_attr_output, "w") as f:
        objs_attr["duration"] = args.duration
        json.dump(objs_attr, f, indent=4)

    # 生成用于绘图的相机和目标的颜色
    colors = generate_n_colors(len(cameras)+len(objs))
    camera_colors = colors[:len(cameras)]
    obj_colors = colors[len(cameras):]

    # # 读取OBJ文件
    # mesh = read_mesh(args.mesh_path)

    # 读取OBJ文件
    triangles_path = "data/mesh_triangles.npy"  # 确保这个路径与read_mesh中的save_path一致
    _ = read_mesh(args.mesh_path, triangles_path)
    try:
        # 直接从npy文件读取三角形网格数据
        mesh = np.load(triangles_path)
        print("Mesh shape: ", mesh.shape)
        print("First 3x3 matrix (first triangle):\n", mesh[0])
        print("Last 3x3 matrix (last triangle):\n", mesh[-1])
    except FileNotFoundError:
        # 如果npy文件不存在，则从OBJ文件读取并保存
        mesh = read_mesh(args.mesh_path, triangles_path)
        print("No load from npy file successfully")

    # 画板
    fig = plt.figure(figsize=(19.20, 10.80), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # 绘制相机的视野范围
    for _, camera in enumerate(cameras):
        points = camera.get_fov_scope()
        plot_camera_fov(ax, camera.get_params()[
                        3], points, color=camera_colors[_])

    # 生成物体的真实运动轨迹,并绘制
    lines = []
    for idx, obj in enumerate(objs):
        line = []
        for frame in range(frame_rate * args.duration):
            now_ponit = obj.next_point()
            status, real_point = get_real_point(now_ponit, mesh)
            if status != PointType.ValidPoint:
                continue
            line.append(real_point)
        distance = np.linalg.norm(np.array(line[0]) - np.array(line[-1]))
        print(
            f"目标{obj.uid}在水平方向移动了{distance:.2f}米,速度为{obj.speed:.2f}m/frame,生成了{len(line)}个点数据")
        lines.append(line)
        plot_line(ax, line, color=obj_colors[idx])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    def update_view(frame):
        ax.view_init(elev=10, azim=frame)

    ani = FuncAnimation(fig, update_view, frames=360, interval=50)
    ani.save(args.vis_output, writer='ffmpeg')
