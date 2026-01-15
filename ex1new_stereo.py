import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 初始化所有存储列表
Frame_list = []
Timestamp_list = []
ObjectPosition_list = []
ObjectRotation_list = []
LeftCameraPosition_list = []
LeftCameraRotation_list = []
LeftProjectionMatrix_list = []       # 存储4x4矩阵
LeftWorldToCameraMatrix_list = []    # 存储4x4矩阵
LeftObjectCameraPosition_list = []
LeftObjectScreenPosition_list = []
LeftDistance_list = []
RightCameraPosition_list = []
RightCameraRotation_list = []
RightProjectionMatrix_list = []      # 存储4x4矩阵
RightWorldToCameraMatrix_list = []   # 存储4x4矩阵
RightObjectCameraPosition_list = []
RightObjectScreenPosition_list = []
RightDistance_list = []

# 新增的射线点对存储列表
LeftWorldRays_list = []  # 存储左相机的射线点对 (p_world_near, p_world_far)
RightWorldRays_list = []  # 存储右相机的射线点对

# 从JSON文件读取数据
with open('map.json', 'r') as f:
    data = json.load(f)  # 加载整个JSON数据

# 提取前30个条目
entries = data[:30]

# 辅助函数：将矩阵字典转换为4x4 numpy数组
def dict_to_matrix(matrix_dict):
    """将字典形式的矩阵数据转换为4x4 numpy数组"""
    # 确定矩阵元素顺序（行主序）
    keys = ['e00', 'e01', 'e02', 'e03',
            'e10', 'e11', 'e12', 'e13',
            'e20', 'e21', 'e22', 'e23',
            'e30', 'e31', 'e32', 'e33']
    
    # 提取值并转换为列表
    matrix_values = [matrix_dict[key] for key in keys]
    
    # 转换为4x4 numpy数组
    return np.array(matrix_values, dtype=np.float32).reshape(4, 4)

# 遍历每个条目并存储到对应列表
for entry in entries:
    Frame_list.append(entry["Frame"])
    Timestamp_list.append(entry["Timestamp"])
    
    # 对象位置和旋转
    ObjectPosition_list.append(entry["ObjectPosition"])
    ObjectRotation_list.append(entry["ObjectRotation"])
    
    # 左相机数据
    LeftCameraPosition_list.append(entry["LeftCameraPosition"])
    LeftCameraRotation_list.append(entry["LeftCameraRotation"])
    
    # 转换矩阵数据
    LeftProjectionMatrix_list.append(dict_to_matrix(entry["LeftProjectionMatrix"]))
    LeftWorldToCameraMatrix_list.append(dict_to_matrix(entry["LeftWorldToCameraMatrix"]))
    
    LeftObjectCameraPosition_list.append(entry["LeftObjectCameraPosition"])
    LeftObjectScreenPosition_list.append(entry["LeftObjectScreenPosition"])
    LeftDistance_list.append(entry["LeftDistance"])
    
    # 右相机数据
    RightCameraPosition_list.append(entry["RightCameraPosition"])
    RightCameraRotation_list.append(entry["RightCameraRotation"])
    
    # 转换矩阵数据
    RightProjectionMatrix_list.append(dict_to_matrix(entry["RightProjectionMatrix"]))
    RightWorldToCameraMatrix_list.append(dict_to_matrix(entry["RightWorldToCameraMatrix"]))
    
    RightObjectCameraPosition_list.append(entry["RightObjectCameraPosition"])
    RightObjectScreenPosition_list.append(entry["RightObjectScreenPosition"])
    RightDistance_list.append(entry["RightDistance"])

# 屏幕分辨率
width, height = 1280, 960

# 定义射线计算函数
def compute_world_ray(ProjectionMatrix, WorldToCameraMatrix, screen_point, width, height):
    """计算从屏幕点到世界空间的射线点对"""
    # 1️⃣ NDC 坐标 (范围 [-1,1])
    
    screen_x, screen_y = screen_point["x"], screen_point["y"]
    screen_x += np.random.uniform(-1, 1)
    screen_y += np.random.uniform(-1, 1)
    # screen_y=height-screen_y
    ndc_x = (screen_x / width) * 2.0 - 1.0
    # ndc_y = 1.0 - (screen_y / height) * 2.0  # Y 轴方向翻转
    ndc_y = (screen_y / height) * 2.0 -1
    
    # 构造两个点：zNear (0)，zFar (1)
    p_ndc_near = np.array([ndc_x, ndc_y, -1.0, 1.0])
    p_ndc_far  = np.array([ndc_x, ndc_y,  1.0, 1.0])
    
    # 2️⃣ NDC -> 相机坐标
    inv_proj = np.linalg.inv(ProjectionMatrix)
    p_cam_near = inv_proj @ p_ndc_near
    p_cam_far  = inv_proj @ p_ndc_far
    
    # 齐次除法
    p_cam_near /= p_cam_near[3]
    p_cam_far  /= p_cam_far[3]
    
    # 3️⃣ 相机坐标 -> 世界坐标
    inv_view = np.linalg.inv(WorldToCameraMatrix)
    p_world_near = inv_view @ p_cam_near
    p_world_far  = inv_view @ p_cam_far
    
    # 齐次除法并返回三维坐标
    p_world_near = p_world_near[:3] / p_world_near[3]
    p_world_far = p_world_far[:3] / p_world_far[3]
    
    return p_world_near, p_world_far

# 为每个帧计算左右相机的世界空间射线点对
for i in range(len(entries)):
    # 左相机计算
    left_near, left_far = compute_world_ray(
        LeftProjectionMatrix_list[i],
        LeftWorldToCameraMatrix_list[i],
        LeftObjectScreenPosition_list[i],
        width, height
    )
    LeftWorldRays_list.append((left_near, left_far))
    
    # 右相机计算
    right_near, right_far = compute_world_ray(
        RightProjectionMatrix_list[i],
        RightWorldToCameraMatrix_list[i],
        RightObjectScreenPosition_list[i],
        width, height
    )
    RightWorldRays_list.append((right_near, right_far))
##########################################################
def ray_plane(worldRaylist,known_world_y):
    crosspointlist= []
    for i in range(30):
        p_world_near,p_world_far=worldRaylist[i][0] ,worldRaylist[i][1] 
    
        t = (known_world_y - p_world_near[1]) / (p_world_far[1] - p_world_near[1])
        worldpos = p_world_near + t * (p_world_far - p_world_near)
    
        crosspointlist.append(worldpos)
    return crosspointlist

y = 65.5
rightpoint=ray_plane(RightWorldRays_list,y)
leftpoint =ray_plane(LeftWorldRays_list,y)
##########################################################
def _ray_affine_y(ray, eps=1e-12):
    """
    将射线与 y=c 平面的交点表示为 p(c) = A + B * c
    ray: (p_near, p_far)，每个为形如 (3,) 的 np.array
    返回 (A, B)；若与平面平行(即 dy≈0)则返回 (None, None)
    """
    p0, p1 = ray
    d = p1 - p0
    dy = d[1]
    if abs(dy) < eps:
        return None, None
    B = d / dy
    A = p0 - (p0[1] / dy) * d
    return A, B

def _intersect_y(ray, c, eps=1e-12):
    """备选：直接用参数 t 求与 y=c 的交点；若平行则返回 None"""
    p0, p1 = ray
    d = p1 - p0
    dy = d[1]
    if abs(dy) < eps:
        return None
    t = (c - p0[1]) / dy
    return p0 + t * d

def midpoints_min_distance(RightWorldRays_list, LeftWorldRays_list, y_fallback=65.6, eps=1e-12):
    """
    对每个 i，寻找最优平面 y=c* 使得 rightpoint_i(c) 与 leftpoint_i(c) 的距离最小；
    然后取两点中点作为 midpoint[i]。
    返回：
      midpoints:    [ (3,) np.array or None ]
      c_opt_list:   [ float or None ]  对应使用的最优 y
      pairs_points: [ (right_point, left_point) or (None, None) ] 方便调试
    """
    n = min(len(RightWorldRays_list), len(LeftWorldRays_list))
    midpoints, c_opt_list, pairs_points = [], [], []

    for i in range(n):
        Rray = RightWorldRays_list[i]
        Lray = LeftWorldRays_list[i]

        Ar, Br = _ray_affine_y(Rray, eps=eps)
        Al, Bl = _ray_affine_y(Lray, eps=eps)

        # 若有任一射线与 y 平面平行，退化到固定 y_fallback
        if Ar is None or Al is None:
            rp = _intersect_y(Rray, y_fallback, eps=eps)
            lp = _intersect_y(Lray, y_fallback, eps=eps)
            if rp is None or lp is None:
                midpoints.append(None)
                c_opt_list.append(None)
                pairs_points.append((None, None))
                continue
            mid = 0.5 * (rp + lp)
            midpoints.append(mid)
            c_opt_list.append(y_fallback)
            pairs_points.append((rp, lp))
            continue

        # 解析最优 c*
        u = Ar - Al
        v = Br - Bl
        v2 = float(np.dot(v, v))
        if v2 < eps:
            # 两交点对 y 的变化几乎一致，距离对 y 不敏感；使用回退 y
            c_opt = y_fallback
        else:
            c_opt = - float(np.dot(v, u)) / v2

        rp = Ar + Br * c_opt
        lp = Al + Bl * c_opt
        mid = 0.5 * (rp + lp)

        midpoints.append(mid)
        c_opt_list.append(c_opt)
        pairs_points.append((rp, lp))

    return midpoints, c_opt_list, pairs_points

midpoints, y_opt_list, pairs = midpoints_min_distance(RightWorldRays_list, LeftWorldRays_list, y_fallback=65.5)
# print(midpoints)
midpoints = np.array(midpoints)

gt =[]
for i in range(len(ObjectPosition_list)):
    gt.append([ObjectPosition_list[i]['x'],ObjectPosition_list[i]['y'],ObjectPosition_list[i]['z']])
gt = np.array(gt)




#######画图##############
def calculate_and_plot_3d_rmse(midpoints, gt_points, save_path="ex3result"):
    """
    计算3D RMSE并在三维空间中绘制路径（支持三轴等比例显示）
    
    参数:
    midpoints: numpy数组 (n×3), 预测点
    gt_points: numpy数组 (n×3), 真实点
    save_path: 结果保存路径前缀
    """
    # 计算3D RMSE
    distances = np.linalg.norm(midpoints - gt_points, axis=1)
    rmse = np.sqrt(np.mean(distances**2))
    
    # 保存RMSE结果
    with open(f"{save_path}.txt", 'w') as f:
        f.write(f"3D RMSE: {rmse:.6f}\n")
        f.write(f"Number of points: {len(distances)}\n")
    
    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取坐标
    mx, my, mz = midpoints[:, 0], midpoints[:, 1], midpoints[:, 2]
    gx, gy, gz = gt_points[:, 0], gt_points[:, 1], gt_points[:, 2]
    
    # 绘制三维路径
    ax.plot(mx, my, mz, 'b-', label='Predicted Path', linewidth=1.5)
    ax.plot(gx, gy, gz, 'r-', label='Ground Truth', linewidth=1.5)
    
    # 标记起点和终点
    ax.scatter(mx[0], my[0], mz[0], c='g', s=50, marker='o', label='Predicted Start')
    ax.scatter(gx[0], gy[0], gz[0], c='y', s=50, marker='o', label='GT Start')
    ax.scatter(mx[-1], my[-1], mz[-1], c='b', s=50, marker='s', label='Predicted End')
    ax.scatter(gx[-1], gy[-1], gz[-1], c='r', s=50, marker='s', label='GT End')
    
    # 添加标签和标题
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title(f'3D Path Comparison\n3D RMSE = {rmse:.4f}')
    ax.legend()
    
    # === 设置三轴等比例显示 ===
    # 获取所有坐标点
    all_points = np.vstack((midpoints, gt_points))
    x_limits = [np.min(all_points[:, 0]), np.max(all_points[:, 0])]
    y_limits = [np.min(all_points[:, 1]), np.max(all_points[:, 1])]
    z_limits = [np.min(all_points[:, 2]), np.max(all_points[:, 2])]
    
    # 计算中心和范围
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range) / 2.0
    
    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)
    
    # 设置等比例显示
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 保存图像
    plt.savefig(f"{save_path}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"3D RMSE: {rmse:.4f} units")
    print(f"Results saved to {save_path}.[txt/png]")
    return rmse
print(midpoints)
# 使用示例:
# calculate_and_plot_3d_rmse(midpoint_array, gt_array, "3d_path_comparison")
calculate_and_plot_3d_rmse(midpoints,gt)