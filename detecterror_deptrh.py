import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# =========================================================
# ✅ 开关：选择用 midpoints 还是 rightpoint 来评估距离误差
#   True  -> 用 midpoints（双目最小距离平面求中点）
#   False -> 用 rightpoint（右相机射线与 y 平面交点）
# =========================================================
USE_MIDPOINTS = True

# 固定与 y=常数平面求交时使用的 y（与你原来一致）
PLANE_Y =65.5

# JSON读取条数
NUM_ENTRIES = 30

# =========================================================
# 初始化所有存储列表
# =========================================================
Frame_list = []
Timestamp_list = []
ObjectPosition_list = []
ObjectRotation_list = []
LeftCameraPosition_list = []
LeftCameraRotation_list = []
LeftProjectionMatrix_list = []
LeftWorldToCameraMatrix_list = []
LeftObjectCameraPosition_list = []
LeftObjectScreenPosition_list = []
LeftDistance_list = []
RightCameraPosition_list = []
RightCameraRotation_list = []
RightProjectionMatrix_list = []
RightWorldToCameraMatrix_list = []
RightObjectCameraPosition_list = []
RightObjectScreenPosition_list = []
RightDistance_list = []

LeftWorldRays_list = []   # (p_world_near, p_world_far)
RightWorldRays_list = []  # (p_world_near, p_world_far)

# =========================================================
# 从JSON文件读取数据
# =========================================================
with open('map.json', 'r') as f:
    data = json.load(f)

entries = data[:NUM_ENTRIES]

def dict_to_matrix(matrix_dict):
    """将字典形式的矩阵数据转换为4x4 numpy数组（行主序）"""
    keys = ['e00', 'e01', 'e02', 'e03',
            'e10', 'e11', 'e12', 'e13',
            'e20', 'e21', 'e22', 'e23',
            'e30', 'e31', 'e32', 'e33']
    matrix_values = [matrix_dict[key] for key in keys]
    return np.array(matrix_values, dtype=np.float32).reshape(4, 4)

for entry in entries:
    Frame_list.append(entry["Frame"])
    Timestamp_list.append(entry["Timestamp"])

    ObjectPosition_list.append(entry["ObjectPosition"])
    ObjectRotation_list.append(entry["ObjectRotation"])

    LeftCameraPosition_list.append(entry["LeftCameraPosition"])
    LeftCameraRotation_list.append(entry["LeftCameraRotation"])
    LeftProjectionMatrix_list.append(dict_to_matrix(entry["LeftProjectionMatrix"]))
    LeftWorldToCameraMatrix_list.append(dict_to_matrix(entry["LeftWorldToCameraMatrix"]))
    LeftObjectCameraPosition_list.append(entry["LeftObjectCameraPosition"])
    LeftObjectScreenPosition_list.append(entry["LeftObjectScreenPosition"])
    LeftDistance_list.append(entry["LeftDistance"])

    RightCameraPosition_list.append(entry["RightCameraPosition"])
    RightCameraRotation_list.append(entry["RightCameraRotation"])
    RightProjectionMatrix_list.append(dict_to_matrix(entry["RightProjectionMatrix"]))
    RightWorldToCameraMatrix_list.append(dict_to_matrix(entry["RightWorldToCameraMatrix"]))
    RightObjectCameraPosition_list.append(entry["RightObjectCameraPosition"])
    RightObjectScreenPosition_list.append(entry["RightObjectScreenPosition"])
    RightDistance_list.append(entry["RightDistance"])

# =========================================================
# 射线计算
# =========================================================
width, height = 1280, 960

def compute_world_ray(ProjectionMatrix, WorldToCameraMatrix, screen_point, width, height):
    """计算从屏幕点到世界空间的射线点对"""
    screen_x, screen_y = screen_point["x"], screen_point["y"]
    # 保留你原本的像素噪声
    screen_x += np.random.uniform(-1, 1)
    screen_y += np.random.uniform(-1, 1)

    ndc_x = (screen_x / width) * 2.0 - 1.0
    ndc_y = (screen_y / height) * 2.0 - 1.0

    p_ndc_near = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)
    p_ndc_far  = np.array([ndc_x, ndc_y,  1.0, 1.0], dtype=np.float32)

    inv_proj = np.linalg.inv(ProjectionMatrix)
    p_cam_near = inv_proj @ p_ndc_near
    p_cam_far  = inv_proj @ p_ndc_far

    p_cam_near /= p_cam_near[3]
    p_cam_far  /= p_cam_far[3]

    inv_view = np.linalg.inv(WorldToCameraMatrix)
    p_world_near = inv_view @ p_cam_near
    p_world_far  = inv_view @ p_cam_far

    p_world_near = p_world_near[:3] / p_world_near[3]
    p_world_far  = p_world_far[:3] / p_world_far[3]
    return p_world_near, p_world_far

for i in range(len(entries)):
    left_near, left_far = compute_world_ray(
        LeftProjectionMatrix_list[i],
        LeftWorldToCameraMatrix_list[i],
        LeftObjectScreenPosition_list[i],
        width, height
    )
    LeftWorldRays_list.append((left_near, left_far))

    right_near, right_far = compute_world_ray(
        RightProjectionMatrix_list[i],
        RightWorldToCameraMatrix_list[i],
        RightObjectScreenPosition_list[i],
        width, height
    )
    RightWorldRays_list.append((right_near, right_far))

# =========================================================
# y=常数平面求交点
# =========================================================
def ray_plane(worldRaylist, known_world_y):
    crosspointlist = []
    for i in range(len(worldRaylist)):
        p_world_near, p_world_far = worldRaylist[i][0], worldRaylist[i][1]
        t = (known_world_y - p_world_near[1]) / (p_world_far[1] - p_world_near[1])
        worldpos = p_world_near + t * (p_world_far - p_world_near)
        crosspointlist.append(worldpos)
    return crosspointlist

rightpoint = ray_plane(RightWorldRays_list, PLANE_Y)
leftpoint  = ray_plane(LeftWorldRays_list,  PLANE_Y)

# =========================================================
# midpoints: 选最优 y=c* 使左右交点距离最小，并取中点
# =========================================================
def _ray_affine_y(ray, eps=1e-12):
    """
    将射线与 y=c 平面的交点表示为 p(c) = A + B * c
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
    """直接求与 y=c 的交点；若平行则返回 None"""
    p0, p1 = ray
    d = p1 - p0
    dy = d[1]
    if abs(dy) < eps:
        return None
    t = (c - p0[1]) / dy
    return p0 + t * d

def midpoints_min_distance(RightWorldRays_list, LeftWorldRays_list, y_fallback=65.5, eps=1e-12):
    n = min(len(RightWorldRays_list), len(LeftWorldRays_list))
    midpoints, c_opt_list, pairs_points = [], [], []

    for i in range(n):
        Rray = RightWorldRays_list[i]
        Lray = LeftWorldRays_list[i]

        Ar, Br = _ray_affine_y(Rray, eps=eps)
        Al, Bl = _ray_affine_y(Lray, eps=eps)

        if Ar is None or Al is None:
            rp = _intersect_y(Rray, y_fallback, eps=eps)
            lp = _intersect_y(Lray, y_fallback, eps=eps)
            if rp is None or lp is None:
                midpoints.append(None)
                c_opt_list.append(None)
                pairs_points.append((None, None))
                continue
            midpoints.append(0.5 * (rp + lp))
            c_opt_list.append(y_fallback)
            pairs_points.append((rp, lp))
            continue

        u = Ar - Al
        v = Br - Bl
        v2 = float(np.dot(v, v))
        if v2 < eps:
            c_opt = y_fallback
        else:
            c_opt = - float(np.dot(v, u)) / v2

        rp = Ar + Br * c_opt
        lp = Al + Bl * c_opt
        midpoints.append(0.5 * (rp + lp))
        c_opt_list.append(c_opt)
        pairs_points.append((rp, lp))

    return midpoints, c_opt_list, pairs_points

midpoints, y_opt_list, pairs = midpoints_min_distance(
    RightWorldRays_list, LeftWorldRays_list, y_fallback=PLANE_Y
)

# =========================================================
# ✅ 距离评估：RMSE / MAE / MaxAbsError
#   - 可用开关选择 pred 点：midpoints 或 rightpoint
#   - 距离定义：相机位置与目标位置的三维欧氏距离
# =========================================================
def calculate_distance_metrics(pred_points,
                               camera_pos_list,
                               object_pos_list,
                               save_path="distance_result",
                               plot=True):
    """
    pred_points: list[np.array(3,)] or list[None]，预测目标点（midpoints 或 rightpoint）
    camera_pos_list: 相机位置列表（LeftCameraPosition_list 或 RightCameraPosition_list）
    object_pos_list: 目标真值位置列表（ObjectPosition_list）
    """

    cam = np.array([[p["x"], p["y"], p["z"]] for p in camera_pos_list], dtype=np.float32)
    gt_obj = np.array([[p["x"], p["y"], p["z"]] for p in object_pos_list], dtype=np.float32)

    n = min(len(cam), len(gt_obj), len(pred_points))
    cam = cam[:n]
    gt_obj = gt_obj[:n]
    pred_points = pred_points[:n]

    valid_mask = np.array([p is not None for p in pred_points], dtype=bool)
    pred_obj = np.zeros_like(gt_obj, dtype=np.float32)

    for i, p in enumerate(pred_points):
        if p is None:
            valid_mask[i] = False
            continue
        p = np.asarray(p, dtype=np.float32).reshape(3,)
        if np.any(~np.isfinite(p)):
            valid_mask[i] = False
            continue
        pred_obj[i] = p

    if valid_mask.sum() == 0:
        raise RuntimeError("没有有效预测点可用于距离评估（全部为None或NaN）。")

    pred_dist_all = np.linalg.norm(cam - pred_obj, axis=1)
    gt_dist_all   = np.linalg.norm(cam - gt_obj, axis=1)

    err = (pred_dist_all - gt_dist_all)[valid_mask]

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae  = float(np.mean(np.abs(err)))
    max_abs_err = float(np.max(np.abs(err)))

    with open(f"{save_path}.txt", "w") as f:
        f.write(f"Distance RMSE: {rmse:.6f}\n")
        f.write(f"Distance MAE: {mae:.6f}\n")
        f.write(f"Distance MaxAbsError: {max_abs_err:.6f}\n")
        f.write(f"Total frames: {n}\n")
        f.write(f"Valid frames: {int(valid_mask.sum())}\n")

    if plot:
        x = np.arange(n)
        plt.figure(figsize=(10, 4))
        plt.plot(x, gt_dist_all, label="GT Distance (Cam → Object)")
        plt.plot(x, pred_dist_all, label="Pred Distance (Cam → PredPoint)")
        if np.any(~valid_mask):
            bad = np.where(~valid_mask)[0]
            plt.scatter(bad, gt_dist_all[bad], marker="x", label="Invalid pred point (GT shown)")
        plt.xlabel("Frame Index")
        plt.ylabel("Distance")
        plt.title(f"Distance Comparison\nRMSE={rmse:.4f}, MAE={mae:.4f}, MaxAbsErr={max_abs_err:.4f}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[Distance Metrics] RMSE={rmse:.4f}, MAE={mae:.4f}, MaxAbsErr={max_abs_err:.4f}")
    print(f"Saved: {save_path}.txt / {save_path}.png")
    return rmse, mae, max_abs_err

# =========================================================
# ✅ 根据开关选择评估对象：
#   - midpoints -> 使用 RightCameraPosition (更合理：midpoints来自左右相机几何)
#   - rightpoint -> 使用 RightCameraPosition
# 真值使用 ObjectPosition
# =========================================================
if USE_MIDPOINTS:
    pred_points = midpoints
    save_prefix = "distance_using_midpoints"
    print(">>> Using MIDPOINTS for distance evaluation")
else:
    pred_points = rightpoint
    save_prefix = "distance_using_rightpoint"
    print(">>> Using RIGHTPOINT for distance evaluation")

calculate_distance_metrics(
    pred_points=pred_points,
    camera_pos_list=RightCameraPosition_list,
    object_pos_list=ObjectPosition_list,
    save_path=save_prefix,
    plot=True
)
