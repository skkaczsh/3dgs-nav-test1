"""阶段一：相机投影几何验证 - 配置文件
从 cam_in_ex.txt 读取标定参数，供各脚本共用
"""

import os
import numpy as np
import yaml

# ==================== 路径配置 ====================
DATA_DIR = os.environ.get("SCAN_DATA_DIR", "/root/epfs/zeende/zeende_planner/data")
IMAGE_DIR = os.environ.get("SCAN_IMAGE_DIR", os.path.join(DATA_DIR, "image"))
VIDEO_DIR = os.environ.get("SCAN_VIDEO_DIR", IMAGE_DIR)
EXTRACTED_DIR = os.environ.get("SCAN_EXTRACTED_DIR", os.path.join(DATA_DIR, "extracted"))
STAGE1_DIR = os.environ.get("SCAN_STAGE1_DIR", "/root/epfs/zeende/zeende_planner/stage1_projection")

CALIB_FILE = os.path.join(IMAGE_DIR, "cam_in_ex.txt")
IMG_POS_FILE = os.path.join(IMAGE_DIR, "img_pos.txt")

VIDEO_FILES = {
    0: os.path.join(VIDEO_DIR, "video_cam0.mkv"),
    1: os.path.join(VIDEO_DIR, "video_cam1.mkv"),
    2: os.path.join(VIDEO_DIR, "video_cam2.mkv"),
}

FRAME_OUTPUT_DIR = os.path.join(STAGE1_DIR, "frames")
FRAME_OUTPUT_DIRS = {
    0: os.path.join(FRAME_OUTPUT_DIR, "cam0"),
    1: os.path.join(FRAME_OUTPUT_DIR, "cam1"),
    2: os.path.join(FRAME_OUTPUT_DIR, "cam2"),
}
OUTPUT_DIR = os.path.join(STAGE1_DIR, "output")

# ==================== 相机内参 ====================
IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 1296

# Kannala-Brandt 鱼眼畸变系数 [k2, k3, k4, k5, k6, k7]
# 注意: OpenCV fisheye 使用 4 参数模型 (k1,k2,k3,k4)
# 本系统使用 7 参数 KB 模型，我们将其映射为 OpenCV 兼容格式
# k1≈k2, k2≈k3, k3≈k4, k4≈k5 (k6,k7 在 OpenCV 中被截断)
# 为精确起见，使用 cv2.fisheye.undistortPoints 对关键点去畸变
# 对于全图去畸变，使用 initUndistortRectifyMap 但只取前4个系数

CAMERA_PARAMS = {
    0: {
        "K": np.array([
            [731.223,  -0.057, 768.095],
            [   0.0, 731.127, 647.706],
            [   0.0,    0.0,   1.0  ],
        ], dtype=np.float64),
        # k2,k3,k4,k5 映射自 KB 模型的 k2,k3,k4,k5
        "D": np.array([-0.00320098, -0.00151995, -0.0389603, 0.0458786], dtype=np.float64),
        # OpenCV 不支持 k6,k7，但实际影响较小（< 5%畸变区域）
        "KB_k6": -0.0274188,
        "KB_k7":  0.00500572,
    },
    1: {
        "K": np.array([
            [734.325,   0.030, 785.839],
            [   0.0, 734.045, 634.320],
            [   0.0,   0.0,   1.0  ],
        ], dtype=np.float64),
        "D": np.array([-0.00118293, -0.00897628, -0.00974779, -0.000293508], dtype=np.float64),
        "KB_k6":  0.00436015,
        "KB_k7": -0.00289687,
    },
    2: {
        "K": np.array([
            [735.542,  -0.119, 769.025],
            [   0.0, 735.266, 620.786],
            [   0.0,   0.0,   1.0  ],
        ], dtype=np.float64),
        "D": np.array([-0.00540356, 0.000903556, -0.0162795, -0.00548787], dtype=np.float64),
        "KB_k6":  0.0131241,
        "KB_k7": -0.00602036,
    },
}

# ==================== 相机外参 (Lidar -> Camera, 4x4 齐次矩阵) ====================
# 格式: Tcl_k = [R|t; 0 0 0 1]
# 使用 cam_in_ex.txt 中的离线优化版本（精度更高）

Tcl = {
    0: np.array([
        [ 0.999898,  -0.0120386, -0.0076699,  0.00227464],
        [-0.00304651,  0.344962,  -0.938612, -0.0515561],
        [ 0.0139454,   0.938539,   0.34489,  -0.0380214],
        [ 0.0,         0.0,        0.0,      1.0      ],
    ], dtype=np.float64),
    1: np.array([
        [-0.00534004, -0.999972, -0.00521993,  0.000467837],
        [ 0.325495,    0.00319757,-0.945538,   -0.0542752],
        [ 0.945529,   -0.00674828, 0.325469,  -0.040306],
        [ 0.0,         0.0,        0.0,         1.0     ],
    ], dtype=np.float64),
    2: np.array([
        [-0.999992,  -0.0018473,  0.0035276,  -0.00161682],
        [-0.0027046, -0.335107,  -0.942176,  -0.0527882],
        [ 0.00292261,-0.942178,   0.335099,  -0.0409297],
        [ 0.0,        0.0,         0.0,        1.0     ],
    ], dtype=np.float64),
}

# IMU -> Lidar 外参 (Til)
Til = np.array([
    [ 0.999984, -0.00553974, 0.00106168, -0.0149559],
    [ 0.00554527, 0.999971, -0.00527896, -0.0231616],
    [-0.00103241, 0.00528476, 0.999986,   0.0451693],
    [ 0.0,        0.0,        0.0,        1.0     ],
], dtype=np.float64)

# ==================== 测试范围 (10 秒) ====================
# img_pos.txt 数据: 帧率 10Hz, 起始时间戳 ~1749910819.820857
# 10 秒 => 约 100 帧 (0 ~ 99)
TEST_START_FRAME = 0
TEST_END_FRAME = 99       # 包含，10 秒共 100 帧

# ==================== 坐标系说明 ====================
# 雷达坐标系 (Lidar/Body): X前, Y左, Z上 (右手系)
# 相机坐标系:              X右, Y下, Z前 (标准视觉惯例)
#
# 变换链: 世界 -> 机器人 -> Lidar -> Camera -> 像素
#
# P_cam    = Tcl_k @ P_lidar
# P_lidar  = Til  @ P_imu
# P_world  = T_world_robot @ P_robot
#
# 逆变换 (点云投影到图像):
# P_cam    = Tcl_k @ Til^{-1} @ T_robot2world^{-1} @ P_world
# (u, v)   = K @ P_cam / Z

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """四元数转旋转矩阵 (SciPy/Robot 惯例: wxyz)"""
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)

def build_pose_matrix(pos, quat):
    """构建 T_world_robot 齐次变换矩阵
    pos: [x, y, z] 机器人位置
    quat: [qw, qx, qy, qz] 四元数
    返回: 4x4 矩阵 [R_world_robot | t; 0 0 0 1]
    """
    R = quaternion_to_rotation_matrix(quat[0], quat[1], quat[2], quat[3])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T

def get_inv_pose(T):
    """求齐次矩阵的逆"""
    R = T[:3, :3]
    t = T[:3, 3]
    inv_T = np.eye(4, dtype=np.float64)
    inv_T[:3, :3] = R.T
    inv_T[:3, 3] = -R.T @ t
    return inv_T

def project_point_to_camera(P_world, cam_id, T_world_robot):
    """将世界点投影到指定相机图像平面
    P_world: 3D 点 (N,3) 或 (3,)
    cam_id:  相机编号 0/1/2
    T_world_robot: 4x4 世界到机器人变换
    返回: (u, v, z_cam) 或 None (不可见)
    """
    P_world = np.asarray(P_world)
    if P_world.ndim == 1:
        P_world = P_world[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    N = P_world.shape[0]

    # 步骤1: 世界 -> 机器人坐标系
    T_robot_world = get_inv_pose(T_world_robot)
    P_robot = (T_robot_world[:3, :3] @ P_world.T + T_robot_world[:3, 3:]).T

    # 步骤2: 机器人 -> Lidar 坐标系
    T_lidar_robot = get_inv_pose(Til)
    P_lidar = (T_lidar_robot[:3, :3] @ P_robot.T + T_lidar_robot[:3, 3:]).T

    # 步骤3: Lidar -> Camera 坐标系
    T_cam_lidar = Tcl[cam_id]
    P_cam = (T_cam_lidar[:3, :3] @ P_lidar.T + T_cam_lidar[:3, 3:]).T

    # 过滤: Z(相机前向) > 0
    valid = P_cam[:, 2] > 0.1
    if np.sum(valid) == 0:
        return None

    # 步骤4: 投影到像素
    K = CAMERA_PARAMS[cam_id]["K"]
    uv = (K @ P_cam[valid].T).T
    u = uv[:, 0] / uv[:, 2]
    v = uv[:, 1] / uv[:, 2]
    z = P_cam[valid, 2]

    # 过滤: 在图像范围内
    in_image = (u >= 0) & (u < IMAGE_WIDTH) & (v >= 0) & (v < IMAGE_HEIGHT)
    if np.sum(in_image) == 0:
        return None

    if squeeze and np.sum(in_image) >= 1:
        idx = np.where(in_image)[0][0]
        return float(u[idx]), float(v[idx]), float(z[idx])
    return u[in_image], v[in_image], z[in_image]

def load_img_pos(frame_start, frame_end=None):
    """加载 img_pos.txt 中的位姿数据

    实际文件格式 (16字段, 无空格分隔的相机帧对):
      0 frame_id
      1 timestamp
      2 pos_x  3 pos_y  4 pos_z
      5 qw  6 qx  7 qy  8 qz
      9 quat_norm
      10 omega_x  11 omega_y  12 omega_z
      13 cam0_frame_info (e.g. "0,1")
      14 cam1_frame_info
      15 cam2_frame_info
    """
    data = []
    with open(IMG_POS_FILE, 'rb') as f:
        for line in f:
            try:
                s = line.decode('utf-8', errors='replace').strip()
                if not s:
                    continue
                parts = s.split()
                if len(parts) < 16:
                    continue

                frame_id = int(parts[0])
                if frame_id < frame_start:
                    continue
                if frame_end is not None and frame_id > frame_end:
                    break  # 文件按 frame_id 排序，可提前终止

                timestamp = float(parts[1])
                pos = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
                quat = np.array([float(parts[5]), float(parts[6]),
                                 float(parts[7]), float(parts[8])])
                omega = np.array([float(parts[10]), float(parts[11]), float(parts[12])])

                # 解析相机帧信息 "frame_idx,frame_count"
                cam_info = {}
                for i, part in enumerate(parts[13:16]):
                    if ',' in part:
                        fid, fcnt = part.split(',')
                        cam_info[i] = (int(fid), int(fcnt))

                T_world_robot = build_pose_matrix(pos, quat)
                data.append({
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "pos": pos,
                    "quat": quat,
                    "omega": omega,
                    "cam_info": cam_info,
                    "T_world_robot": T_world_robot,
                })
            except (ValueError, IndexError):
                continue
    return data

def get_frame_timestamp(frame_id, pose_data):
    """根据帧号查找对应时间戳 (最近邻匹配)"""
    if not pose_data:
        return None
    # img_pos 每 1s 间隔约 1 行 (10Hz)，frame_id 即行索引
    for p in pose_data:
        if p["frame_id"] == frame_id:
            return p["timestamp"]
    return None

if __name__ == "__main__":
    print("=== 阶段一配置检查 ===")
    print(f"标定文件: {CALIB_FILE} (存在: {os.path.exists(CALIB_FILE)})")
    print(f"img_pos:  {IMG_POS_FILE} (存在: {os.path.exists(IMG_POS_FILE)})")
    for k, v in VIDEO_FILES.items():
        print(f"视频 {k}: {v} (存在: {os.path.exists(v)})")

    print(f"\n测试范围: 帧 {TEST_START_FRAME} ~ {TEST_END_FRAME} (10秒)")
    print(f"图像尺寸: {IMAGE_WIDTH} x {IMAGE_HEIGHT}")
    print(f"\n相机内参示例 (cam_0):")
    print(f"  K = \n{CAMERA_PARAMS[0]['K']}")
    print(f"  D = {CAMERA_PARAMS[0]['D']}")
    print(f"\n相机外参 Tcl_0:")
    print(f"  t = {Tcl[0][:3, 3]}")
    print(f"\nIMU->Lidar Til:")
    print(f"  t = {Til[:3, 3]}")

    print(f"\n测试加载位姿数据 (帧 0~10)...")
    poses = load_img_pos(0, 10)
    print(f"  加载了 {len(poses)} 帧")
    if poses:
        p = poses[0]
        print(f"  帧0: timestamp={p['timestamp']:.6f}, pos={p['pos']}, quat={p['quat']}")
        print(f"  T_world_robot:\n{p['T_world_robot']}")
