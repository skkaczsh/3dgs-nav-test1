#!/usr/bin/env python3
"""单帧点云着色：将 LiDAR 点云投影到相机图像采样 RGB 颜色。

关键理解：
  - PLY section 中的点云是世界坐标系（非本地 LiDAR 坐标系）
  - 需要通过 img_pos.txt 的位姿（位置+四元数）构建世界→相机变换
  - 变换链：P_world → P_body(=P_local) → P_lidar → P_cam → 像素

用法：
    python colorize_single_frame.py --frame 0
    python colorize_single_frame.py --frame 100 --output colorized.ply
    python colorize_single_frame.py --frame 500 --camera 1
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# ============================================================
# 数据解析
# ============================================================

def parse_cam_in_ex(path):
    """解析 cam_in_ex.txt：设备信息和 3 相机内外参。"""
    import re
    with open(path) as f:
        text = f.read()

    result = {'device': '', 'cameras': {}, 'Tcl': {}, 'Til': None}

    m = re.search(r'device:\s*(.+)', text)
    if m:
        result['device'] = m.group(1).strip()

    for i in range(3):
        pattern = rf'Tcl_{i}:\s*\[([^\]]+)\]'
        m = re.search(pattern, text)
        if m:
            vals = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', m.group(1))]
            result['Tcl'][i] = np.array(vals).reshape(4, 4)

    m = re.search(r'Til:\s*\[([^\]]+)\]', text)
    if m:
        vals = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', m.group(1))]
        result['Til'] = np.array(vals).reshape(4, 4)

    for i in range(3):
        cam = {}
        section = re.search(
            rf'cam_{i}:\s*(.*?)(?=cam_\d:|Til:|\Z)', text, re.DOTALL
        )
        if not section:
            continue
        block = section.group(1)

        for key in ['image_width', 'image_height']:
            m = re.search(rf'{key}:\s*(\d+)', block)
            if m:
                cam[key] = int(m.group(1))

        for key in ['k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'p1', 'p2',
                     'A11', 'A12', 'A22', 'u0', 'v0']:
            m = re.search(rf'{key}:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', block)
            if m:
                cam[key] = float(m.group(1))

        result['cameras'][i] = cam

    return result


def parse_img_pos(path):
    """解析 img_pos.txt，返回所有帧的位姿信息。

    返回: dict[idx] = {tx, ty, tz, qw, qx, qy, qz, timestamp}
    """
    entries = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            idx = int(parts[0])
            entries[idx] = {
                'timestamp': float(parts[1].replace(',', '')),
                'tx': float(parts[2].replace(',', '')),
                'ty': float(parts[3].replace(',', '')),
                'tz': float(parts[4].replace(',', '')),
                'qw': float(parts[5].replace(',', '')),
                'qx': float(parts[6].replace(',', '')),
                'qy': float(parts[7].replace(',', '')),
                'qz': float(parts[8].replace(',', '')),
            }
    return entries


def read_ply_section(path):
    """读取 ASCII PLY 文件，返回 (N, 3) 的 xyz 点云。"""
    points = []
    with open(path) as f:
        header_done = False
        for line in f:
            if not header_done:
                if line.strip() == 'end_header':
                    header_done = True
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(points, dtype=np.float64)


def read_video_frame(video_path, frame_index):
    """从视频文件读取指定帧，返回 RGB 图像。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: 无法打开视频 {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"  WARNING: 无法读取帧 {frame_index}")
        return None

    return frame[:, :, ::-1]  # BGR -> RGB


# ============================================================
# 变换构建
# ============================================================

def build_world_pose(entry):
    """从 img_pos 条目构建 T_world_body（4×4 刚体变换矩阵）。

    img_pos.txt 的列3-5 是 IMU 位置，列6-9 是四元数。
    组合构成 world→body 变换（body = IMU 坐标系）。
    """
    R = Rotation.from_quat([entry['qx'], entry['qy'], entry['qz'], entry['qw']]).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [entry['tx'], entry['ty'], entry['tz']]
    return T


# ============================================================
# Kannala 鱼眼投影
# ============================================================

def kannala_project(points_cam, cam_params, w, h):
    """Kannala 鱼眼模型投影：3D 点（相机坐标系）→ 2D 像素坐标。

    返回:
        pixels: (N, 2) 像素坐标 (u, v)
        valid: (N,) bool，有效投影标记
    """
    X = points_cam[:, 0]
    Y = points_cam[:, 1]
    Z = points_cam[:, 2]

    # 在相机前方的点才有效
    valid = Z > 0.1
    x = np.where(valid, X / Z, 0.0)
    y = np.where(valid, Y / Z, 0.0)

    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan(r)

    # Kannala 畸变: ρ = θ + k₂θ³ + k₃θ⁵ + k₄θ⁷ + k₅θ⁹ + k₆θ¹¹ + k₇θ¹³
    k2 = cam_params.get('k2', 0)
    k3 = cam_params.get('k3', 0)
    k4 = cam_params.get('k4', 0)
    k5 = cam_params.get('k5', 0)
    k6 = cam_params.get('k6', 0)
    k7 = cam_params.get('k7', 0)

    t2 = theta * theta
    theta_p = theta
    theta_p += k2 * t2 * theta       # k2 * theta^3
    theta_p += k3 * t2 * t2 * theta   # k3 * theta^5
    theta_p += k4 * t2**3 * theta      # k4 * theta^7
    theta_p += k5 * t2**4 * theta      # k5 * theta^9
    theta_p += k6 * t2**5 * theta      # k6 * theta^11
    theta_p += k7 * t2**6 * theta      # k7 * theta^13
    rho = theta_p

    # 避免除零
    r_safe = np.where(r > 1e-8, r, 1.0)
    scale = np.where(r > 1e-8, rho / r_safe, 1.0)

    fx = cam_params['A11']
    fy = cam_params['A22']
    skew = cam_params.get('A12', 0)
    cx = cam_params['u0']
    cy = cam_params['v0']

    u = fx * scale * x + skew * scale * y + cx
    v = fy * scale * y + cy

    pixels = np.stack([u, v], axis=-1)
    in_image = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    valid = valid & in_image

    return pixels, valid


# ============================================================
# 着色
# ============================================================

def colorize_frame(ply_path, video_paths, cam_data, img_pos_entries,
                   frame_index, output_path, specific_camera=None):
    """对单帧 LiDAR 点云着色。

    变换链：
    P_world → T_world_lidar_inv → P_local_lidar → T_lidar_cam(inv(Icl)) → P_cam → 像素

    其中:
    - T_world_lidar = T_world_body @ Til  (img_pos 位姿 × IMU-LiDAR 外参)
    - T_world_body = R(q) | t  (img_pos 四元数 + 位置)
    - T_lidar_cam = inv(Tcl)  (相机→LiDAR 外参的逆)
    """
    print(f"{'='*60}")
    print(f"着色帧 {frame_index}")
    print(f"{'='*60}")

    # 1. 读取 PLY
    print(f"\n读取点云: {ply_path}")
    pts_world = read_ply_section(ply_path)
    N = len(pts_world)
    print(f"  点数: {N:,}")
    print(f"  世界坐标范围: X[{pts_world[:,0].min():.1f}, {pts_world[:,0].max():.1f}] "
          f"Y[{pts_world[:,1].min():.1f}, {pts_world[:,1].max():.1f}] "
          f"Z[{pts_world[:,2].min():.1f}, {pts_world[:,2].max():.1f}]")

    # 2. 读取位姿
    if frame_index not in img_pos_entries:
        print(f"ERROR: 帧 {frame_index} 不在 img_pos.txt 中")
        return None, None, None

    entry = img_pos_entries[frame_index]
    T_world_body = build_world_pose(entry)
    Til = cam_data['Til']
    T_world_lidar = T_world_body @ Til

    print(f"\n帧 {frame_index} 位姿:")
    print(f"  IMU 位置: ({entry['tx']:.4f}, {entry['ty']:.4f}, {entry['tz']:.4f})")
    print(f"  四元数: qw={entry['qw']:.6f}, qx={entry['qx']:.6f}, "
          f"qy={entry['qy']:.6f}, qz={entry['qz']:.6f}")

    # 3. 齐次坐标
    pts_h = np.hstack([pts_world, np.ones((N, 1))])  # (N, 4)

    # 4. 确定使用哪些相机
    if specific_camera is not None:
        cameras_to_use = [specific_camera]
    else:
        cameras_to_use = [0, 1, 2]

    # 5. 初始化输出
    colors = np.zeros((N, 3), dtype=np.uint8)
    best_depth = np.full(N, np.inf)
    best_count = np.zeros(N, dtype=np.int32)

    # 6. 局部坐标（用于调试）
    pts_local = (np.linalg.inv(T_world_lidar) @ pts_h.T).T[:, :3]
    print(f"  本地坐标范围: X[{pts_local[:,0].min():.2f}, {pts_local[:,0].max():.2f}] "
          f"Y[{pts_local[:,1].min():.2f}, {pts_local[:,1].max():.2f}] "
          f"Z[{pts_local[:,2].min():.2f}, {pts_local[:,2].max():.2f}]")

    for cam_id in cameras_to_use:
        print(f"\n--- 相机 cam{cam_id} ---")

        if cam_id not in video_paths or not os.path.exists(video_paths[cam_id]):
            print(f"  跳过：视频文件不存在")
            continue

        if cam_id not in cam_data['Tcl']:
            print(f"  跳过：无 Tcl 外参")
            continue

        # 读取视频帧
        image = read_video_frame(video_paths[cam_id], frame_index)
        if image is None:
            continue
        h_img, w_img = image.shape[:2]
        print(f"  图像: {w_img}×{h_img}")

        # 变换: P_world → P_cam
        Tcl = cam_data['Tcl'][cam_id]
        T_lidar_cam = np.linalg.inv(Tcl)
        # P_cam = T_lidar_cam @ inv(T_world_lidar) @ P_world
        # 即: P_cam = T_lidar_cam @ T_lidar_world @ P_world
        T_lidar_world = np.linalg.inv(T_world_lidar)
        T_cam_world = T_lidar_cam @ T_lidar_world  # (4, 4)

        pts_cam = (T_cam_world @ pts_h.T).T[:, :3]  # (N, 3)

        # 前方点的比例
        in_front = pts_cam[:, 2] > 0
        print(f"  相机前方点数: {in_front.sum():,} / {N:,} ({in_front.sum()/N*100:.1f}%)")

        # Kannala 投影
        cam_params = cam_data['cameras'][cam_id]
        pixels, valid = kannala_project(pts_cam, cam_params, w_img, h_img)

        n_valid = valid.sum()
        print(f"  有效投影点数: {n_valid:,} / {N:,} ({n_valid/N*100:.1f}%)")

        if n_valid == 0:
            continue

        # 采样颜色
        u_coords = np.clip(np.round(pixels[valid, 0]).astype(np.int32), 0, w_img - 1)
        v_coords = np.clip(np.round(pixels[valid, 1]).astype(np.int32), 0, h_img - 1)
        sampled_colors = image[v_coords, u_coords]  # (n_valid, 3)

        # 深度融合：保留最近的相机颜色
        depths = pts_cam[valid, 2]
        valid_indices = np.where(valid)[0]
        for i in range(len(valid_indices)):
            idx = valid_indices[i]
            if depths[i] < best_depth[idx]:
                best_depth[idx] = depths[i]
                colors[idx] = sampled_colors[i]
            best_count[idx] += 1

    # 7. 结果统计
    colored = best_count > 0
    n_colored = colored.sum()
    print(f"\n{'='*60}")
    print(f"着色结果:")
    print(f"  总点数: {N:,}")
    print(f"  着色点数: {n_colored:,} ({n_colored/N*100:.1f}%)")
    print(f"  未着色点数: {N - n_colored:,}")
    print(f"{'='*60}")

    # 8. 写入 PLY
    write_colored_ply(output_path, pts_world, colors, colored)
    print(f"已保存: {output_path}")

    return pts_world, colors, colored


def write_colored_ply(path, points, colors, valid_mask,
                      uncolored_color=(128, 128, 128)):
    """写入带颜色的 PLY 文件。"""
    N = len(points)
    default_color = np.array(uncolored_color, dtype=np.uint8)

    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(N):
            x, y, z = points[i]
            if valid_mask[i]:
                r, g, b = int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])
            else:
                r, g, b = int(default_color[0]), int(default_color[1]), int(default_color[2])
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='单帧点云着色')
    parser.add_argument('--frame', type=int, default=0,
                        help='帧号 (0~7630)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 PLY 路径')
    parser.add_argument('--camera', type=int, default=None,
                        help='仅使用指定相机 (0/1/2)')
    parser.add_argument('--data-dir', type=str,
                        default='/Users/skkac/Work/SCAN/new_route',
                        help='数据目录')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    frame = args.frame

    # 路径
    ply_path = data_dir / 'extracted' / f'section_{frame:04d}.ply'
    video_paths = {
        0: str(data_dir / 'image' / 'video_cam0.mkv'),
        1: str(data_dir / 'image' / 'video_cam1.mkv'),
        2: str(data_dir / 'image' / 'video_cam2.mkv'),
    }
    cam_in_ex_path = data_dir / 'image' / 'cam_in_ex.txt'
    img_pos_path = data_dir / 'image' / 'img_pos.txt'

    output_path = args.output or str(data_dir / f'colorized_frame_{frame:04d}.ply')

    if not ply_path.exists():
        print(f"ERROR: PLY 文件不存在: {ply_path}")
        sys.exit(1)

    # 解析数据
    print("解析标定数据...")
    cam_data = parse_cam_in_ex(str(cam_in_ex_path))
    print(f"  设备: {cam_data['device']}")
    for cid, cam in cam_data['cameras'].items():
        print(f"  cam{cid}: fx={cam['A11']:.1f} fy={cam['A22']:.1f} "
              f"cx={cam['u0']:.1f} cy={cam['v0']:.1f}")
    print(f"  Tcl: {list(cam_data['Tcl'].keys())}")
    print(f"  Til: {'present' if cam_data['Til'] is not None else 'missing'}")

    print("\n解析位姿数据...")
    img_pos_entries = parse_img_pos(str(img_pos_path))
    print(f"  帧数: {len(img_pos_entries)}")
    print(f"  帧号范围: {min(img_pos_entries.keys())} ~ {max(img_pos_entries.keys())}")

    colorize_frame(
        ply_path=str(ply_path),
        video_paths=video_paths,
        cam_data=cam_data,
        img_pos_entries=img_pos_entries,
        frame_index=frame,
        output_path=output_path,
        specific_camera=args.camera,
    )


if __name__ == '__main__':
    main()