#!/usr/bin/env python3
"""合并点云着色：将所有 PLY section 合并后，用多相机多帧图像进行着色。

思路：
  - PLY 点云已是世界坐标系，直接合并
  - 对每帧的 3 个相机，计算相机世界位姿 T_world_cam
  - 将合并点云投影到每帧图像，采样 RGB 颜色
  - 按深度融合：同一点被多个视角覆盖时，取最近相机的颜色

用法：
    python colorize_merged.py                          # 着色全部帧
    python colorize_merged.py --max-frames 200         # 只用前 200 帧
    python colorize_merged.py --stride 5               # 每隔 5 帧取 1 帧
    python colorize_merged.py --output merged.ply
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
    """解析 cam_in_ex.txt：3 相机内外参 + Til。"""
    import re
    with open(path) as f:
        text = f.read()

    result = {'cameras': {}, 'Tcl': {}, 'Til': None}

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
    """解析 img_pos.txt，返回 dict[idx] = {timestamp, tx, ty, tz, qw, qx, qy, qz}。"""
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


# ============================================================
# 点云读取
# ============================================================

def merge_ply_sections(extracted_dir, section_indices=None):
    """合并多个 PLY section 为一个点云阵列。"""
    if section_indices is None:
        section_indices = sorted([
            int(fn.replace('section_', '').replace('.ply', ''))
            for fn in os.listdir(extracted_dir)
            if fn.startswith('section_') and fn.endswith('.ply')
        ])

    all_points = []
    for idx in section_indices:
        path = os.path.join(extracted_dir, f'section_{idx:04d}.ply')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            in_data = False
            for line in f:
                if line.strip() == 'end_header':
                    in_data = True
                    continue
                if in_data:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        all_points.append([float(parts[0]), float(parts[1]), float(parts[2])])

    return np.array(all_points, dtype=np.float32)


# ============================================================
# Kannala 鱼眼投影
# ============================================================

def kannala_project_batch(points_cam, cam_params, w, h):
    """Kannala 鱼眼模型投影，向量化版本。

    返回:
        pixels: (N, 2) 像素坐标
        valid: (N,) bool
    """
    X, Y, Z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    valid = Z > 0.1

    x = np.where(valid, X / Z, 0.0)
    y = np.where(valid, Y / Z, 0.0)
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan(r)

    k2 = cam_params.get('k2', 0)
    k3 = cam_params.get('k3', 0)
    k4 = cam_params.get('k4', 0)
    k5 = cam_params.get('k5', 0)
    k6 = cam_params.get('k6', 0)
    k7 = cam_params.get('k7', 0)

    t2 = theta * theta
    theta_p = theta + k2 * t2 * theta + k3 * t2 * t2 * theta
    theta_p += k4 * t2**3 * theta + k5 * t2**4 * theta
    theta_p += k6 * t2**5 * theta + k7 * t2**6 * theta
    rho = theta_p

    r_safe = np.where(r > 1e-8, r, 1.0)
    scale = np.where(r > 1e-8, rho / r_safe, 1.0)

    fx = cam_params['A11']
    fy = cam_params['A22']
    skew = cam_params.get('A12', 0)
    cx = cam_params['u0']
    cy = cam_params['v0']

    u = fx * scale * x + skew * scale * y + cx
    v = fy * scale * y + cy

    in_image = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    valid = valid & in_image

    return np.stack([u, v], axis=-1), valid


# ============================================================
# 合并着色
# ============================================================

def colorize_merged(points_world, extracted_dir, video_paths, cam_data,
                    img_pos_entries, section_indices, lidar_to_video,
                    output_path, stride=1, max_frames=None):
    """对合并点云进行多帧多相机着色。

    变换链（世界坐标→相机坐标）：
    P_cam = inv(Tcl) @ inv(T_world_lidar) @ P_world
    其中 T_world_lidar = T_world_body @ Til
    """
    N = len(points_world)
    print(f"\n合并点云: {N:,} 点")

    # 预分配颜色和深度缓冲
    colors = np.zeros((N, 3), dtype=np.uint8)
    best_depth = np.full(N, np.inf, dtype=np.float32)
    colored = np.zeros(N, dtype=bool)

    pts_h = np.hstack([points_world, np.ones((N, 1), dtype=np.float32)])  # (N, 4)

    # 确定使用的帧
    frame_indices = sorted(lidar_to_video.keys())[::stride]
    if max_frames:
        frame_indices = frame_indices[:max_frames]

    print(f"使用 {len(frame_indices)} 帧着色 (stride={stride})")

    # 读取第一帧视频获取尺寸
    cap = cv2.VideoCapture(str(video_paths[0]))
    ret, sample_frame = cap.read()
    cap.release()
    h_img, w_img = sample_frame.shape[:2]

    # 预加载相机参数（只加载一次）
    cam_params_list = []
    for cam_id in range(3):
        cp = cam_data['cameras'][cam_id]
        cam_params_list.append({
            'A11': cp['A11'], 'A12': cp.get('A12', 0), 'A22': cp['A22'],
            'u0': cp['u0'], 'v0': cp['v0'],
            'k2': cp.get('k2', 0), 'k3': cp.get('k3', 0),
            'k4': cp.get('k4', 0), 'k5': cp.get('k5', 0),
            'k6': cp.get('k6', 0), 'k7': cp.get('k7', 0),
            'image_width': w_img, 'image_height': h_img,
        })

    # 逐帧着色
    for i, lidar_idx in enumerate(frame_indices):
        if lidar_idx not in img_pos_entries:
            continue

        entry = img_pos_entries[lidar_idx]
        R = Rotation.from_quat([entry['qx'], entry['qy'], entry['qz'], entry['qw']]).as_matrix()
        T_world_body = np.eye(4, dtype=np.float32)
        T_world_body[:3, :3] = R
        T_world_body[:3, 3] = [entry['tx'], entry['ty'], entry['tz']]
        T_world_lidar = T_world_body @ cam_data['Til']

        video_frame = lidar_to_video[lidar_idx]

        # 逐相机着色
        for cam_id in range(3):
            Tcl = cam_data['Tcl'][cam_id]
            T_lidar_cam = np.linalg.inv(Tcl)
            T_lidar_world = np.linalg.inv(T_world_lidar)
            T_cam_world = T_lidar_cam @ T_lidar_world

            pts_cam = (T_cam_world @ pts_h.T).T[:, :3]

            # 快速剔除：Z <= 0 的点
            front_mask = pts_cam[:, 2] > 0.1
            if not front_mask.any():
                continue

            pixels, valid = kannala_project_batch(pts_cam, cam_params_list[cam_id], w_img, h_img)
            n_valid = valid.sum()
            if n_valid == 0:
                continue

            # 读取视频帧
            cap = cv2.VideoCapture(str(video_paths[cam_id]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                continue

            # 采样颜色（深度融合）
            valid_indices = np.where(valid)[0]
            u_coords = np.clip(np.round(pixels[valid, 0]).astype(np.int32), 0, w_img - 1)
            v_coords = np.clip(np.round(pixels[valid, 1]).astype(np.int32), 0, h_img - 1)
            sampled = frame[v_coords, u_coords, ::-1]  # BGR → RGB

            depths = pts_cam[valid, 2]
            for j, idx in enumerate(valid_indices):
                if depths[j] < best_depth[idx]:
                    best_depth[idx] = depths[j]
                    colors[idx] = sampled[j]
                    colored[idx] = True

        n_colored = colored.sum()
        if (i + 1) % 50 == 0 or i == len(frame_indices) - 1:
            print(f"  [{i+1}/{len(frame_indices)}] 帧 {lidar_idx} (视频帧 {video_frame}) "
                  f"已着色 {n_colored:,} / {N:,} ({n_colored/N*100:.1f}%)")

    # 输出
    write_colored_ply_binary(output_path, points_world, colors, colored)
    print(f"\n{'='*60}")
    print(f"着色结果:")
    print(f"  总点数: {N:,}")
    print(f"  着色点数: {colored.sum():,} ({colored.sum()/N*100:.1f}%)")
    print(f"  未着色点数: {(~colored).sum():,}")
    print(f"  输出: {output_path}")
    print(f"{'='*60}")


def write_colored_ply_binary(path, points, colors, valid_mask,
                             uncolored_color=(128, 128, 128)):
    """写入二进制 PLY 文件（速度快，体积小）。"""
    N = len(points)
    default = np.array(uncolored_color, dtype=np.uint8)

    with open(path, 'wb') as f:
        header = "ply\nformat binary_little_endian 1.0\n"
        header += f"element vertex {N}\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        header += "end_header\n"
        f.write(header.encode())

        out_colors = np.where(valid_mask[:, None], colors, default[None, :])
        for i in range(N):
            f.write(points[i].tobytes())  # 12 bytes: 3×float32
            f.write(out_colors[i].tobytes())  # 3 bytes: 3×uint8


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='合并点云多帧着色')
    parser.add_argument('--stride', type=int, default=1,
                        help='每隔 N 帧取 1 帧 (默认 1)')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='最多使用 N 帧')
    parser.add_argument('--max-sections', type=int, default=None,
                        help='最多合并 N 个 section (默认全部)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出 PLY 路径')
    parser.add_argument('--data-dir', type=str,
                        default='/Users/skkac/Work/SCAN/new_route')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    extracted_dir = str(data_dir / 'extracted')
    video_paths = {
        0: str(data_dir / 'image' / 'video_cam0.mkv'),
        1: str(data_dir / 'image' / 'video_cam1.mkv'),
        2: str(data_dir / 'image' / 'video_cam2.mkv'),
    }
    cam_in_ex_path = str(data_dir / 'image' / 'cam_in_ex.txt')
    img_pos_path = str(data_dir / 'image' / 'img_pos.txt')
    output_path = args.output or str(data_dir / 'merged_colorized.ply')

    # 解析标定
    print("解析标定数据...")
    cam_data = parse_cam_in_ex(cam_in_ex_path)
    print(f"  相机: {list(cam_data['cameras'].keys())}")
    print(f"  Tcl: {list(cam_data['Tcl'].keys())}")
    print(f"  Til: {'present' if cam_data['Til'] is not None else 'missing'}")

    # 解析位姿
    print("解析位姿数据...")
    img_pos_entries = parse_img_pos(img_pos_path)
    print(f"  帧数: {len(img_pos_entries)}")

    # 构建 LiDAR → 视频映射
    t0 = img_pos_entries[min(img_pos_entries.keys())]['timestamp']
    lidar_to_video = {}
    for idx, entry in img_pos_entries.items():
        video_frame = round((entry['timestamp'] - t0) * 10.0)
        if 0 <= video_frame <= 7630:
            lidar_to_video[idx] = video_frame
    print(f"  有视频对应的 LiDAR 帧: {len(lidar_to_video)}")

    # 合并点云
    print("合并 PLY sections...")
    section_indices = sorted([
        int(fn.replace('section_', '').replace('.ply', ''))
        for fn in os.listdir(extracted_dir)
        if fn.startswith('section_') and fn.endswith('.ply')
    ])
    if args.max_sections:
        section_indices = section_indices[:args.max_sections]

    points_world = merge_ply_sections(extracted_dir, section_indices)
    print(f"  合并完成: {len(points_world):,} 点")
    print(f"  X: [{points_world[:,0].min():.1f}, {points_world[:,0].max():.1f}]")
    print(f"  Y: [{points_world[:,1].min():.1f}, {points_world[:,1].max():.1f}]")
    print(f"  Z: [{points_world[:,2].min():.1f}, {points_world[:,2].max():.1f}]")

    # 着色
    colorize_merged(
        points_world=points_world,
        extracted_dir=extracted_dir,
        video_paths=video_paths,
        cam_data=cam_data,
        img_pos_entries=img_pos_entries,
        section_indices=section_indices,
        lidar_to_video=lidar_to_video,
        output_path=output_path,
        stride=args.stride,
        max_frames=args.max_frames,
    )


if __name__ == '__main__':
    main()