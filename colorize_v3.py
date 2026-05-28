#!/usr/bin/env python3
"""逐帧着色 v3：Convention B (Tcl = LiDAR→Cam) + OpenCV fisheye。

验证结论：
  - Tcl 矩阵含义：P_camera = Tcl @ P_lidar（LiDAR→Camera）
  - LiDAR 环在图像中水平（平行宽度方向）
  - 使用 calib_online_final.yaml 的 Tcl（更准确）
  - 使用 OpenCV fisheye.projectPoints (4系数)

用法：
    python colorize_v3.py
    python colorize_v3.py --end 200
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation


def parse_img_pos(path):
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


def read_ply_points(path):
    points = []
    with open(path) as f:
        in_data = False
        for line in f:
            if line.strip() == 'end_header':
                in_data = True
                continue
            if in_data:
                parts = line.strip().split()
                if len(parts) >= 3:
                    points.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.array(points, dtype=np.float32)


def colorize_frame(points_world, lidar_idx, img_pos_entries,
                   caps, cam_K, cam_D, Tcl, Til, t0_ts):
    """Convention B: T_cam = Tcl @ T_lidar_world"""
    N = len(points_world)
    colors = np.zeros((N, 3), dtype=np.uint8)
    best_depth = np.full(N, np.inf, dtype=np.float32)
    colored = np.zeros(N, dtype=bool)

    if lidar_idx not in img_pos_entries:
        return colors, colored

    entry = img_pos_entries[lidar_idx]
    R = Rotation.from_quat([entry['qx'], entry['qy'], entry['qz'], entry['qw']]).as_matrix()
    T_world_body = np.eye(4, dtype=np.float64)
    T_world_body[:3, :3] = R
    T_world_body[:3, 3] = [entry['tx'], entry['ty'], entry['tz']]
    T_world_lidar = T_world_body @ Til
    T_lidar_world = np.linalg.inv(T_world_lidar)

    pts_h = np.hstack([points_world, np.ones((N, 1), dtype=np.float32)])
    video_frame = round((entry['timestamp'] - t0_ts) * 10.0)

    for cam_id in range(3):
        # Convention B: T_cam = Tcl @ T_lidar_world
        T_cam_world = Tcl[cam_id] @ T_lidar_world
        pts_cam = (T_cam_world @ pts_h.T).T[:, :3]

        front = pts_cam[:, 2] > 0.1
        if not front.any():
            continue

        pts_front = pts_cam[front].astype(np.float64)
        front_idx = np.where(front)[0]

        # OpenCV fisheye 投影
        pts3d = pts_front.reshape(-1, 1, 3)
        projected, _ = cv2.fisheye.projectPoints(
            pts3d, np.zeros(3), np.zeros(3), cam_K[cam_id], cam_D[cam_id])

        u = projected[:, 0, 0]
        v = projected[:, 0, 1]
        w, h = 1600, 1296
        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not in_img.any():
            continue

        # 读取视频帧
        caps[cam_id].set(cv2.CAP_PROP_POS_FRAMES, video_frame)
        ret, frame = caps[cam_id].read()
        if not ret:
            continue

        # 采样 + 深度融合
        valid_local = np.where(in_img)[0]
        valid_global = front_idx[valid_local]

        u_idx = np.clip(np.round(u[in_img]).astype(np.int32), 0, w - 1)
        v_idx = np.clip(np.round(v[in_img]).astype(np.int32), 0, h - 1)
        sampled = frame[v_idx, u_idx, ::-1]  # BGR → RGB
        depths = pts_front[in_img, 2]

        closer = depths < best_depth[valid_global]
        update_idx = valid_global[closer]
        best_depth[update_idx] = depths[closer]
        colors[update_idx] = sampled[closer]
        colored[update_idx] = True

    return colors, colored


def batch_colorize(extracted_dir, video_paths, calib_path, img_pos_path,
                   output_path, start=0, end=None, stride=1):

    # 加载标定
    with open(calib_path) as f:
        calib = yaml.safe_load(f)

    Tcl = {}
    for i in range(3):
        Tcl[i] = np.array(calib[f'Tcl_{i}'], dtype=np.float64).reshape(4, 4)
    Til = np.array(calib['Til'], dtype=np.float64).reshape(4, 4)

    # 相机内参和畸变（从 image/cam_in_ex.txt 读取）
    import re
    with open(str(Path(calib_path).parent / 'image' / 'cam_in_ex.txt')) as f:
        text = f.read()

    cam_K = {}
    cam_D = {}
    for i in range(3):
        section = re.search(rf'cam_{i}:\s*(.*?)(?=cam_\d:|Til:|\Z)', text, re.DOTALL)
        if not section:
            continue
        block = section.group(1)
        vals = {}
        for key in ['A11', 'A12', 'A22', 'u0', 'v0', 'k2', 'k3', 'k4', 'k5']:
            m = re.search(rf'{key}:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', block)
            if m:
                vals[key] = float(m.group(1))
        cam_K[i] = np.array([[vals['A11'], vals.get('A12', 0), vals['u0']],
                              [0, vals['A22'], vals['v0']],
                              [0, 0, 1]])
        cam_D[i] = np.array([vals['k2'], vals['k3'], vals['k4'], vals['k5']])

    # 加载位姿
    img_pos_entries = parse_img_pos(img_pos_path)

    # 帧列表
    all_sections = sorted([
        int(fn.replace('section_', '').replace('.ply', ''))
        for fn in os.listdir(extracted_dir)
        if fn.startswith('section_') and fn.endswith('.ply')
    ])
    if end is not None:
        all_sections = [s for s in all_sections if start <= s < end]
    else:
        all_sections = [s for s in all_sections if s >= start]
    all_sections = all_sections[::stride]

    t0_ts = img_pos_entries[min(img_pos_entries.keys())]['timestamp']
    has_video = []
    for s in all_sections:
        if s not in img_pos_entries:
            continue
        vf = round((img_pos_entries[s]['timestamp'] - t0_ts) * 10.0)
        if 0 <= vf < 7630:
            has_video.append(s)

    print(f"总 section: {len(all_sections)}, 有视频: {len(has_video)}")

    # 打开视频
    caps = {}
    for cam_id in range(3):
        caps[cam_id] = cv2.VideoCapture(str(video_paths[cam_id]))

    # 逐帧着色
    all_points = []
    all_colors = []
    total_input = 0
    total_colored = 0
    t_start = time.time()
    last_report = time.time()

    for i, lidar_idx in enumerate(has_video):
        ply_path = os.path.join(extracted_dir, f'section_{lidar_idx:04d}.ply')
        if not os.path.exists(ply_path):
            continue

        points = read_ply_points(ply_path)
        n_pts = len(points)
        total_input += n_pts

        colors, colored_mask = colorize_frame(
            points, lidar_idx, img_pos_entries,
            caps, cam_K, cam_D, Tcl, Til, t0_ts
        )

        n_colored = colored_mask.sum()
        total_colored += n_colored

        if n_colored > 0:
            all_points.append(points[colored_mask])
            all_colors.append(colors[colored_mask])

        now = time.time()
        if now - last_report >= 10 or i == len(has_video) - 1:
            elapsed = now - t_start
            fps = (i + 1) / elapsed
            eta = (len(has_video) - i - 1) / fps if fps > 0 else 0
            rate = total_colored / total_input * 100 if total_input > 0 else 0
            print(f"  [{i+1}/{len(has_video)}] 帧 {lidar_idx} "
                  f"本帧 {n_colored}/{n_pts} ({n_colored/n_pts*100:.0f}%) "
                  f"总着色率 {rate:.1f}% "
                  f"速度 {fps:.0f} fps ETA {eta/60:.1f}min")
            last_report = now

    for cap in caps.values():
        cap.release()

    elapsed = time.time() - t_start
    print(f"\n着色耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    print("拼接着色点云...")
    merged_points = np.concatenate(all_points, axis=0)
    merged_colors = np.concatenate(all_colors, axis=0)
    print(f"  输入: {total_input:,}, 着色: {total_colored:,} ({total_colored/total_input*100:.1f}%), "
          f"删除: {total_input-total_colored:,}")

    write_ply_binary(output_path, merged_points, merged_colors)


def write_ply_binary(path, points, colors):
    N = len(points)
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {N}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "end_header\n"
    dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                      ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    buf = np.empty(N, dtype=dtype)
    buf['x'] = points[:, 0]; buf['y'] = points[:, 1]; buf['z'] = points[:, 2]
    buf['r'] = colors[:, 0]; buf['g'] = colors[:, 1]; buf['b'] = colors[:, 2]
    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(buf.tobytes())
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  输出: {path} ({size_mb:.0f} MB, {N:,} 点)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--data-dir', type=str, default='/Users/skkac/Work/SCAN/new_route')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    batch_colorize(
        extracted_dir=str(data_dir / 'extracted'),
        video_paths={i: str(data_dir / 'image' / f'video_cam{i}.mkv') for i in range(3)},
        calib_path=str(data_dir / 'calib_online_final.yaml'),
        img_pos_path=str(data_dir / 'image' / 'img_pos.txt'),
        output_path=args.output or str(data_dir / 'colorized_v3.ply'),
        start=args.start, end=args.end, stride=args.stride,
    )


if __name__ == '__main__':
    main()
