#!/usr/bin/env python3
"""逐帧着色 v2：OpenCV fisheye 投影，直接在畸变图上采样。

核心修复：
  - 用 cv2.fisheye.projectPoints (4系数) 替代自实现6系数Kannala
  - 6系数在 theta>65° 时投影偏移最大120像素，4系数在179°内单调
  - 不做去畸变，直接在畸变图像上采样

用法：
    python colorize_v2.py                          # 全量
    python colorize_v2.py --start 0 --end 100
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def parse_cam_in_ex(path):
    import re
    with open(path) as f:
        text = f.read()
    result = {'cameras': {}, 'Tcl': {}, 'Til': None}
    for i in range(3):
        m = re.search(rf'Tcl_{i}:\s*\[([^\]]+)\]', text)
        if m:
            vals = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', m.group(1))]
            result['Tcl'][i] = np.array(vals, dtype=np.float64).reshape(4, 4)
    m = re.search(r'Til:\s*\[([^\]]+)\]', text)
    if m:
        vals = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', m.group(1))]
        result['Til'] = np.array(vals, dtype=np.float64).reshape(4, 4)
    for i in range(3):
        cam = {}
        section = re.search(rf'cam_{i}:\s*(.*?)(?=cam_\d:|Til:|\Z)', text, re.DOTALL)
        if not section:
            continue
        block = section.group(1)
        for key in ['k2', 'k3', 'k4', 'k5', 'A11', 'A12', 'A22', 'u0', 'v0']:
            m = re.search(rf'{key}:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', block)
            if m:
                cam[key] = float(m.group(1))
        result['cameras'][i] = cam
    return result


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


def colorize_frame_opencv(points_world, lidar_idx, cam_data, img_pos_entries,
                           caps, cam_K, cam_D, T_lidar_cam):
    """用 OpenCV fisheye.projectPoints 投影 + 畸变图采样。"""
    N = len(points_world)
    colors = np.zeros((N, 3), dtype=np.uint8)
    best_depth = np.full(N, np.inf, dtype=np.float32)
    colored = np.zeros(N, dtype=bool)

    if lidar_idx not in img_pos_entries:
        return points_world, colors, colored

    entry = img_pos_entries[lidar_idx]
    R = Rotation.from_quat([entry['qx'], entry['qy'], entry['qz'], entry['qw']]).as_matrix()
    cam_pos = np.array([entry['tx'], entry['ty'], entry['tz']], dtype=np.float64)

    T_world_body = np.eye(4, dtype=np.float64)
    T_world_body[:3, :3] = R
    T_world_body[:3, 3] = cam_pos
    T_world_lidar = T_world_body @ cam_data['Til']
    T_lidar_world = np.linalg.inv(T_world_lidar)

    pts_h = np.hstack([points_world, np.ones((N, 1), dtype=np.float32)])

    t0_ts = img_pos_entries[min(img_pos_entries.keys())]['timestamp']
    video_frame = round((entry['timestamp'] - t0_ts) * 10.0)

    for cam_id in range(3):
        T_cam_world = T_lidar_cam[cam_id] @ T_lidar_world
        pts_cam = (T_cam_world @ pts_h.T).T[:, :3]

        front = pts_cam[:, 2] > 0.1
        if not front.any():
            continue

        pts_cam_front = pts_cam[front].astype(np.float64)
        front_mask_idx = np.where(front)[0]

        # OpenCV fisheye 投影
        pts3d = pts_cam_front.reshape(-1, 1, 3)
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

        # 采样颜色 + 深度融合
        valid_local = np.where(in_img)[0]
        valid_global = front_mask_idx[valid_local]

        u_idx = np.clip(np.round(u[in_img]).astype(np.int32), 0, w - 1)
        v_idx = np.clip(np.round(v[in_img]).astype(np.int32), 0, h - 1)
        sampled = frame[v_idx, u_idx, ::-1]  # BGR → RGB
        depths = pts_cam_front[in_img, 2]

        closer = depths < best_depth[valid_global]
        update_idx = valid_global[closer]
        best_depth[update_idx] = depths[closer]
        colors[update_idx] = sampled[closer]
        colored[update_idx] = True

    return points_world, colors, colored


def batch_colorize(extracted_dir, video_paths, cam_data, img_pos_entries,
                   output_path, start=0, end=None, stride=1):

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

    # 预编译相机参数
    cam_K = {}
    cam_D = {}
    T_lidar_cam = {}

    for cam_id in range(3):
        cp = cam_data['cameras'][cam_id]
        cam_K[cam_id] = np.array([
            [cp['A11'], cp.get('A12', 0), cp['u0']],
            [0, cp['A22'], cp['v0']],
            [0, 0, 1]
        ])
        # OpenCV fisheye 用 k1,k2,k3,k4 → 对应 MANIFOLD 的 k2,k3,k4,k5
        cam_D[cam_id] = np.array([
            cp.get('k2', 0), cp.get('k3', 0),
            cp.get('k4', 0), cp.get('k5', 0)
        ])
        T_lidar_cam[cam_id] = np.linalg.inv(cam_data['Tcl'][cam_id])

    # 打开视频
    caps = {}
    for cam_id in range(3):
        caps[cam_id] = cv2.VideoCapture(str(video_paths[cam_id]))

    # 逐帧着色
    all_colored_points = []
    all_colored_colors = []
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

        _, colors, colored_mask = colorize_frame_opencv(
            points, lidar_idx, cam_data, img_pos_entries,
            caps, cam_K, cam_D, T_lidar_cam
        )

        n_colored = colored_mask.sum()
        total_colored += n_colored

        if n_colored > 0:
            all_colored_points.append(points[colored_mask])
            all_colored_colors.append(colors[colored_mask])

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
    merged_points = np.concatenate(all_colored_points, axis=0)
    merged_colors = np.concatenate(all_colored_colors, axis=0)
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
    cam_data = parse_cam_in_ex(str(data_dir / 'image' / 'cam_in_ex.txt'))
    img_pos = parse_img_pos(str(data_dir / 'image' / 'img_pos.txt'))

    batch_colorize(
        extracted_dir=str(data_dir / 'extracted'),
        video_paths={i: str(data_dir / 'image' / f'video_cam{i}.mkv') for i in range(3)},
        cam_data=cam_data, img_pos_entries=img_pos,
        output_path=args.output or str(data_dir / 'colorized_v2.ply'),
        start=args.start, end=args.end, stride=args.stride,
    )


if __name__ == '__main__':
    main()
