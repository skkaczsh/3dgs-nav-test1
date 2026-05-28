#!/usr/bin/env python3
"""逐帧着色 + 删除未着色点 + 拼接为完整彩色点云。

思路：
  - PLY 点云已是世界坐标，逐帧着色时不需要变换点云坐标
  - 每帧 LiDAR 点云投影到 3 路相机图像，深度融合取最近颜色
  - 未着色点直接删除（背面/遮挡点不可靠）
  - 拼接所有帧的着色点，输出完整彩色 PLY

用法：
    python colorize_batch.py                          # 全量着色
    python colorize_batch.py --start 0 --end 100      # 指定范围
    python colorize_batch.py --stride 5               # 每隔 5 帧
    python colorize_batch.py --output result.ply
"""

import argparse
import os
import time
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
            result['Tcl'][i] = np.array(vals, dtype=np.float64).reshape(4, 4)

    m = re.search(r'Til:\s*\[([^\]]+)\]', text)
    if m:
        vals = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', m.group(1))]
        result['Til'] = np.array(vals, dtype=np.float64).reshape(4, 4)

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
    """解析 img_pos.txt。"""
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
# PLY 读取
# ============================================================

def read_ply_points(path):
    """读取 PLY 点云，返回 (N, 3) float32。"""
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


# ============================================================
# Kannala 鱼眼投影
# ============================================================

def kannala_project_batch(points_cam, cam_params, w, h):
    """Kannala 鱼眼模型投影。返回 (pixels, valid)。"""
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
    theta_p = theta * (1 + k2*t2 + k3*t2*t2 + k4*t2**3
                        + k5*t2**4 + k6*t2**5 + k7*t2**6)
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
# 单帧着色
# ============================================================

def colorize_frame(points_world, lidar_idx, cam_data, img_pos_entries,
                   caps, cam_params_list, h_img, w_img, T_lidar_cam):
    """对单帧世界坐标点云着色，返回 (colored_points, colors, colored_mask)。"""
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

    # 视频帧号（从 img_pos 计算）
    t0 = img_pos_entries[min(img_pos_entries.keys())]['timestamp']
    video_frame = round((entry['timestamp'] - t0) * 10.0)

    for cam_id in range(3):
        T_cam_world = T_lidar_cam[cam_id] @ T_lidar_world
        pts_cam = (T_cam_world @ pts_h.T).T[:, :3]

        front = pts_cam[:, 2] > 0.1
        if not front.any():
            continue

        pts_cam_front = pts_cam[front]
        front_mask_idx = np.where(front)[0]

        pixels, valid = kannala_project_batch(
            pts_cam_front, cam_params_list[cam_id], w_img, h_img)
        if not valid.any():
            continue

        # 读视频帧
        caps[cam_id].set(cv2.CAP_PROP_POS_FRAMES, video_frame)
        ret, frame = caps[cam_id].read()
        if not ret:
            continue

        # 采样颜色 + 向量化深度融合
        valid_local = np.where(valid)[0]
        valid_global = front_mask_idx[valid]

        u = np.clip(np.round(pixels[valid, 0]).astype(np.int32), 0, w_img - 1)
        v = np.clip(np.round(pixels[valid, 1]).astype(np.int32), 0, h_img - 1)
        sampled = frame[v, u, ::-1]  # BGR → RGB
        depths = pts_cam_front[valid, 2]

        closer = depths < best_depth[valid_global]
        update_idx = valid_global[closer]
        best_depth[update_idx] = depths[closer]
        colors[update_idx] = sampled[closer]
        colored[update_idx] = True

    return points_world, colors, colored


# ============================================================
# 批量着色主流程
# ============================================================

def batch_colorize(extracted_dir, video_paths, cam_data, img_pos_entries,
                   output_path, start=0, end=None, stride=1):
    """逐帧着色 + 删除未着色点 + 拼接。"""

    # 构建帧列表
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

    # 视频覆盖范围
    t0_ts = img_pos_entries[min(img_pos_entries.keys())]['timestamp']
    last_video_frame = max(
        round((e['timestamp'] - t0_ts) * 10.0)
        for e in img_pos_entries.values()
    )

    # 分离：有视频 vs 无视频
    has_video = []
    no_video = []
    for s in all_sections:
        if s not in img_pos_entries:
            no_video.append(s)
            continue
        vf = round((img_pos_entries[s]['timestamp'] - t0_ts) * 10.0)
        if 0 <= vf <= last_video_frame:
            has_video.append(s)
        else:
            no_video.append(s)

    print(f"总 section: {len(all_sections)}")
    print(f"有视频覆盖: {len(has_video)}")
    print(f"无视频覆盖: {len(no_video)}（跳过）")

    # 打开视频
    print("打开视频文件...")
    caps = {}
    for cam_id in range(3):
        caps[cam_id] = cv2.VideoCapture(str(video_paths[cam_id]))
    ret, sample = caps[0].read()
    h_img, w_img = sample.shape[:2]
    caps[0].set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"  视频尺寸: {w_img}×{h_img}")

    # 预编译相机参数
    cam_params_list = []
    for cam_id in range(3):
        cp = cam_data['cameras'][cam_id]
        cam_params_list.append({
            'A11': cp['A11'], 'A12': cp.get('A12', 0), 'A22': cp['A22'],
            'u0': cp['u0'], 'v0': cp['v0'],
            'k2': cp.get('k2', 0), 'k3': cp.get('k3', 0),
            'k4': cp.get('k4', 0), 'k5': cp.get('k5', 0),
            'k6': cp.get('k6', 0), 'k7': cp.get('k7', 0),
        })

    # 预计算 T_lidar_cam
    T_lidar_cam = {}
    for cam_id in range(3):
        T_lidar_cam[cam_id] = np.linalg.inv(cam_data['Tcl'][cam_id])

    # 收集所有着色点
    all_colored_points = []
    all_colored_colors = []

    total_input_points = 0
    total_colored_points = 0
    t_start = time.time()
    last_report = time.time()

    for i, lidar_idx in enumerate(has_video):
        ply_path = os.path.join(extracted_dir, f'section_{lidar_idx:04d}.ply')
        if not os.path.exists(ply_path):
            continue

        points_world = read_ply_points(ply_path)
        n_pts = len(points_world)
        total_input_points += n_pts

        _, colors, colored_mask = colorize_frame(
            points_world, lidar_idx, cam_data, img_pos_entries,
            caps, cam_params_list, h_img, w_img, T_lidar_cam
        )

        n_colored = colored_mask.sum()
        total_colored_points += n_colored

        # 只保留着色点
        if n_colored > 0:
            all_colored_points.append(points_world[colored_mask])
            all_colored_colors.append(colors[colored_mask])

        # 进度
        now = time.time()
        if now - last_report >= 10 or i == len(has_video) - 1:
            elapsed = now - t_start
            fps = (i + 1) / elapsed
            eta = (len(has_video) - i - 1) / fps if fps > 0 else 0
            rate = total_colored_points / total_input_points * 100 if total_input_points > 0 else 0
            print(f"  [{i+1}/{len(has_video)}] 帧 {lidar_idx} "
                  f"本帧 {n_colored}/{n_pts} ({n_colored/n_pts*100:.0f}%) "
                  f"总着色率 {rate:.1f}% "
                  f"速度 {fps:.0f} fps ETA {eta/60:.1f}min")
            last_report = now

    # 释放视频
    for cap in caps.values():
        cap.release()

    elapsed = time.time() - t_start
    print(f"\n着色耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # 拼接
    print("拼接着色点云...")
    merged_points = np.concatenate(all_colored_points, axis=0)
    merged_colors = np.concatenate(all_colored_colors, axis=0)

    print(f"  输入总点数: {total_input_points:,}")
    print(f"  着色点数: {total_colored_points:,} ({total_colored_points/total_input_points*100:.1f}%)")
    print(f"  删除点数: {total_input_points - total_colored_points:,}")

    # 写入
    write_ply_binary(output_path, merged_points, merged_colors)


def write_ply_binary(path, points, colors):
    """写入二进制 PLY。"""
    N = len(points)
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {N}\n"
    header += "property float x\nproperty float y\nproperty float z\n"
    header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
    header += "end_header\n"

    dtype = np.dtype([('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                      ('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
    buf = np.empty(N, dtype=dtype)
    buf['x'] = points[:, 0]
    buf['y'] = points[:, 1]
    buf['z'] = points[:, 2]
    buf['r'] = colors[:, 0]
    buf['g'] = colors[:, 1]
    buf['b'] = colors[:, 2]

    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(buf.tobytes())

    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  输出: {path} ({size_mb:.0f} MB, {N:,} 点)")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='逐帧着色 + 删除未着色点 + 拼接')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--output', type=str, default=None)
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
    output_path = args.output or str(data_dir / 'colorized_full.ply')

    print("解析标定数据...")
    cam_data = parse_cam_in_ex(cam_in_ex_path)

    print("解析位姿数据...")
    img_pos_entries = parse_img_pos(img_pos_path)
    print(f"  帧数: {len(img_pos_entries)}")

    batch_colorize(
        extracted_dir=extracted_dir,
        video_paths=video_paths,
        cam_data=cam_data,
        img_pos_entries=img_pos_entries,
        output_path=output_path,
        start=args.start,
        end=args.end,
        stride=args.stride,
    )


if __name__ == '__main__':
    main()
