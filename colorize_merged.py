#!/usr/bin/env python3
"""合并点云着色：将所有 PLY section 合并后，用多相机多帧图像进行着色。

核心优化：
  - 体素空间索引：预构建 3D 体素网格，每帧只处理相机附近体素内的点
  - 向量化深度融合：无 Python 逐点循环
  - 批量二进制 PLY 输出

用法：
    python colorize_merged.py                          # 全量着色
    python colorize_merged.py --stride 5               # 每隔 5 帧取 1 帧
    python colorize_merged.py --voxel 10               # 体素大小 10m
    python colorize_merged.py --output merged.ply
"""

import argparse
import os
import sys
import time
from collections import defaultdict
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
# 体素空间索引
# ============================================================

class VoxelIndex:
    """3D 体素空间索引：快速查找相机附近体素内的点。"""

    def __init__(self, points, voxel_size=10.0):
        self.voxel_size = voxel_size
        self.points = points
        self.N = len(points)

        # 计算每个点所属体素
        self.voxel_coords = np.floor(points[:, :3] / voxel_size).astype(np.int32)

        # 建立体素 → 点索引映射
        self.voxel_map = defaultdict(list)
        for i in range(self.N):
            key = (self.voxel_coords[i, 0], self.voxel_coords[i, 1], self.voxel_coords[i, 2])
            self.voxel_map[key].append(i)

        # 转为 numpy 数组加速
        self.voxel_arrays = {}
        for key, indices in self.voxel_map.items():
            self.voxel_arrays[key] = np.array(indices, dtype=np.int64)

        print(f"  体素索引: {len(self.voxel_map)} 个体素, "
              f"平均 {self.N / len(self.voxel_map):.0f} 点/体素")

    def query_nearby(self, center, radius):
        """查询中心点 radius 范围内的所有点索引。"""
        r_voxels = int(np.ceil(radius / self.voxel_size))
        cx, cy, cz = np.floor(center / self.voxel_size).astype(np.int32)

        index_lists = []
        for dx in range(-r_voxels, r_voxels + 1):
            for dy in range(-r_voxels, r_voxels + 1):
                for dz in range(-r_voxels, r_voxels + 1):
                    key = (cx + dx, cy + dy, cz + dz)
                    if key in self.voxel_arrays:
                        index_lists.append(self.voxel_arrays[key])

        if not index_lists:
            return np.array([], dtype=np.int64)

        return np.concatenate(index_lists)


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

    # 先计算总点数
    total = 0
    for idx in section_indices:
        path = os.path.join(extracted_dir, f'section_{idx:04d}.ply')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                if 'element vertex' in line:
                    total += int(line.split()[-1])
                    break

    # 预分配
    all_points = np.empty((total, 3), dtype=np.float32)
    offset = 0

    for i, idx in enumerate(section_indices):
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
                        all_points[offset, 0] = float(parts[0])
                        all_points[offset, 1] = float(parts[1])
                        all_points[offset, 2] = float(parts[2])
                        offset += 1

        if (i + 1) % 500 == 0:
            print(f"    已加载 {i+1}/{len(section_indices)} sections, "
                  f"{offset:,} 点")

    return all_points[:offset]


# ============================================================
# Kannala 鱼眼投影
# ============================================================

def kannala_project_batch(points_cam, cam_params, w, h):
    """Kannala 鱼眼模型投影，向量化版本。"""
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
    theta_p = theta * (1 + k2*t2 + k3*t2*t2 + k4*t2**3 + k5*t2**4 + k6*t2**5 + k7*t2**6)
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

def colorize_merged(points_world, cam_data, img_pos_entries,
                    lidar_to_video, video_paths, output_path,
                    stride=1, max_frames=None, voxel_size=10.0,
                    search_radius=80.0):
    """对合并点云进行多帧多相机着色。"""
    N = len(points_world)
    print(f"\n合并点云: {N:,} 点")
    print(f"体素大小: {voxel_size}m, 搜索半径: {search_radius}m")

    # 构建体素索引
    print("构建体素空间索引...")
    t0 = time.time()
    vidx = VoxelIndex(points_world, voxel_size=voxel_size)
    print(f"  耗时: {time.time()-t0:.1f}s")

    # 预分配
    colors = np.zeros((N, 3), dtype=np.uint8)
    best_depth = np.full(N, np.inf, dtype=np.float32)
    colored = np.zeros(N, dtype=bool)

    # 帧列表
    frame_indices = sorted(lidar_to_video.keys())[::stride]
    if max_frames:
        frame_indices = frame_indices[:max_frames]
    n_frames = len(frame_indices)
    print(f"使用 {n_frames} 帧着色 (stride={stride})")

    # 视频尺寸 + 持续打开 3 路 VideoCapture
    print("打开视频文件...")
    caps = {}
    for cam_id in range(3):
        caps[cam_id] = cv2.VideoCapture(str(video_paths[cam_id]))
    ret, sample = caps[0].read()
    h_img, w_img = sample.shape[:2]
    caps[0].set(cv2.CAP_PROP_POS_FRAMES, 0)  # reset
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

    t_start = time.time()
    last_progress = time.time()

    # 逐帧着色
    for i, lidar_idx in enumerate(frame_indices):
        if lidar_idx not in img_pos_entries:
            continue

        entry = img_pos_entries[lidar_idx]
        R = Rotation.from_quat([entry['qx'], entry['qy'], entry['qz'], entry['qw']]).as_matrix()
        cam_pos = np.array([entry['tx'], entry['ty'], entry['tz']], dtype=np.float64)

        T_world_body = np.eye(4, dtype=np.float64)
        T_world_body[:3, :3] = R
        T_world_body[:3, 3] = cam_pos
        T_world_lidar = T_world_body @ cam_data['Til']
        T_lidar_world = np.linalg.inv(T_world_lidar)

        video_frame = lidar_to_video[lidar_idx]

        # 体素查询：只获取相机附近的点
        nearby_indices = vidx.query_nearby(cam_pos, search_radius)
        if len(nearby_indices) == 0:
            continue

        # 只投影未着色点（大幅加速）
        uncolored_mask = ~colored[nearby_indices]
        n_uncolored_nearby = uncolored_mask.sum()

        if n_uncolored_nearby == 0:
            # 附近点全部已着色，跳过此帧
            continue

        proj_indices = nearby_indices[uncolored_mask]

        pts_nearby = points_world[proj_indices]
        pts_h = np.hstack([pts_nearby, np.ones((len(pts_nearby), 1), dtype=np.float32)])

        # 逐相机着色
        for cam_id in range(3):
            T_cam_world = T_lidar_cam[cam_id] @ T_lidar_world

            pts_cam = (T_cam_world @ pts_h.T).T[:, :3]

            # 前方点
            front = pts_cam[:, 2] > 0.1
            if not front.any():
                continue

            pts_cam_front = pts_cam[front]
            front_indices = proj_indices[front]

            pixels, valid = kannala_project_batch(
                pts_cam_front, cam_params_list[cam_id], w_img, h_img)
            if not valid.any():
                continue

            # 读视频帧（保持 cap 打开，seek）
            caps[cam_id].set(cv2.CAP_PROP_POS_FRAMES, video_frame)
            ret, frame = caps[cam_id].read()
            if not ret:
                continue

            # 采样颜色 + 向量化深度融合
            valid_global = front_indices[valid]
            u = np.clip(np.round(pixels[valid, 0]).astype(np.int32), 0, w_img - 1)
            v = np.clip(np.round(pixels[valid, 1]).astype(np.int32), 0, h_img - 1)
            sampled = frame[v, u, ::-1]  # BGR → RGB
            depths = pts_cam_front[valid, 2]

            closer = depths < best_depth[valid_global]
            update_idx = valid_global[closer]
            best_depth[update_idx] = depths[closer]
            colors[update_idx] = sampled[closer]
            colored[update_idx] = True

        # 进度（每 10 秒或最后一帧输出）
        now = time.time()
        if now - last_progress >= 10 or i == n_frames - 1:
            n_colored = colored.sum()
            elapsed = now - t_start
            fps = (i + 1) / elapsed
            eta = (n_frames - i - 1) / fps if fps > 0 else 0
            uncolored_count = (~colored).sum()
            print(f"  [{i+1}/{n_frames}] 帧 {lidar_idx} "
                  f"着色 {n_colored:,}/{N:,} ({n_colored/N*100:.1f}%) "
                  f"未着色 {uncolored_count:,} "
                  f"速度 {fps:.1f} fps ETA {eta/60:.1f}min")
            last_progress = now

            # 早期终止：99.9% 已着色时退出
            if n_colored / N > 0.999:
                print(f"  → 着色率 >99.9%，提前终止")
                break

    # 释放视频
    for cap in caps.values():
        cap.release()

    elapsed = time.time() - t_start
    print(f"\n着色耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")

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
    """写入二进制 PLY 文件。"""
    N = len(points)
    default = np.array(uncolored_color, dtype=np.uint8)
    out_colors = np.where(valid_mask[:, None], colors, default[None, :])

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
    buf['r'] = out_colors[:, 0]
    buf['g'] = out_colors[:, 1]
    buf['b'] = out_colors[:, 2]

    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(buf.tobytes())

    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  PLY 文件大小: {size_mb:.1f} MB")


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
    parser.add_argument('--voxel', type=float, default=10.0,
                        help='体素大小 (m)，越小越精确但内存更多 (默认 10)')
    parser.add_argument('--radius', type=float, default=80.0,
                        help='搜索半径 (m)，超出此距离的点不考虑 (默认 80)')
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
    t0_ts = img_pos_entries[min(img_pos_entries.keys())]['timestamp']
    lidar_to_video = {}
    for idx, entry in img_pos_entries.items():
        video_frame = round((entry['timestamp'] - t0_ts) * 10.0)
        if 0 <= video_frame <= 7630:
            lidar_to_video[idx] = video_frame
    print(f"  有视频对应的 LiDAR 帧: {len(lidar_to_video)}")

    # 合并点云
    print("合并 PLY sections...")
    t_merge = time.time()
    section_indices = sorted([
        int(fn.replace('section_', '').replace('.ply', ''))
        for fn in os.listdir(extracted_dir)
        if fn.startswith('section_') and fn.endswith('.ply')
    ])
    if args.max_sections:
        section_indices = section_indices[:args.max_sections]

    points_world = merge_ply_sections(extracted_dir, section_indices)
    print(f"  合并完成: {len(points_world):,} 点 ({time.time()-t_merge:.1f}s)")
    print(f"  X: [{points_world[:,0].min():.1f}, {points_world[:,0].max():.1f}]")
    print(f"  Y: [{points_world[:,1].min():.1f}, {points_world[:,1].max():.1f}]")
    print(f"  Z: [{points_world[:,2].min():.1f}, {points_world[:,2].max():.1f}]")

    # 着色
    colorize_merged(
        points_world=points_world,
        cam_data=cam_data,
        img_pos_entries=img_pos_entries,
        lidar_to_video=lidar_to_video,
        video_paths=video_paths,
        output_path=output_path,
        stride=args.stride,
        max_frames=args.max_frames,
        voxel_size=args.voxel,
        search_radius=args.radius,
    )


if __name__ == '__main__':
    main()
