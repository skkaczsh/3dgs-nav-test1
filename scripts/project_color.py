#!/usr/bin/env python3
"""步骤2: 点云投影赋色 (多进程版本)

将 section_XXXX.ply 点云帧投影到对应相机图像，
根据投影像素位置采样图像颜色，生成带颜色的点云文件。

使用 multiprocessing.Pool 并行处理各帧。

使用方法:
    python project_color.py [--start 0] [--end 99] [--cams 0 1 2]
        [--skip-existing] [--max-points 50000] [--workers 16]
python project_color.py --start 0 --end 7631 --workers 128 --max-points 50000

"""

import os
import sys
import argparse
import time
import struct
import numpy as np
import cv2
import multiprocessing as mp

# ==================== 全局变量 (fork 后子进程共享只读副本) ====================
_worker_cam_ids = None
_worker_max_points = None
_worker_skip_existing = None
_worker_sky_mask_dir = None
_worker_sky_threshold = None


def load_ply_xyz(ply_path):
    """读取只含 XYZ 或前 3 个字段为 XYZ 的 PLY，支持 ASCII 和 binary_little_endian。"""
    with open(ply_path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                break
            s = line.decode("utf-8", errors="replace").strip()
            header.append(s)
            if s == "end_header":
                break

        fmt = "ascii"
        vertex_count = 0
        properties = []
        in_vertex = False
        for s in header:
            if s.startswith("format "):
                fmt = s.split()[1]
            elif s.startswith("element vertex"):
                vertex_count = int(s.split()[-1])
                in_vertex = True
            elif s.startswith("element "):
                in_vertex = False
            elif in_vertex and s.startswith("property "):
                properties.append(s.split()[1])

        if vertex_count <= 0:
            return np.empty((0, 3), dtype=np.float64)

        if fmt == "ascii":
            pts = []
            for line in f:
                p = line.decode("utf-8", errors="replace").strip().split()
                if len(p) >= 3:
                    try:
                        pts.append([float(p[0]), float(p[1]), float(p[2])])
                    except ValueError:
                        continue
            return np.array(pts, dtype=np.float64)

        if fmt != "binary_little_endian":
            raise ValueError(f"Unsupported PLY format: {fmt}")

        # 当前数据 section 为 float x/y/z。若后续字段扩展，按常见标量大小跳过。
        type_sizes = {
            "char": 1, "uchar": 1, "int8": 1, "uint8": 1,
            "short": 2, "ushort": 2, "int16": 2, "uint16": 2,
            "int": 4, "uint": 4, "float": 4, "float32": 4,
            "double": 8, "float64": 8,
        }
        stride = sum(type_sizes.get(t, 4) for t in properties)
        if len(properties) < 3 or stride < 12:
            return np.empty((0, 3), dtype=np.float64)
        raw = f.read(vertex_count * stride)
        pts = np.empty((vertex_count, 3), dtype=np.float64)
        for i in range(vertex_count):
            off = i * stride
            pts[i] = struct.unpack_from("<fff", raw, off)
        return pts


def _init_worker(cam_ids, max_points, skip_existing, sky_mask_dir, sky_threshold):
    """进程池初始化: 接收参数 (通过共享内存副本)"""
    global _worker_cam_ids, _worker_max_points, _worker_skip_existing
    global _worker_sky_mask_dir, _worker_sky_threshold
    _worker_cam_ids = cam_ids
    _worker_max_points = max_points
    _worker_skip_existing = skip_existing
    _worker_sky_mask_dir = sky_mask_dir
    _worker_sky_threshold = sky_threshold


def find_sky_mask_path(mask_dir, cam_id, frame_id):
    """查找 sky mask，mask 中 >= threshold 表示天空。"""
    if not mask_dir:
        return None
    candidates = [
        os.path.join(mask_dir, f"cam{cam_id}_{frame_id:07d}_sky.png"),
        os.path.join(mask_dir, f"cam{cam_id}_{frame_id:05d}_sky.png"),
        os.path.join(mask_dir, f"cam{cam_id}_{frame_id:04d}_sky.png"),
        os.path.join(mask_dir, f"cam{cam_id}_{frame_id:06d}_final_mask.png"),
        os.path.join(mask_dir, f"cam{cam_id}_{frame_id:04d}_final_mask.png"),
        os.path.join(mask_dir, f"cam{cam_id}_{frame_id:06d}.png"),
        os.path.join(mask_dir, f"cam{cam_id}_{frame_id:04d}.png"),
        os.path.join(mask_dir, f"frame_{frame_id:06d}_cam{cam_id}.png"),
        os.path.join(mask_dir, f"frame_{frame_id:04d}_cam{cam_id}.png"),
        os.path.join(mask_dir, f"cam{cam_id}", f"frame_{frame_id:06d}.png"),
        os.path.join(mask_dir, f"cam{cam_id}", f"frame_{frame_id:04d}.png"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _project_and_save(args_dict):
    """在子进程中执行: 加载点云和图像，投影，保存

    参数通过字典传递 (可 pickle)。
    返回: (frame_id, status, colored_count, total_count)
    """
    global _worker_cam_ids, _worker_max_points, _worker_skip_existing

    frame_id   = args_dict["frame_id"]
    pcd_path   = args_dict["pcd_path"]
    output_path = args_dict["output_path"]
    R_rw       = args_dict["R_rw"]       # 3x3 ndarray
    t_rw       = args_dict["t_rw"]       # 3-element ndarray

    # 子进程内导入 config (避免 pickle 大对象)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import FRAME_OUTPUT_DIRS, CAMERA_PARAMS, Tcl, Til, get_inv_pose

    if _worker_skip_existing and os.path.exists(output_path):
        return frame_id, "skip", 0, 0

    if pcd_path is None or not os.path.exists(pcd_path):
        return frame_id, "no_pcd", 0, 0

    # 加载点云
    points_world = load_ply_xyz(pcd_path)
    if len(points_world) == 0:
        return frame_id, "empty_pcd", 0, 0

    if _worker_max_points and len(points_world) > _worker_max_points:
        idx = np.random.choice(len(points_world), _worker_max_points, replace=False)
        points_world = points_world[idx]

    # 加载图像
    img_dict = {}
    sky_dict = {}
    for cam_id in _worker_cam_ids:
        img_path = os.path.join(FRAME_OUTPUT_DIRS[cam_id], f"frame_{frame_id:04d}.png")
        if os.path.exists(img_path):
            img_dict[cam_id] = cv2.imread(img_path)
        else:
            img_dict[cam_id] = None
        mask_path = find_sky_mask_path(_worker_sky_mask_dir, cam_id, frame_id)
        sky_dict[cam_id] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None

    if all(v is None for v in img_dict.values()):
        return frame_id, "no_img", 0, 0

    # 投影
    N = points_world.shape[0]
    colors = np.zeros((N, 3), dtype=np.uint8)
    point_depths = np.full(N, np.inf, dtype=np.float64)

    R_wr = R_rw.T
    t_wr = (-R_wr @ t_rw).reshape(3)
    R_li = Til[:3, :3].T
    t_li = (-R_li @ Til[:3, 3]).reshape(3)

    for cam_id in _worker_cam_ids:
        img = img_dict[cam_id]
        if img is None:
            continue

        K = CAMERA_PARAMS[cam_id]["K"]
        T_cl = Tcl[cam_id]
        R_cl = T_cl[:3, :3]
        t_cl = T_cl[:3, 3]

        # 世界 -> 相机
        P_robot = (R_wr @ points_world.T + t_wr.reshape(3, 1)).T
        P_lidar = (R_li @ P_robot.T + t_li.reshape(3, 1)).T
        P_cam   = (R_cl @ P_lidar.T + t_cl.reshape(3, 1)).T

        z = P_cam[:, 2]
        valid = z > 0.1
        if not np.any(valid):
            continue

        uv_h = (K @ P_cam[valid].T).T
        u = uv_h[:, 0] / uv_h[:, 2]
        v = uv_h[:, 1] / uv_h[:, 2]
        H, W = img.shape[:2]
        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not np.any(in_img):
            continue

        u_v = u[in_img]; v_v = v[in_img]
        z_v = z[valid][in_img]
        valid_idx = np.where(valid)[0][in_img]

        sky_mask = sky_dict.get(cam_id)
        if sky_mask is not None:
            if sky_mask.shape[:2] != (H, W):
                sky_mask = cv2.resize(sky_mask, (W, H), interpolation=cv2.INTER_NEAREST)
            mu = np.clip(u_v.astype(np.int32), 0, W - 1)
            mv = np.clip(v_v.astype(np.int32), 0, H - 1)
            not_sky = sky_mask[mv, mu] < _worker_sky_threshold
            if not np.any(not_sky):
                continue
            u_v = u_v[not_sky]
            v_v = v_v[not_sky]
            z_v = z_v[not_sky]
            valid_idx = valid_idx[not_sky]

        # 双线性插值采样
        u0 = u_v.astype(np.int32); v0 = v_v.astype(np.int32)
        su = u_v - u0; sv = v_v - v0
        c00 = img[np.clip(v0, 0, H-1), np.clip(u0,   0, W-1)]
        c10 = img[np.clip(v0, 0, H-1), np.clip(u0+1, 0, W-1)]
        c01 = img[np.clip(v0+1, 0, H-1), np.clip(u0,   0, W-1)]
        c11 = img[np.clip(v0+1, 0, H-1), np.clip(u0+1, 0, W-1)]
        sampled = (c00*(1-su[:,None])*(1-sv[:,None]) +
                   c10*su[:,None]*(1-sv[:,None]) +
                   c01*(1-su[:,None])*sv[:,None] +
                   c11*su[:,None]*sv[:,None])
        sampled = np.clip(sampled, 0, 255).astype(np.uint8)
        # BGR (OpenCV) -> RGB (PLY)
        sampled = sampled[:, ::-1]

        for j, idx in enumerate(valid_idx):
            if z_v[j] < point_depths[idx]:
                colors[idx] = sampled[j]
                point_depths[idx] = z_v[j]

    colored = int(np.sum(colors.sum(axis=1) > 0))

    try:
        with open(output_path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {N}\n")
            f.write("property float x\nproperty float y\nproperty float z\n"
                    "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                    "end_header\n")
            for i in range(N):
                p = points_world[i]; c = colors[i]
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                        f"{c[0]} {c[1]} {c[2]}\n")
        return frame_id, "ok", colored, N
    except Exception:
        return frame_id, "error", colored, N


def main():
    from config import (
        EXTRACTED_DIR, OUTPUT_DIR, STAGE1_DIR,
        TEST_START_FRAME, TEST_END_FRAME,
    )

    parser = argparse.ArgumentParser(description="点云投影赋色 (多进程)")
    parser.add_argument("--start", type=int, default=TEST_START_FRAME)
    parser.add_argument("--end", type=int, default=TEST_END_FRAME)
    parser.add_argument("--cams", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--workers", type=int, default=16,
                        help="并发进程数 (默认: 16)")
    parser.add_argument("--sky-mask-dir", type=str, default=None,
                        help="可选 sky mask 目录，mask 中 >= threshold 的像素跳过")
    parser.add_argument("--sky-threshold", type=int, default=128)
    args = parser.parse_args()

    print("=" * 60)
    print("步骤2: 点云投影赋色 (多进程)")
    print("=" * 60)
    print(f"帧范围: {args.start} ~ {args.end}")
    print(f"使用相机: {args.cams}")
    print(f"最大点数: {args.max_points}")
    print(f"并发进程: {args.workers}")
    print(f"点云目录: {EXTRACTED_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"Sky mask: {args.sky_mask_dir or 'disabled'}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载位姿数据
    print("\n[1/4] 加载位姿数据...")
    from config import load_img_pos
    pose_data = load_img_pos(args.start, args.end)
    print(f"  加载了 {len(pose_data)} 帧位姿")

    # 构建任务列表 (传递 numpy 数组的副本)
    print("\n[2/4] 构建任务列表...")
    tasks = []
    for pose in pose_data:
        frame_id = pose["frame_id"]
        output_path = os.path.join(OUTPUT_DIR, f"frame_{frame_id:04d}.ply")
        pcd_path = os.path.join(EXTRACTED_DIR, f"section_{frame_id:04d}.ply")
        if len(tasks) == 0:
            print(f"  样例点云: {pcd_path} (存在: {os.path.exists(pcd_path)})")
        if not os.path.exists(pcd_path):
            pcd_path = None

        T = pose["T_world_robot"]
        tasks.append({
            "frame_id": frame_id,
            "pcd_path": pcd_path,
            "output_path": output_path,
            "R_rw": T[:3, :3].copy(),
            "t_rw": T[:3, 3].copy(),
        })
    print(f"  共 {len(tasks)} 个任务")

    # 并行执行
    print(f"\n[3/4] 并行投影 (workers={args.workers})...")
    stats = {"ok": 0, "skip": 0, "no_pcd": 0, "empty_pcd": 0, "no_img": 0, "error": 0}
    t_start = time.time()
    last_report = 0

    # 使用 fork 上下文，子进程自动继承父进程内存 (Linux 高效 copy-on-write)
    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(args.cams, args.max_points, args.skip_existing,
                  args.sky_mask_dir, args.sky_threshold),
    ) as pool:
        results = pool.imap_unordered(_project_and_save, tasks, chunksize=4)
        for fid, status, colored, total in results:
            stats[status] = stats.get(status, 0) + 1
            done = sum(stats.values())
            now = time.time()
            if now - last_report >= 1.0:
                pct = 100 * done / len(tasks) if tasks else 0
                eta = (now - t_start) / done * (len(tasks) - done) if done > 0 else 0
                print(f"  进度: {done}/{len(tasks)} ({pct:.1f}%) | "
                      f"成功 {stats['ok']} | 错误 {stats['error']} | ETA: {eta:.0f}s")
                last_report = now

    elapsed = time.time() - t_start
    print(f"\n  总耗时: {elapsed:.1f}s")
    print(f"  吞吐量: {len(tasks) / elapsed:.1f} 帧/秒")
    print(f"  统计: 成功 {stats['ok']}, 跳过 {stats['skip']}, "
          f"缺文件 {stats['no_pcd']}, 空点云 {stats['empty_pcd']}, "
          f"无图像 {stats['no_img']}, 错误 {stats['error']}")

    # 保存元数据
    meta_path = os.path.join(OUTPUT_DIR, "projection_meta.txt")
    with open(meta_path, "w") as f:
        f.write(f"start_frame={args.start}\n")
        f.write(f"end_frame={args.end}\n")
        f.write(f"cams={args.cams}\n")
        f.write(f"workers={args.workers}\n")
        f.write(f"stats={stats}\n")
        f.write(f"elapsed_sec={elapsed:.1f}\n")
    print(f"\n[4/4] 元数据已保存: {meta_path}")
    print("\n✅ 步骤2完成!")


if __name__ == "__main__":
    main()
