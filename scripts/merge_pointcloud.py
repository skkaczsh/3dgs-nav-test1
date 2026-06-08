#!/usr/bin/env python3
"""步骤3: 点云合成

将步骤2输出的所有带色点云帧合并为单个点云文件。

使用方法:
    python merge_pointcloud.py [--start 0] [--end 99] [--output merged.ply]
        [--voxel-size 0.02] [--no-cpp]
    python merge_pointcloud.py --start 0 --end 7631 --voxel-size 0.05
"""

import os
import sys
import argparse
import time
import tempfile
import numpy as np
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_DIR, STAGE1_DIR, TEST_START_FRAME, TEST_END_FRAME

# C++ 降采样程序路径
VOXEL_DOWNSAMPLE_BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voxel_downsample")


def load_colored_ply(ply_path):
    """加载带颜色的 PLY 点云"""
    points, colors = [], []
    reading = False
    with open(ply_path, "rb") as f:
        for line in f:
            s = line.decode("utf-8", errors="replace")
            if s.strip() == "end_header":
                reading = True
                continue
            if reading:
                p = s.strip().split()
                if len(p) >= 6:
                    try:
                        points.append([float(p[0]), float(p[1]), float(p[2])])
                        colors.append([float(p[3]), float(p[4]), float(p[5])])
                    except ValueError:
                        continue
    pts = np.array(points, dtype=np.float64)
    cols = np.array(colors, dtype=np.float64)
    if cols.max() > 1.0:
        cols = cols / 255.0
    return pts, cols


def save_colored_ply(ply_path, points, colors):
    """保存带颜色的 ASCII PLY，colors 使用 0-1 浮点或 0-255 数值均可。"""
    cols = np.asarray(colors)
    if cols.size and cols.max() <= 1.0:
        cols = np.clip(cols * 255.0, 0, 255)
    cols = cols.astype(np.uint8)
    with open(ply_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, cols):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def voxel_downsample_cpp(input_ply, output_ply, voxel_size):
    """使用 C++ 程序进行体素降采样（更快）"""
    try:
        result = subprocess.run(
            [VOXEL_DOWNSAMPLE_BIN, input_ply, output_ply, str(voxel_size)],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode == 0:
            return True, result.stdout + result.stderr
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)


def voxel_downsample_python(points, colors, voxel_size=0.02):
    """体素降采样（Python 实现，作为回退）"""
    if voxel_size <= 0 or len(points) == 0:
        return points, colors

    idx = (points / voxel_size).astype(np.int64)
    packed = (idx[:, 0].astype(np.int64) * 1000000000000 +
              idx[:, 1].astype(np.int64) * 1000000 +
              idx[:, 2].astype(np.int64))
    unique_keys, inv_idx = np.unique(packed, return_inverse=True)

    sampled_pts, sampled_cols = [], []
    for k in range(len(unique_keys)):
        mask = inv_idx == k
        center = (idx[mask][0] + 0.5) * voxel_size
        sampled_pts.append(center)
        sampled_cols.append(colors[mask].mean(axis=0))

    return np.array(sampled_pts), np.array(sampled_cols)


def main():
    parser = argparse.ArgumentParser(description="点云合成")
    parser.add_argument("--start", type=int, default=TEST_START_FRAME)
    parser.add_argument("--end", type=int, default=TEST_END_FRAME)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.02)
    parser.add_argument("--use-cpp", action="store_true", default=True,
                        help="使用 C++ 版本进行降采样（默认开启，快10-100倍）")
    parser.add_argument("--no-cpp", action="store_true",
                        help="禁用 C++ 版本，使用纯 Python 降采样")
    args = parser.parse_args()

    if args.no_cpp:
        args.use_cpp = False

    if args.output is None:
        args.output = os.path.join(STAGE1_DIR, "merged.ply")

    use_cpp = args.use_cpp and os.path.exists(VOXEL_DOWNSAMPLE_BIN)

    print("=" * 60)
    print("步骤3: 点云合成")
    print("=" * 60)
    print(f"帧范围: {args.start} ~ {args.end}")
    print(f"输出文件: {args.output}")
    print(f"体素大小: {args.voxel_size}m")
    print(f"使用 C++ 降采样: {'是' if use_cpp else '否'}")

    t_start = time.time()

    # 扫描文件
    print("\n[1/4] 扫描文件...")
    all_files = sorted([f for f in os.listdir(OUTPUT_DIR)
                       if f.startswith("frame_") and f.endswith(".ply")])
    print(f"  找到 {len(all_files)} 个 PLY 文件")

    # 逐个加载
    print("\n[2/4] 加载点云...")
    all_points, all_colors = [], []
    loaded, failed = 0, 0
    for i, fname in enumerate(all_files):
        frame_id = int(fname.replace("frame_", "").replace(".ply", ""))
        if frame_id < args.start or frame_id > args.end:
            continue
        ply_path = os.path.join(OUTPUT_DIR, fname)
        try:
            pts, cols = load_colored_ply(ply_path)
            if pts is not None and len(pts) > 0:
                valid = cols.sum(axis=1) > 0
                all_points.append(pts[valid])
                all_colors.append(cols[valid])
                loaded += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1

        if (i + 1) % 20 == 0:
            print(f"  进度: {i+1}/{len(all_files)}")

    if not all_points:
        print("  错误: 没有找到有效的点云文件!")
        sys.exit(1)

    print(f"  加载完成: {loaded} 帧成功, {failed} 失败")

    # 合并
    print("\n[3/4] 合并点云...")
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    total_pts = len(all_points)
    print(f"  合并后: {total_pts} 点")

    # 体素降采样
    if args.voxel_size > 0:
        print(f"\n[4/4] 体素降采样 (voxel_size={args.voxel_size}m)...")
        t_vox = time.time()

        if use_cpp:
            # 使用 C++ 版本（高效）
            print("  使用 C++ 版本降采样...")
            with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_in:
                tmp_input = tmp_in.name
            with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp_out:
                tmp_output = tmp_out.name

            try:
                # 保存合并后的点云到临时文件
                save_colored_ply(tmp_input, all_points, all_colors)

                # 调用 C++ 程序
                success, msg = voxel_downsample_cpp(tmp_input, tmp_output, args.voxel_size)
                if success:
                    # 加载降采样结果
                    all_points, all_colors = load_colored_ply(tmp_output)
                    print(f"  {total_pts} -> {len(all_points)} 点 ({100*len(all_points)/total_pts:.1f}%)")
                else:
                    print(f"  C++ 版本失败，回退到 Python: {msg}")
                    all_points, all_colors = voxel_downsample_python(all_points, all_colors, args.voxel_size)
            finally:
                # 清理临时文件
                for f in [tmp_input, tmp_output]:
                    if os.path.exists(f):
                        os.remove(f)
        else:
            # 使用 Python 版本（慢）
            print("  使用 Python 版本降采样...")
            all_points, all_colors = voxel_downsample_python(all_points, all_colors, args.voxel_size)
            print(f"  {total_pts} -> {len(all_points)} 点 ({100*len(all_points)/total_pts:.1f}%)")

        print(f"  体素化耗时: {time.time()-t_vox:.1f}s")

    # 保存
    print(f"\n  保存到: {args.output}")
    save_colored_ply(args.output, all_points, all_colors)

    elapsed = time.time() - t_start
    print(f"\n  === 合成结果统计 ===")
    print(f"  总帧数: {loaded}")
    print(f"  总点数: {len(all_points)}")
    print(f"  包围盒 X: [{all_points[:,0].min():.2f}, {all_points[:,0].max():.2f}]")
    print(f"  包围盒 Y: [{all_points[:,1].min():.2f}, {all_points[:,1].max():.2f}]")
    print(f"  包围盒 Z: [{all_points[:,2].min():.2f}, {all_points[:,2].max():.2f}]")
    print(f"  颜色均值: R={all_colors[:,0].mean():.2f}, "
          f"G={all_colors[:,1].mean():.2f}, B={all_colors[:,2].mean():.2f}")
    print(f"  文件大小: {os.path.getsize(args.output)/1024/1024:.1f} MB")
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"\n✅ 步骤3完成! 输出: {args.output}")


if __name__ == "__main__":
    main()
