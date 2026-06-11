#!/usr/bin/env python3
"""步骤1: 视频抽帧 + 去畸变 (多线程版本)

根据 img_pos.txt 中的时间戳，从三路视频中提取对应的帧，
并使用 Kannala-Brandt 鱼眼模型进行去畸变校正。

使用 ThreadPoolExecutor 并行提取帧，充分利用多核。

使用方法:
    python extract_frames.py [--start 0] [--end 99] [--skip-existing] [--workers 32
    python extract_frames.py --start 0 --end 7631 --workers 64
"""

import os
import sys
import argparse
import subprocess
import time
import threading
import shutil
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    VIDEO_FILES, FRAME_OUTPUT_DIRS, IMG_POS_FILE, STAGE1_DIR,
    CAMERA_PARAMS, TEST_START_FRAME, TEST_END_FRAME, load_img_pos,
)


# ==================== 全局状态 ====================
_lock = threading.Lock()
_stats = {0: {"ok": 0, "skip": 0, "err": 0, "deltas": []},
          1: {"ok": 0, "skip": 0, "err": 0, "deltas": []},
          2: {"ok": 0, "skip": 0, "err": 0, "deltas": []}}
_progress_lock = threading.Lock()
_done_count = 0
_total_count = 0


def load_video_timestamps(video_path):
    """从 mkv 视频中提取所有帧的时间戳"""
    if shutil.which("ffprobe") is None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return [(i, i / fps) for i in range(count)]

    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "frame=pkt_pts_time", "-of", "csv=p=0", video_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    frames = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            pts = float(line.split(",")[0])
            frames.append((len(frames), pts))
        except (ValueError, IndexError):
            continue
    return frames


def find_best_video_frame(target_ts_rel, frame_idxs, rel_times, max_delta=0.15):
    """二分查找最近帧"""
    idx = np.searchsorted(rel_times, target_ts_rel)
    candidates = []
    if idx > 0:
        candidates.append((idx - 1, abs(rel_times[idx - 1] - target_ts_rel)))
    if idx < len(rel_times):
        candidates.append((idx, abs(rel_times[idx] - target_ts_rel)))
    best_rel_idx, best_delta = min(candidates, key=lambda x: x[1])
    if best_delta > max_delta:
        return None, float("inf")
    return int(frame_idxs[best_rel_idx]), best_delta


def undistort_fisheye_image(image, K, D):
    """鱼眼图像去畸变"""
    h, w = image.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
    )
    return cv2.remap(image, map1, map2, cv2.INTER_LINEAR)


def read_video_frame(video_path, frame_idx, rel_ts, output_path):
    """读取一帧视频；优先 ffmpeg，缺失时回退到 OpenCV。"""
    if shutil.which("ffmpeg") is not None:
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{rel_ts:.4f}",
            "-i", video_path,
            "-vframes", "1",
            "-q:v", "2",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    return cv2.imwrite(output_path, frame)


def extract_one_task(args_tuple):
    """执行一个 (frame_id, cam_id) 抽取任务"""
    global _done_count, _total_count
    frame_id, cam_id, video_path, video_timestamps_arr, K, D, output_path, skip_existing = args_tuple

    frame_idxs = video_timestamps_arr[:, 0]
    rel_times = video_timestamps_arr[:, 1]

    if skip_existing and os.path.exists(output_path):
        with _lock:
            _stats[cam_id]["skip"] += 1
        with _progress_lock:
            _done_count += 1
        return "skip"

    best_idx, delta = find_best_video_frame(frame_id * 0.1, frame_idxs, rel_times)
    if best_idx is None:
        with _lock:
            _stats[cam_id]["err"] += 1
        with _progress_lock:
            _done_count += 1
        return "err"

    best_rel_ts = float(video_timestamps_arr[best_idx][1])

    if not read_video_frame(video_path, best_idx, best_rel_ts, output_path):
        with _lock:
            _stats[cam_id]["err"] += 1
        with _progress_lock:
            _done_count += 1
        return "err"

    img = cv2.imread(output_path)
    if img is None:
        os.remove(output_path)
        with _lock:
            _stats[cam_id]["err"] += 1
        with _progress_lock:
            _done_count += 1
        return "err"

    undistorted = undistort_fisheye_image(img, K, D)
    cv2.imwrite(output_path, undistorted)

    with _lock:
        _stats[cam_id]["ok"] += 1
        _stats[cam_id]["deltas"].append(delta)
    with _progress_lock:
        _done_count += 1
    return "ok"


def print_progress(t_start):
    """打印进度（主线程调用）"""
    with _progress_lock:
        done = _done_count
    elapsed = time.time() - t_start
    pct = 100 * done / _total_count if _total_count else 0
    cam0_ok = _stats[0]["ok"]
    eta = elapsed / done * (_total_count - done) if done > 0 else 0
    print(f"  进度: {done}/{_total_count} ({pct:.1f}%) | "
          f"Cam0成功 {cam0_ok} | ETA: {eta:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="视频抽帧 + 去畸变 (多线程)")
    parser.add_argument("--start", type=int, default=TEST_START_FRAME)
    parser.add_argument("--end", type=int, default=TEST_END_FRAME)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--workers", type=int, default=32,
                        help="并发线程数 (默认: 32)")
    parser.add_argument("--min-success-ratio", type=float, default=0.95,
                        help="低于该成功率时退出非零，避免静默产生空缓存")
    args = parser.parse_args()

    print("=" * 60)
    print("步骤1: 视频抽帧 + 去畸变 (多线程)")
    print("=" * 60)
    print(f"帧范围: {args.start} ~ {args.end} (共 {args.end - args.start + 1} 帧)")
    print(f"并发线程数: {args.workers}")

    missing_videos = [path for path in VIDEO_FILES.values() if not os.path.exists(path)]
    if missing_videos:
        print("\n错误: 缺失视频文件:")
        for path in missing_videos:
            print(f"  - {path}")
        sys.exit(2)

    # 加载位姿数据
    print("\n[1/4] 加载位姿数据...")
    pose_data = load_img_pos(args.start, args.end)
    print(f"  加载了 {len(pose_data)} 帧位姿")
    if not pose_data:
        sys.exit("错误: 未找到位姿数据！")

    video_start_unix = pose_data[0]["timestamp"]

    # 加载视频时间戳
    print("\n[2/4] 加载视频帧时间戳索引...")
    video_ts = {}
    for cam_id, video_path in VIDEO_FILES.items():
        ts_file = os.path.join(FRAME_OUTPUT_DIRS[cam_id], "..",
                               f"video_timestamps_cam{cam_id}.npy")
        os.makedirs(os.path.dirname(ts_file), exist_ok=True)
        if os.path.exists(ts_file):
            video_ts[cam_id] = np.load(ts_file, allow_pickle=True)
            print(f"  Cam{cam_id}: {len(video_ts[cam_id])} 帧 (缓存)")
        else:
            video_ts[cam_id] = np.array(load_video_timestamps(video_path))
            np.save(ts_file, video_ts[cam_id])
            print(f"  Cam{cam_id}: {len(video_ts[cam_id])} 帧")

    # 构建任务列表
    print("\n[3/4] 并行提取帧...")
    tasks = []
    for pose in pose_data:
        frame_id = pose["frame_id"]
        for cam_id in [0, 1, 2]:
            os.makedirs(FRAME_OUTPUT_DIRS[cam_id], exist_ok=True)
            output_path = os.path.join(FRAME_OUTPUT_DIRS[cam_id],
                                       f"frame_{frame_id:04d}.png")
            tasks.append((
                frame_id, cam_id,
                VIDEO_FILES[cam_id],
                video_ts[cam_id],
                CAMERA_PARAMS[cam_id]["K"],
                CAMERA_PARAMS[cam_id]["D"],
                output_path,
                args.skip_existing,
            ))

    global _done_count, _total_count
    _done_count = 0
    _total_count = len(tasks)

    t_start = time.time()
    last_print = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_one_task, t): t for t in tasks}
        for future in as_completed(futures):
            # 每 0.5s 打印一次进度
            now = time.time()
            if now - last_print >= 0.5:
                print_progress(t_start)
                last_print = now

    elapsed = time.time() - t_start
    print(f"\n  总耗时: {elapsed:.1f}s")
    print(f"  吞吐量: {_total_count / elapsed:.1f} 帧/秒")

    print("\n  Cam0: 成功 {0[ok]}, 跳过 {0[skip]}, 错误 {0[err]}".format(_stats[0]))
    print("  Cam1: 成功 {0[ok]}, 跳过 {0[skip]}, 错误 {0[err]}".format(_stats[1]))
    print("  Cam2: 成功 {0[ok]}, 跳过 {0[skip]}, 错误 {0[err]}".format(_stats[2]))
    ok_total = sum(_stats[cam_id]["ok"] + _stats[cam_id]["skip"] for cam_id in [0, 1, 2])
    success_ratio = ok_total / max(_total_count, 1)

    for cam_id in [0, 1, 2]:
        deltas = _stats[cam_id]["deltas"]
        if deltas:
            d = np.array(deltas)
            print(f"  Cam{cam_id} 时间戳偏移: mean={d.mean()*1000:.1f}ms, max={d.max()*1000:.1f}ms")

    # 保存元数据
    meta_path = os.path.join(STAGE1_DIR, "frame_extraction_meta.txt")
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    compact_stats = {
        cam_id: {key: _stats[cam_id][key] for key in ("ok", "skip", "err")}
        for cam_id in [0, 1, 2]
    }
    with open(meta_path, "w") as f:
        f.write(f"start_frame={args.start}\n")
        f.write(f"end_frame={args.end}\n")
        f.write(f"workers={args.workers}\n")
        f.write(f"video_start_unix={video_start_unix}\n")
        f.write(f"extracted={{0: {len(pose_data)}, 1: {len(pose_data)}, 2: {len(pose_data)}}}\n")
        f.write(f"stats={compact_stats}\n")
        f.write(f"success_ratio={success_ratio:.6f}\n")
        f.write(f"elapsed_sec={elapsed:.1f}\n")
    print(f"\n[4/4] 元数据已保存: {meta_path}")
    if success_ratio < args.min_success_ratio:
        print(f"\n错误: 抽帧成功率 {success_ratio:.3f} < {args.min_success_ratio:.3f}")
        sys.exit(3)
    print("\n✅ 步骤1完成!")


if __name__ == "__main__":
    main()
