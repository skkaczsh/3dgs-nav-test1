# Semantic 3D Reconstruction — TIMELINE

## 坐标系关系（关键发现）

### 坐标系定义

**PCD（激光扫描仪）**：
- Z 轴 = 上下（高度），范围 [-7.12, 52.77]m
- Y 轴 = 前后（距离），范围 [-0.004, 94.73]m
- X 轴 = 左右（垂直于前后），范围 [-0.001, 118.97]m

**Splat / COLMAP（3DGS 重建）**：
- X 轴 = 上下（高度）
- Z 轴 = 前后
- Y 轴 = 左右

### PCD vs Splat 轴向关系

| 轴向 | PCD（扫描仪） | Splat（COLMAP） |
|------|--------------|----------------|
| **上下** | Z 轴 | X 轴 |
| **前后** | Y 轴 | Z 轴 |
| **左右** | X 轴 | Y 轴 |
| **X 方向** | 正向（远离原点） | 反向（两个系统 X 方向相反）|

### PCD → Splat 坐标变换

```python
# PCD X 方向与 SPLAT X 相反，故取反
# PCD Z（上下）→ SPLAT Y（对应高度映射到上下轴）
# PCD Y（前后）→ SPLAT Z（前后一致）
pcd_local_pts = np.stack([-pts[:, 0], pts[:, 2], pts[:, 1]], axis=1) + T_offset
```

变换后 PCD 在 SPLAT 坐标系中：
- Splat X ∈ [-118.97, -0.001]（PCD 的 X 正向变为 SPLAT 的 X 负向）
- Splat Y ∈ [-7.12, 52.77]（PCD 的 Z 上下）
- Splat Z ∈ [-0.004, 94.73]（PCD 的 Y 前后）

### Camera 与 PCD 方向错位

COLMAP 相机朝向 SPLAT 正 X 方向，但 PCD 变换后在 SPLAT 负 X 区域。
相机全部朝向远离 PCD 的方向，导致无法将 mask 标签融合到 PCD voxels。

**修复方案**：对所有 COLMAP 相机应用 180° 绕 Y 轴旋转：
```python
# R_y(180) = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
CAMERA_ROTATION_Y_180 = np.array([
    [-1.0, 0.0, 0.0],
    [ 0.0, 1.0, 0.0],
    [ 0.0, 0.0, -1.0],
], dtype=np.float64)

def apply_camera_rotation(view):
    T_cw = view["T_cw"].copy()
    new_pos = CAMERA_ROTATION_Y_180 @ T_cw[:3, 3]
    new_R = CAMERA_ROTATION_Y_180 @ T_cw[:3, :3]
    view["T_cw"][:3, :3] = new_R
    view["T_cw"][:3, 3] = new_pos
    return view
```

旋转前相机 forward X = [+0.14, +0.70, +0.81]（远离 PCD）
旋转后相机 forward X = [-0.14, -0.70, -0.81]（朝向 PCD）

---

## 项目进展日志

### 2026-05-19

**Session Start**

- 项目：Semantic 3D Reconstruction — 基于 Gaussian Splatting 的语义分割 pipeline
- 目标：对天台场景进行 3D 语义重建

**Phase 1: 数据准备**

- PCD: `24,610,163` 点，来自激光扫描
- Splat: `3,078,419` Gaussians，来自 3DGS 重建

**Phase 2: SplatLoader 修复（关键 Bug）**

问题：SplatLoader 将位置除以 1000（假设文件存毫米），但文件实际存的是米。

验证：
```
Splat extent（原始）: X=[-99.99, 100.00] range=199.98m
→ 符合天台尺寸 ~120m → 文件存的就是米
```

修复：
```python
# 修复前（错误）
positions = np.stack([pos_x, pos_y, pos_z], axis=1).astype(np.float32) / 1000.0

# 修复后（正确）
positions = np.stack([pos_x, pos_y, pos_z], axis=1).astype(np.float32)
```

### 2026-05-20（凌晨）

**Phase 3: Near/Far Plane 修复**

问题：near=0, far=100 无法覆盖完整的 ±300m 场景。

诊断：Camera Z 范围 [-389, +408]m，大部分 Gaussians 在相机背后（z_cam < 0）。

修复：
```python
near_plane = -400.0
far_plane = 500.0
```

**Phase 4: 深度计算修复**

问题：深度用 alpha 加权平均，导致被远处的 Gaussians 污染。

修复：改用 max-alpha 深度（最前面的 Gaussian 贡献）：
```python
chunk_contrib = chunk_alpha > alpha_full
alpha_full = np.maximum(alpha_full, chunk_alpha)
rgb_full = np.where(chunk_contrib[:, :, None], chunk_rgb[..., :3], rgb_full)
depth_weighted = np.where(chunk_contrib, chunk_alpha * chunk_depth, depth_weighted)
```

**Phase 5: VLM JSON 解析修复**

问题：VLM API 返回截断的 JSON（`resp=''` 或带 markdown fences），导致所有标签变成 unknown。

修复：重写 `_parse_json_response` 支持：
1. Markdown code fences 移除
2. 截断 JSON 修复（补全 `]}` 结尾）
3. 非贪婪正则匹配

修复前：ANALYZE Attempt 1: JSON parse failed, resp=''
修复后：Attempt 1 成功，quality=0.85, 24/24 masks labeled ✅

**Phase 6: 分辨率不匹配修复**

问题：E2E 脚本将渲染分辨率改为 1600x1280，但 K 矩阵未缩放，导致融合时投影错误。

修复：移除 `v["width"] = 1600; v["height"] = 1280`，保持 800x640。

**E2E 测试结果**

| Test | Camera 0 | Camera 1 | Camera 2 | Labels | 备注 |
|------|----------|----------|----------|--------|------|
| test3 | 27 masks | 109 masks | 193 masks | mostly unknown | VLM 解析失败 |
| test4 | 24 masks | 105 masks | 192 masks | 321 total | 相机方向未修正 |
| test5 | 24 masks | 105 masks | 192 masks | 321 total | 相机方向未修正 |
| test6 | 待运行 | | | | **相机旋转修复** |

### 2026-05-20（下午）

**Phase 7: 坐标系关系确认与相机旋转修复**

用户确认：
- PCD 上下 = Z 轴，Splat 上下 = X 轴
- PCD 前后 = Y 轴，Splat 前后 = Z 轴
- PCD X 方向与 Splat X 方向相反

关键发现：COLMAP 相机朝向 SPLAT 正 X 方向（forward X = [+0.14, +0.70, +0.81]），
但 PCD 变换后在 SPLAT 负 X 区域（X=[-119, 0]）。相机朝向远离 PCD 的方向，
导致 fusion 时所有 voxel 投影到图像范围外，产生 100% unknown。

修复：对所有 COLMAP 相机应用 180° 绕 Y 轴旋转（R_y = [[-1,0,0],[0,1,0],[0,0,-1]]），
使相机朝向 SPLAT 负 X 方向（PCD 所在位置）。
