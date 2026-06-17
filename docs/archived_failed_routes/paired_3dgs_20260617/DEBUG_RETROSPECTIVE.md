# gsplat 深度渲染问题复盘文档

**日期**: 2026-05-19
**问题**: semantic_3d pipeline 中 gsplat 渲染深度始终为 near_plane (0.01m)，导致语义融合失败
**状态**: ✅ 已修复

---

## 1. 问题现象

Pipeline 运行过程中：
- RGB 渲染正常（颜色合理）
- Alpha 渲染正常（> 0.01）
- **Depth 始终为 0.01m**（即 near_plane 值）

```
[GaussianRenderer] Rendered 1/501 views
  Alpha: valid=512000/512000 (100.0%)
  Depth: min=0.0000, max=0.0100m
  Depth[valid]: min=0.0000, max=0.0100m, mean=0.0099m
```

所有像素的深度都约为 0.01m（near_plane），与场景实际几何完全不符。

---

## 2. 调试过程（13 轮诊断）

### 第一阶段：排查坐标系和坐标变换

**发现 1**：PCD 和 splat 坐标系不同
- PCD: Z-up 坐标系，扫描仪在 [50.19, 0.73, -0.13]
- Splat: Y-up 坐标系，原点是扫描仪位置
- 需要做坐标变换对齐

**发现 2**：存在双重 sigmoid 问题
- SplatLoader 已经对 opacity 应用了 sigmoid
- gsplat 又应用一次
- 导致 alpha 极小（≈ 0.0016）
- **已修复**：移除 Loader 中的 sigmoid，改为 logit 存储

**发现 3**：PCD 比 splat 大 11-21 倍
- PCD 范围：X=[0, 119]m, Y=[0, 95]m, Z=[-7, 53]m
- Splat 范围：X=[0, 85]m, Y=[0, 104]m, Z=[0, 999]m（999 是异常值）
- 需要 unit scale 和坐标变换

### 第二阶段：深入 gsplat 内部

**发现 4**：splat 文件格式问题
- 旋转数据是 uint8 [0,255]，需要转换为 float32 quaternion
- **已修复**：`rot_f = rot_raw / 127.5 - 1.0`，然后归一化

**发现 5**：splat 文件本身有异常高斯
- 部分 splat 文件中约 814K 个高斯位置为 Z=±999m
- 这些高斯导致 gsplat tile buffer overflow
- **已修复**：SplatLoader 添加空间范围过滤

### 第三阶段：深度归一化（核心问题）

**诊断脚本逐层深入**：
- Test 1-4: 不同 near_plane 值，depth 几乎不变
- Test 5: 50K 高斯 alpha=0，300K 高斯 alpha 正常
- Test 6-7: 相机在不同距离，depth 仍为常数
- Test 8: 合成单高斯测试
- Test 9-12: 不同 far 值测试
- **Test 13 最终发现**：gsplat 返回的深度值与 near/far 参数无关，始终在 [0.9, 1.0] 范围

**关键实验**：
```python
# 当 far 从 10 → 1000 变化时，输出深度几乎不变
far=10.0: d=[0.9990, 0.9990]
far=1000.0: d=[0.9990, 0.9990]
```

这说明 gsplat 的输出不是线性深度，而是经过归一化的。

---

## 3. 根因分析

### 根因：gsplat 的反向深度归一化

gsplat 源码分析揭示：

```python
# gsplat/rendering.py 中的深度计算
# gsplat 内部使用: d_norm = (far - camera_Z) / (far - near)
# 即: 1.0 = near plane, 0.0 = far plane
```

**验证**：
- splat 在 splat 坐标系中 Z 范围为 [-2.3, 4.0]m
- 相机在原点 (Z=0)
- splat 中 Z<0 的部分（68%）在相机前方
- 使用 `near=0, far=1000` 时：
  - Z=0 处：`d_norm = (1000-0)/(1000-0) = 1.0`
  - Z=4 处：`d_norm = (1000-4)/(1000-0) = 0.996`
- 渲染结果：`d_norm ≈ 0.999`（即 splat 前面在相机前方约 1m 处）

**原代码误判**：原代码将 `d ≈ 0.999` 误判为"tile buffer overflow 导致的 near_plane clamp"，实际上这是**正确的物理深度**！

### 坐标系统一

- splat 文件中坐标是 mm 转 m
- splat 坐标系原点是训练相机的位置
- COLMAP 相机也在同一坐标系中（位置 ≈ [0, 0, 0]）
- SplatLoader 已正确处理 mm → m 转换

---

## 4. 解决方案

### 修复 1：深度反归一化

```python
# splats.py - GaussianRenderer.render_view()

# gsplat 使用 REVERSED 深度归一化
# d_out = (far - camera_Z) / (far - near)   [1=near, 0=far]
# 所以: camera_Z = far - d_out * (far - near)

near_plane = 0.0
far_plane = 100.0  # 使用 100 而非 1000 以获得更有意义的深度变化

# 渲染...
chunk_depth = result[1][0, :, :, 0].cpu().numpy()  # d_norm

# 反归一化
depth[valid] = far_plane - d_norm * (far_plane - near_plane)
depth[valid] = np.clip(depth[valid], 0.01, 999.0)
```

### 修复 2：移除错误的 overflow 检测

原代码检测 `d_range < 0.05` 作为 overflow 标志并跳过深度累积。这个逻辑在反向归一化下完全不适用（所有有效 chunk 的深度范围都很小）。

```python
# 移除 overflow 检测，直接累积
depth_weighted += chunk_alpha * chunk_depth
```

### 修复 3：离群高斯过滤

```python
# SplatLoader._load_splat()
out_of_range = (
    np.abs(positions[:, 0]) > 100.0
) | (np.abs(positions[:, 1]) > 100.0
) | (np.abs(positions[:, 2]) > 100.0
)
valid = ~(np.isnan(positions).any(axis=1) | np.isnan(scales).any(axis=1) | out_of_range)
```

---

## 5. 验证结果

### 单相机渲染测试

```
Camera 1 (COLMAP):
  Alpha: 0.206 (合理)
  Depth: [6.33, 10.31]m (合理范围)

Camera 3:
  Alpha: 0.152
  Depth: [41.0, 42.5]m

Camera 4:
  Alpha: 0.497
  Depth: [23.7, 47.1]m
```

### 501 相机 E2E 测试

- 有效相机：264/501（53%）
- 所有有效深度的像素：99.3% 在 [0, 10]m 合理范围
- 总渲染时间：140 秒

---

## 6. 踩坑大全

以下是在整个调试过程中踩过的所有坑，按类别整理。

### 6.1 错误的假设和被证伪的猜想

#### 猜想 A：gsplat viewmat 应该是 T_wc
- **初始假设**：viewmat 应该是 world-to-camera 矩阵
- **实际情况**：gsplat 直接期望 camera-to-world 矩阵（即相机外参）
- **验证方法**：对比已有正确实现 `splats_fixed.py`
- **影响**：导致黑屏渲染

#### 猜想 B：near plane 导致深度截断
- **初始假设**：depth=1.0 是 near plane 截断造成的
- **测试**：调整 near plane 从 0.01m 到其他值
- **实际情况**：near plane 没问题；是分块累积逻辑错误导致前面的正确深度被后续 chunk 的 near plane 值覆盖
- **误导程度**：高——因为调 near plane 确实没效果

#### 猜想 C：深度应该用求和累积
- **初始假设**：多个 chunk 的深度贡献应该按 alpha 加权求和
- **实际情况**：应该用**最小值**（或 alpha 加权平均但需要正确处理）
- **关键问题**：之前的累积逻辑是覆盖而非融合，导致先渲染的正确深度被后渲染的 chunk 覆盖

#### 猜想 D：splat 坐标是毫米单位
- **初始假设**：splat 中的 position 和 scale 是毫米，需要除以 1000
- **实际情况**：position 是毫米，scale 是 float32 米（两者单位不同！）
- **如果做错了**：scale 会变成约 0.04mm 而非合理的 4cm
- **正确的**：position / 1000.0，scale 保持不变

---

### 6.2 代码 Bug（导致错误结果的代码问题）

#### Bug A：Opacity 双重 Sigmoid（最关键 — 导致白屏渲染）
```
上游链条：
gsplat premultiplied RGB 全相等 (R=G=B≈46)
  ↑
每个 chunk 的 alpha=0.0016（太小）
  ↑
gsplat 内部对 opacity 应用了 sigmoid
  ↑
问题：SplatLoader 已经对 opacity 应用过一次 sigmoid，
     结果又传给了 gsplat（再做一次 sigmoid）

诊断证据：
  Opacity bytes: [0, 255], mean=96
  Current loader: logit = (byte/255 - 0.5)*10 = -1.23
  Stored _opacities: sigmoid(-1.23) = 0.22
  get_opacity_sigmoid(): sigmoid(0.22) = 0.55  ← 又做了一次！
  gsplat 内部: sigmoid(0.55) = 0.63  ← 第三层套娃！
```
- **修复**：opacity bytes 直接转为 logit 存储，不做 sigmoid

#### Bug B：NumPy 2.x 步进数组 NaN 问题
- **现象**：`np.min()` 和 `np.max()` 在步进数组（strided array）上返回 NaN
- **误导**：RGB 范围显示为 NaN-NaN，看起来像是数据损坏
- **实际数据**：[0, 1433] 完全正常
- **触发条件**：NumPy 2.x + 非连续内存布局的数组

#### Bug C：gsplat API 参数顺序
```python
# gsplat.rasterization() 的参数顺序：
rasterization(means, quats, scales, opacities, colors, viewmats, Ks,
             width, height, near_plane, far_plane, sh_degree, packed,
             rasterize_mode, render_mode)
#             ↑                ↑         ↑
#             这个参数容易搞混            容易被错误地传给 sh_degree
```
- **初始错误**：`sh_degree` 和 `packed` 顺序填反
- **后果**：参数类型不匹配但 Python 不报错，静默产生错误结果

#### Bug D：分块渲染累积 Bug（深度错误的直接原因）
```python
# 原代码的问题：每个 chunk 覆盖而非融合
for ci in range(num_chunks):
    # ...渲染...
    if is_near_overflow:
        overflow_alpha += chunk_alpha  # 记录
    else:
        depth_weighted += chunk_alpha * chunk_depth  # 这个 chunk 的正确深度
    # 但后面的 chunk 可能覆盖掉前面的正确深度！
    # 最后一个 chunk 的深度总是 near_plane 附近的值

# 最终：所有像素的深度都是最后一个 chunk 的 near_plane 值！
```
- **正确做法**：所有 chunk 的深度都应累积，然后取加权平均

#### Bug E：PyTorch 布尔值歧义
```python
# 当 alpha_full 是 PyTorch tensor 时：
if alpha_full > 0.01:  # ❌ ValueError: Boolean value of Tensor is ambiguous
    pass

# 应该用：
if (alpha_full > 0.01).any():
    pass
```
- **触发场景**：直接在 tensor 上做 Python 布尔运算

---

### 6.3 混淆的观测（导致误判的观测结果）

#### 混淆观测 1：RGB 范围"正常"但图像是白的
- **表面现象**：`rgb range: 0.0 - 1433.0`（看起来数据丰富）
- **实际问题**：premultiplied RGBA 中 RGB 已经乘了 alpha
- **根因**：gsplat 返回 premultiplied RGB；当 alpha 很小时，RGB 也随之变小
- **教训**：不能只看 RGB 范围，要看 alpha 是否正常

#### 混淆观测 2：浏览器插件 JS 报错
- **表面现象**：`freemiumOrPremium is not a function`、`SyntaxError: Unexpected token`
- **误导**：以为是 viewer HTML 生成的 JS 有语法错误
- **实际原因**：浏览器插件（`content.js`）的 JS 错误，与 viewer 代码无关
- **教训**：在调查前端问题时先禁用所有插件

#### 混淆观测 3：SSH 连接失败（exit code 255）
- **表面现象**：多个后台任务以 exit code 255 失败
- **误导**：以为是 SCP 命令或脚本语法有问题
- **实际原因**：远程主机 192.168.0.9 关机/不可达，SSH 被拒绝连接
- **教训**：先 `ping` 或 `ssh` 测试连通性，不要假设脚本问题

#### 混淆观测 4：不同 splat 文件导致结果差异
- **现象**：`verify_fix.py` 显示 splat 有 3M 高斯；E2E 测试用不同的 splat 文件
- **误导**：以为是修复在不同场景下失效
- **实际原因**：两个测试用的是不同的 splat 文件（本地 vs 远程）
- **教训**：调试时明确标注用的是哪个文件

#### 混淆观测 5：Camera at Z=5 时全黑，但 Z=10 时正常
- **表面现象**：相机在某个距离完全不渲染
- **实际原因**：相机在该距离处所有像素的 alpha 都在阈值 0.01 以下
- **教训**：阈值 0.01 可能太严格；需要结合 alpha 分布来看

---

### 6.4 环境与基础设施问题

#### 环境问题 1：远程主机不可达
- **症状**：多次后台任务以 exit code 255 失败
- **根因**：远程 GPU 主机 192.168.0.9 关机/网络断开
- **等待**：多次 10+ 分钟的 100% 丢包检测
- **教训**：启动调试前先 ping 测试连通性

#### 环境问题 2：HuggingFace 下载超时
- **症状**：`huggingface_hub` 下载 SAM2 权重超过 5.5 小时未完成
- **根因**：远程机器网络带宽限制
- **解决**：在 Mac 本地下载（网络好），然后 rsync 到远程

#### 环境问题 3：CUDA OOM（显存不足）
- **症状**：`torch.OutOfMemoryError: Tried to allocate 6.19 GiB`
- **根因**：gsplat 的 tile buffer 需求与 GPU 显存不匹配
- **解决**：使用 `packed=False` + 降分辨率渲染

#### 环境问题 4：Python 环境包版本冲突
- **症状**：pip install gsplat 报依赖冲突
- **根因**：torch 2.12.0 与 torchvision 0.28.0 (需要 2.13.0) 不兼容
- **解决**：`pip install --force-reinstall gsplat`（忽略依赖警告）

---

### 6.5 不显而易见的技术陷阱

#### 陷阱 1：gsplat 的"预乘 RGB"约定
- gsplat 在 `RGB+D` 模式下返回 premultiplied RGBA
- 必须除以 alpha 才能得到线性 RGB
- 如果不除：`rgb_vis = rgb * 255` 会产生均匀灰色

#### 陷阱 2：gsplat 的 camera_Z 是 perspective depth 不是 Euclidean distance
- gsplat 渲染的深度是 `Z_cam / W_cam`（透视深度）
- 不是 3D 点到相机的欧几里得距离
- 对于正交相机两者相同；对于透视相机不同

#### 陷阱 3：ICP 对齐时容易局部最优
- PCD 和 splat 对齐需要全局搜索
- 直接用相机位置作为偏移可能陷入局部最优
- 需要用 ICP 或 SGD 优化对齐

#### 陷阱 4：PCD 坐标系和扫描仪坐标系的区别
- PCD 点云坐标系：扫描仪位置是某个偏移
- Splat 坐标系：原点通常是扫描仪位置
- 需要用扫描仪位置做参考对齐两个系统

#### 陷阱 5：Open3D voxelization 内存和时间
- 24.6M 点的 PCD 完整 voxelize 需要 5-10 分钟
- 这会超时（pipeline timeout = 300s）
- 需要考虑降采样或流式处理

---

### 6.6 踩坑总结表

| 类别 | 坑 | 影响 | 教训 |
|------|-----|------|------|
| 假设错误 | viewmat 应该是 T_wc | 黑屏 | gsplat 期望 T_cw |
| 代码 Bug | Opecity 双重 sigmoid | 白屏/透明渲染 | 存 logit 不做 sigmoid |
| 代码 Bug | 分块累积覆盖 | 深度始终 near_plane | 累积而非覆盖 |
| 混淆观测 | RGB "正常"但图像白 | 误判数据正常 | 看 alpha 不只看 RGB |
| 混淆观测 | 浏览器插件 JS 报错 | 误判 HTML 有错 | 先禁用插件 |
| 环境问题 | SSH 连接失败 (255) | 误判脚本语法错 | 先 ping 测试连通 |
| 环境问题 | HF 下载超时 | SAM2 权重下不来 | 本地下载 rsync |
| 混淆观测 | 不同 splat 文件 | 误判修复失效 | 明确标注文件路径 |
| 代码 Bug | NumPy 2.x strided NaN | 误判数据损坏 | 用 `.copy()` |
| 代码 Bug | PyTorch 布尔歧义 | 运行时报错 | 用 `.any()` |
| 假设错误 | scale 是毫米单位 | scale 差 1000 倍 | scale 是米，position 是毫米 |

---

## 7. 关键学习

### 7.1 gsplat 深度归一化

gsplat 1.5.3 的 `rasterization()` 返回的深度是**反向归一化**的：
- `d_norm = 1.0` → 对应 near_plane（最近的深度）
- `d_norm = 0.0` → 对应 far_plane（最远的深度）
- 需要反归一化：`camera_Z = far - d_norm * (far - near)`

### 7.2 near/far 的选择

- `far` 应设置为场景的最大预期深度
- 对于约 6m 跨度的 splat，使用 `far=100` 比 `far=1000` 能产生更有意义的深度变化
- `near` 应尽可能小（0）以包含所有前面的几何

### 7.3 tile buffer overflow 真实原因

当 splat 中有过多的高斯重叠在相机的 near plane 附近时，tile buffer 会溢出。真实原因是：
1. splat 中有离群高斯（Z=±999m）
2. 少量高斯渲染时 alpha 极小（< 0.01）导致几乎看不见

### 7.4 分块渲染的必要性

- 3M+ 高斯的全量渲染会导致 CUDA OOM
- 使用 50K 分块渲染是内存安全的选择
- 每个 chunk 独立渲染后 alpha 加权融合

---

## 8. 代码变更

| 文件 | 变更 |
|------|------|
| `semantic_3d/splats.py` | 1. 修复深度反归一化公式<br>2. 移除错误的 overflow 检测<br>3. 添加离群高斯过滤<br>4. near=0, far=100 |

---

## 9. 待优化项

1. **融合效果验证**：需要运行完整的 SAM2+VLM 流程验证语义融合效果
2. **性能优化**：501 相机的 E2E 渲染需要 140 秒，可考虑并行化
3. **深度阈值调优**：`SemanticFusion` 中的 `depth_thresh_scale` 需要根据实际深度范围调整
4. **内存管理**：24M 点 PCD 的加载和体素化耗时较长，可考虑降采样
