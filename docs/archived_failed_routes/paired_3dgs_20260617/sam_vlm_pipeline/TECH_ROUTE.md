# SAM2-VLM 单图像全景 Mask 生成技术路线说明

> **版本：** v1.0
> **日期：** 2026-05-16
> **适用场景：** 3DGS / NeRF 场景中，对单张图像进行语义级遮蔽（天空/边缘/远景）

---

## 1. 任务定义与目标

### 1.1 问题背景

在基于 3D Gaussian Splatting (3DGS) 的场景重建中，未定义区域（天空、黑色边缘、远场几何）会导致以下问题：

- **透明高斯膨胀**：无纹理信号的区域被大、低透明度的高斯球覆盖
- **背景污染**：高斯球延伸到图像边界外
- **训练不稳定**：背景区域的随机颜色干扰几何收敛

### 1.2 解决思路

对每张图像生成**逐像素的二值语义 mask**，遮蔽以下区域：
- 天空 / 云层
- 黑色边缘 / 错位区域
- 远景几何（深度 > 阈值）

生成的 mask 可直接用于 3DGS 训练时遮蔽输入图像，或作为几何验证的先验。

### 1.3 目标输出

```
输入: 一张 RGB 图像 (H × W × 3)
                    ↓
输出: 二值 mask 图像 (H × W, 0/255)
       255 = 遮蔽区域（天空/边缘/远景）
       0   = 保留区域（有效场景）
```

---

## 2. 技术路线

### 2.1 总体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                    SAM2-VLM Mask Pipeline                        │
│                                                                  │
│  输入图像 ──► Phase 1 (SAM2) ──► Phase 2 (VLM) ──► 最终 Mask   │
│     RGB       GPU 并发分割      异步并发标注     二值输出         │
│                                                                  │
│  单图处理时间: ~1-2s (SAM2) + ~3-5s (VLM 串行)                  │
│              ≈ 3-7s / 图 (16 workers 异步并发: ~0.4s / 图)        │
└──────────────────────────────────────────────────────────────────┘
```

**两阶段分离的核心优势：**
- Phase 1 使用 GPU，Phase 2 使用 CPU VLM，互不争抢
- 两阶段可交错并行：Phase 1 完成一张即可立即送入 Phase 2
- 支持断点续传：每张图像独立处理，中间结果持久化

### 2.2 Phase 1: SAM2 语义分割

**方法：** Segment Anything Model 2 (SAM2 Hiera Large)

**核心思想：** 使用 SAM2 的全自动分割模式，在图像上均匀采样点阵（32×32），对每个点预测分割区域，再合并去重叠。

**参数配置：**

| 参数 | 值 | 说明 |
|------|----|------|
| `points_per_side` | 32 | 每边采样点数 |
| `points_per_batch` | 64 | 每批点数 |
| `pred_iou_thresh` | 0.7 | 预测 IoU 阈值 |
| `stability_score_thresh` | 0.92 | 稳定性分数阈值 |
| `crop_n_layers` | 1 | 裁剪层数（更高分辨率） |
| `min_mask_region_area` | 500 px | 最小 mask 面积 |

**处理流程：**

```
输入图像 (1600×1296)
        │
        ▼
SAM2 点阵推理 (32×32=1024 点)
        │
        ├── 点 1 → predicted IoU=0.95, stability=0.98 → 保留
        ├── 点 2 → predicted IoU=0.72, stability=0.88 → 保留
        └── 点 3 → 重叠 → 丢弃
        │
        ▼
去重叠处理 (每像素只保留得分最高的 mask)
  score = predicted_iou × stability_score
        │
        ▼
输出: N 个独立 mask (N ≈ 20-50, 按面积降序)
```

**去重叠算法：** 构建每像素的 score map，贪心分配给得分最高的 mask，保证每个像素仅属于一个区域。

**输出文件：**

| 文件 | 内容 |
|------|------|
| `{name}_sam_masks.json` | mask 数据（segmentation 数组、bbox、area、scores） |
| `{name}_numbered.png` | 带编号的彩色 mask（供 VLM 可视化） |
| `{name}_sam_masks.png` | 所有 mask 叠加可视化 |
| `{name}_sam_done.flag` | 完成标志（JSON） |

### 2.3 Phase 2: VLM 语义标注

**方法：** 大视觉语言模型（qwen:32b via Ollama）

**核心思想：** 将带编号的 mask 图像发送给 VLM，让模型为每个区域标注语义类别。

**并发优化（关键）：**

| 优化前 | 优化后 |
|--------|--------|
| 同步 `requests`，4 workers | 异步 `httpx`，16 workers |
| ~0.25 图/秒 | ~4 图/秒 |
| 2292 张需 ~2.5 小时 | 2292 张需 ~10 分钟 |

**Prompt 设计：**

```
分析这张图，图中有{N}个分割区域。
请给每个区域一个中文标签：天空、地面、墙壁、建筑、物体、管道、边缘、背景、其他。

格式示例：
区域1是天空，区域2是地面，区域3是建筑。
或者：1=天空，2=地面，3=建筑。
```

**标签筛选规则：**

```python
MASK_OUT_KEYWORDS = [
    "天空", "sky", "云", "cloud",
    "黑色", "边缘", "黑边", "边框",
    "远景", "远处", "背景", "背景墙", "远景建筑",
    "模糊", "不清晰",
]
# should_mask_out(label) = any(keyword in label.lower())
```

**VLM 输出解析：** 支持 JSON / `1=标签` / `区域1是` 三种格式。

**异步并发控制：**

```python
# 使用 Semaphore 限制最大并发
semaphore = asyncio.Semaphore(max_concurrent=16)

# httpx 连接池配置
connector = aiohttp.TCPConnector(
    limit=16,           # 最大连接数
    limit_per_host=16,  # 每主机最大连接
    keepalive_timeout=30
)
timeout = aiohttp.ClientTimeout(total=240)  # 4 分钟超时
```

### 2.4 最终 Mask 生成

```
对于每个 mask 区域:
  if VLM_label 匹配 MASK_OUT_KEYWORDS:
      该区域像素 → 255 (遮蔽)
  else:
      该区域像素 → 0 (保留)
```

---

## 3. 技术指标

### 3.1 性能基准（RTX 4090 D + Ollama qwen:32b）

| 阶段 | 串行耗时 | 并发耗时（16 workers） | 加速比 |
|------|---------|---------------------|--------|
| Phase 1 (SAM2) | ~1-2s/图 | ~0.25-0.5s/图 | ~4x |
| Phase 2 (VLM) | ~3-5s/图 | ~0.3s/图 | ~10x |
| **总计** | **~4-7s/图** | **~0.4-0.8s/图** | **~8x** |

### 3.2 资源占用

| 资源 | Phase 1 | Phase 2 |
|------|---------|---------|
| GPU | RTX 4090 D 38GB | — |
| CPU | — | httpx 异步线程 |
| 内存 | ~2GB/worker | ~200MB 总计 |
| 网络 | — | localhost Ollama |

### 3.3 端到端管道耗时估算

| 数据集规模 | 串行管道 | 两阶段并行 | 异步优化后 |
|-----------|---------|-----------|-----------|
| 100 张图 | ~15 分钟 | ~5 分钟 | ~1 分钟 |
| 1000 张图 | ~2.5 小时 | ~50 分钟 | ~10 分钟 |
| 2292 张图 | ~5.7 小时 | ~2 小时 | ~25 分钟 |

---

## 4. 文件结构

```
sam_vlm_pipeline/
├── __init__.py                    # 包入口
├── phase1_sam.py                  # Phase 1: SAM2 分割
├── phase2_vlm.py                  # Phase 2: VLM 标注 (异步)
├── pipeline.py                     # 统一管道协调器
├── config.py                       # 配置管理
├── utils.py                        # 共享工具
├── run.sh                          # 启动脚本
└── requirements.txt                # 依赖列表

输出目录/
├── sam_masks/                      # Phase 1 输出
│   ├── {img}_sam_masks.json
│   ├── {img}_numbered.png
│   ├── {img}_sam_done.flag
│   └── phase1_progress.json
└── final_masks/                    # Phase 2 输出
    ├── {img}_final_mask.png
    ├── {img}_phase2_done.flag
    └── phase2_progress.json
```

---

## 5. 使用方法

### 5.1 环境准备

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载 SAM2 模型
mkdir -p weights
# 下载 sam2_hiera_large.pt 到 weights/

# 3. 启动 Ollama 服务
ollama serve
ollama pull qwen:32b
```

### 5.2 快速启动

```bash
# 方式 1: 使用启动脚本
./run.sh --images "path/to/images/*.png" \
         --output-dir ./output \
         --phase1-workers 4 \
         --phase2-workers 16

# 方式 2: Python API
python -c "
from sam_vlm_pipeline import SAMVLM pipeline
pipeline = SAMVLMPipeline(
    sam2_checkpoint='weights/sam2_hiera_large.pt',
    vlm_api_url='http://localhost:11434/v1/chat/completions',
    vlm_model='qwen:32b'
)
pipeline.run('path/to/images/', 'output/', phase1_workers=4, phase2_workers=16)
"

# 方式 3: 分阶段运行
python -m sam_vlm_pipeline.phase1_sam --images "*.png" --output-dir ./sam_masks --workers 4
python -m sam_vlm_pipeline.phase2_vlm --sam-masks-dir ./sam_masks --output-dir ./final_masks --workers 16 --watch
```

### 5.3 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sam2-checkpoint` | `weights/sam2_hiera_large.pt` | SAM2 模型路径 |
| `--vlm-api-url` | `http://localhost:11434/v1/chat/completions` | VLM API 地址 |
| `--vlm-model` | `qwen:32b` | VLM 模型名称 |
| `--phase1-workers` | 4 | SAM2 并发进程数 |
| `--phase2-workers` | 16 | VLM 异步并发数 |
| `--min-mask-area` | 500 | 最小 mask 面积 |
| `--image-max-size` | 800 | VLM 图像最大边长 |
| `--watch` | False | 持续监控新文件 |

---

## 6. 技术原理详解

### 6.1 为什么用 SAM2 而非语义分割网络？

| 方案 | 优点 | 缺点 |
|------|------|------|
| 语义分割网络 | 速度快 | 需要训练数据，类别固定 |
| **SAM2** | 无需训练，类别无关，精度高 | 速度较慢 |
| 深度阈值 | 简单 | 无法区分天空和近景白色物体 |

**选择 SAM2 的理由：**
- 不需要人工标注的训练数据
- 不受限于预定义类别，可适应任何场景
- 分割质量高，边界精确

### 6.2 为什么需要 VLM 而非规则匹配？

SAM2 产生的 mask 是**几何分割**，而非**语义分割**。同一几何区域可能有不同语义：

```
SAM2 mask 1: 包含天空 + 远处墙壁 → 无法用几何规则区分
SAM2 mask 2: 纯地面              → 规则可区分
SAM2 mask 3: 天空 + 建筑物顶      → 规则无法区分
```

**VLM 的优势：**
- 理解像素级分割区域的真实语义
- 处理复杂的混合区域
- 不需要手工设计几何规则

### 6.3 为什么不直接用深度阈值？

深度阈值（如 `depth > 50m → 遮蔽`）的问题：
- 无法区分"近处白色表面"和"远处天空"（可能深度相同）
- 无法处理黑色边缘（深度无效区域）
- 无法处理相机运动导致的边缘错位

**组合策略（最优）：**
```
if VLM_label in ["天空", "远景", "背景"]:
    mask = 255
else if depth > depth_threshold:
    mask = 255  # 深度过滤作为补充
```

### 6.4 异步并发为什么有效？

VLM 调用的时间分布：
```
总时间 = 排队等待 + 网络传输 + 模型推理 + 响应解析
       ≈ 0.1s      + 0.05s    + 5s        + 0.05s
```

16 个并发请求时，模型推理时间被完全重叠：
```
请求 1: ──────────[推理5s]──→
请求 2:    ──────────[推理5s]──→
请求 3:       ──────────[推理5s]──→
...
请求 16: ─────────────────[推理5s]──→
总耗时: ≈ 5s (而非 16×5s=80s)
```

---

## 7. 已知局限与改进方向

### 7.1 当前局限

1. **VLM 标注不稳定**：部分图像所有 mask 被标为"其他"，可能需要优化 prompt 或使用更强模型
2. **JSON 文件大**：segmentation 数组为完整 H×W 布尔矩阵，单文件可达数百 MB
3. **深度信息未集成**：当前管道未使用深度图过滤远景
4. **OOM 风险**：Phase 1 多进程共享 GPU 显存，需根据显存调整 workers

### 7.2 改进方向

| 改进 | 方法 | 预期效果 |
|------|------|---------|
| 压缩 segmentation 存储 | RLE 编码替代完整矩阵 | 文件大小减少 90% |
| 深度感知遮蔽 | 接入 MiDaS / Depth-Pro | 补充远景过滤 |
| 更强 VLM | qwen:72b 或 GPT-4o | 提高标签准确率 |
| 流式处理 | 逐张完成即写入 | 支持超大 dataset |
| 批量 VLM 调用 | 多图拼接为一张 | 减少 API 调用次数 |

---

## 8. 附录

### 8.1 依赖列表

```
# Core
torch >= 2.0
numpy
Pillow
opencv-python

# SAM2
segment-anything-2 @ git+https://github.com/facebookresearch/segment-anything-2

# VLM
httpx >= 0.27.0
aiohttp >= 3.9.0

# Pipeline
tqdm
```

### 8.2 推荐硬件配置

| 配置 | Phase 1 | Phase 2 | 说明 |
|------|---------|---------|------|
| 最低 | RTX 3060 12GB | 8 核 CPU | 1 worker |
| 推荐 | RTX 4090 D 24GB | 16 核 CPU | 4 workers |
| 最佳 | A100 40GB | 32 核 CPU | 8+ workers |

### 8.3 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| CUDA OOM | workers 过多 | 减少 `--phase1-workers` |
| VLM 超时 | 模型加载慢 | 增加 timeout 或减少 workers |
| 全部标签为"其他" | prompt 不够明确 | 优化 prompt 或换更强模型 |
| JSON 文件损坏 | 进程被强制终止 | 使用 `--force` 重新生成 |
| Ollama 内存不足 | 模型过大 | 使用 4-bit 量化版本 |

---

*文档版本：1.0*
*编写时间：2026-05-16*
