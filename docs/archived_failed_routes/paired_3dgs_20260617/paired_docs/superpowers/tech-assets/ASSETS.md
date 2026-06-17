# 技术资产沉淀总表

> 本文件汇总项目开发过程中验证过的模式、踩过的坑、可复用的代码片段。
> 按类别组织，便于未来检索和复用。

---

## 1. 已验证的设计模式

### SSH + nohup 远程后台任务模式
**场景:** 在远程服务器启动长时间运行的测试任务，不阻塞 SSH 连接
**实现:**
```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=10 remote-alias \
  'cd ~/project && export PATH=/home/user/miniconda3/envs/myenv/bin:$PATH && \
  rm -f ~/project/output_e2e_test/e2e_log.txt && \
  nohup python3 tmp_e2e_semantic_test.py > ~/project/output_e2e_test/e2e_log.txt 2>&1 & \
  echo "PID=$!"'
```
**验证状态:** ✅ 已验证
**注意事项:**
- SSH 连接断开后任务继续运行
- `ServerAliveInterval=30` 防止空闲连接被 kill
- `echo "PID=$!"` 获取后台进程 PID 方便管理
- `rm -f` 清理旧日志避免混淆
- `nohup` + 重定向 `> file 2>&1` 缺一不可

---

### E2E 测试流水线多次重试模式
**场景:** 远程测试可能因网络等原因失败，需要多次重试
**实现:**
```bash
for i in 1 2 3; do
  ssh -o BatchMode=yes remote-alias 'kill $(ps aux | grep tmp_e2e | grep -v grep | awk "{print \$2}") 2>/dev/null
  cd ~/project && nohup python3 tmp_e2e_semantic_test.py > e2e_log.txt 2>&1 & echo "PID=$!"' 2>&1 && break
  sleep 3
done
```
**验证状态:** ✅ 已验证
**注意事项:**
- `kill` 旧进程避免重复运行
- `2>/dev/null` 避免 grep 无结果时报错
- SSH 连接失败时循环重试

---

## 2. 踩坑记录 (Pitfalls)

### Bug: Opacity 双重 Sigmoid 导致渲染全白
**时间:** 2026-05-19
**症状:** 3D Gaussian Splatting 渲染输出全白/灰色，无有效颜色
**根因:** `SplatLoader` 对 opacity bytes 先做 sigmoid 得到 `_opacities`，`get_opacity_sigmoid()` 又做一次 sigmoid → gsplat 收到 ~0.56 的 effective opacity → alpha=0.0016 → 几乎透明
**修复:** 移除 `SplatLoader` 中的 sigmoid 操作，或移除 `get_opacity_sigmoid()` 中的 sigmoid
**预防:** 渲染前检查 effective opacity 分布；单测渲染管线
**关联文件:** `splatting/splats.py`

### Bug: 语义融合后所有体素为 unknown
**时间:** 2026-05-19
**症状:** E2E 测试完成但 `Voxel label distribution: unknown: 100%`，`Labeled PCD: 0/N points`
**根因:** 待查 — SAM2 生成了 mask，VLM 识别了标签，但标签未写入体素
**修复:** 待实现
**预防:** 在 E2E pipeline 每步添加 label count 断言
**关联文件:** `semantic/fuse.py`, E2E test script

### Bug: SSH 连接因空闲超时被 kill
**时间:** 2026-05-19
**症状:** 长时间运行的 SSH 会话无响应，连接被远程服务器关闭
**根因:** 服务器端 SSH 服务有空闲超时配置
**修复:** 添加 `-o ServerAliveInterval=30 -o ServerAliveCountMax=10`
**预防:** 所有 SSH 命令默认带上 keepalive 参数

### Bug: T_cw / T_wc 坐标变换混淆
**时间:** 2026-05-18
**症状:** Camera-to-world 和 world-to-camera 矩阵传递错误，渲染视角不对
**根因:** 旧版 `splats_fixed.py` 直接传 `T_cw`，重构后传 `T_wc`，不一致
**修复:** 统一坐标系约定，确保 splatting 模块与 camera loader 对齐
**预防:** 坐标系约定写入代码注释 + 单元测试验证

---

## 3. 可复用代码片段

### 远程 Python 环境激活
**来源:** E2E test SSH 会话
**语言:** Shell
**用途:** SSH 到远程服务器后激活 conda 环境并执行 Python 脚本
```bash
export PATH=/home/zsh/miniconda3/envs/semantic3d/bin:$PATH
python3 tmp_e2e_semantic_test.py
```
**验证状态:** ✅ 已验证

### 点云 PLY 文件解析（Python）
**来源:** E2E 测试结果分析
**语言:** Python
**用途:** 读取 PLY 文件中的顶点属性（坐标 + 颜色）
```python
from plyfile import PlyData
ply = PlyData.read('labeled_pointcloud.ply')
data = ply['vertex']
x, y, z = data['x'], data['y'], data['z']
r = data['red'] / 255.0
```
**验证状态:** ✅ 已验证（需 pip install plyfile）

### Plotly 交互式 3D 点云可视化
**来源:** 本次会话
**语言:** Python
**用途:** 生成可交互的 HTML 3D 可视化（旋转/缩放/点击）
```python
import plotly.graph_objects as go
import numpy as np

# 下采样减少渲染量
step = 10
fig = go.Figure(data=[go.Scatter3d(
    x=x[::step], y=y[::step], z=z[::step],
    mode='markers',
    marker=dict(size=2, color=np.stack([r,g,b], axis=1) * 255)
)])
fig.write_html('pointcloud_visualization.html')
```
**验证状态:** ⚠️ 待验证（脚本已提供，尚未执行）
**依赖:** `pip install plotly plyfile`

---

## 4. 工具/命令备忘

### SSH + nohup 远程后台任务
**场景:** 启动长时间测试，不阻塞终端
**用法:**
```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
  -o ServerAliveInterval=30 remote-alias \
  'cd ~/project && nohup python3 script.py > log.txt 2>&1 & echo "PID=$!"'
```
**输出解读:** `PID=12345` 表示进程号

### 远程查看日志（实时 tail）
**场景:** 监控正在运行的 E2E 测试进度
**用法:**
```bash
ssh -o BatchMode=yes remote-alias 'tail -f ~/project/output_e2e_test/e2e_log.txt'
```

### 杀死远程 Python 进程
**场景:** 重启测试前清理旧进程
**用法:**
```bash
ssh -o BatchMode=yes remote-alias \
  'kill $(ps aux | grep tmp_e2e_semantic | grep -v grep | awk "{print \$2}") 2>/dev/null'
```

### 批量下载远程文件
**场景:** 将 E2E 输出文件拉回本地分析
**用法:**
```bash
scp remote-alias:~/project/output_e2e_test/{e2e_log.txt,labeled_pointcloud.ply,voxel_labels.json} ./
```

---

## 5. 架构决策记录 (ADR)

### ADR-001: E2E 测试结果以点云 PLY + JSON 体素标签输出
**时间:** 2026-05-19
**上下文:** Semantic3D pipeline 需要同时输出可可视化的点云和便于程序读取的体素标签
**决策:** 输出 `labeled_pointcloud.ply`（颜色编码标签）和 `voxel_labels.json`（体素坐标→标签映射）
**备选:** 单输出 HDF5 / NPY
**理由:** PLY 可直接用 MeshLab/Open3D/Plotly 可视化；JSON 便于调试和人类阅读
**后果:** 文件较大（664MB + 261MB），但无需额外转换

### ADR-002: SSH keepalive 参数作为默认配置
**时间:** 2026-05-19
**上下文:** 远程测试任务需要长时间稳定运行，但服务器有空闲超时
**决策:** 所有 SSH 命令默认添加 `-o ServerAliveInterval=30 -o ServerAliveCountMax=10`
**备选:** tmux/screen 持久会话
**理由:** 参数方式侵入性最小，无需在服务器安装额外工具
**后果:** 每次命令稍多几个字节参数

---

## 6. 待解决的技术债务

| ID | 问题 | 影响 | 优先级 |
|----|------|------|--------|
| TD-001 | 语义融合中所有体素为 unknown，根因未明 | E2E pipeline 无法产出有效语义标签 | 🔴 高 |
| TD-002 | labeled_pointcloud.ply 中 0 点有颜色 | 融合后的颜色/标签未写入点云 | 🔴 高 |
| TD-003 | 无 E2E pipeline 单元测试 | 无法自动化检测融合失败 | 🟡 中 |
| TD-004 | voxel_labels.json 结构未验证 | 249MB JSON 格式正确性未知 | 🟡 中 |
