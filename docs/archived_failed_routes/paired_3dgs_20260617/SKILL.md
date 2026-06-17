---
name: superpowers-comprehensive-workflow
description: Use when starting any development task, feature implementation, bug fix, or code modification. This is the master workflow skill that governs all agentic development activities with mandatory session logging and technical asset documentation.
---

# Superpowers 综合开发总则

## 概述

本 Skill 是**单一入口总则**，覆盖从需求理解到交付完成的完整开发闭环。所有开发活动（新功能、Bug 修复、重构、文档编写）**必须**遵循此流程。

**核心原则：**
- **纪律优先于速度**：流程步骤不可跳过，不可因"简单"或"紧急"而省略
- **证据优先于主张**：任何完成声明必须有可验证的输出证据
- **日志优先于记忆**：每一次操作、每一轮会话必须落盘记录，不可依赖上下文记忆
- **资产沉淀**：踩过的坑、验证过的方案、可复用的代码片段必须归档到技术资产文档

**两个硬文档（不可省略）：**
1. **`docs/superpowers/session-logs/YYYY-MM-DD-HH-MM--session.md`** — 会话操作日志
2. **`docs/superpowers/tech-assets/ASSETS.md`** — 可沉淀技术资产总表

---

## 全局触发规则

收到任何用户消息后，按以下顺序判断：

```
用户消息接收
  ├─ 是否已脑暴过？
  │   ├─ 否 → 触发 brainstorming
  │   └─ 是 → 继续
  ├─ 是否有适用 Skill？（即使1%概率）
  │   ├─ 是 → 加载并遵循
  │   └─ 否 → 直接响应
  └─ 任何操作前 → 先写日志
```

**铁律：如果某个子 Skill 可能适用，你没有选择权，必须使用它。**

---

## 第一阶段：启动与脑暴 (brainstorming)

**触发条件：** 任何创造性工作（新功能、组件构建、行为修改、配置变更）开始前。

**禁止：** 未获设计批准前，禁止调用任何实现类 Skill、禁止写代码、禁止脚手架。

### 流程

1. **探索项目上下文** — 检查文件、文档、近期提交
2. **评估范围** — 若涉及多个独立子系统，先分解为子项目
3. **提出澄清问题** — 一次一个问题，理解目的/约束/成功标准
4. **提供 2-3 种方案** — 含权衡分析与推荐
5. **分节展示设计** — 每节结束后请求用户确认
6. **编写设计文档** — 保存至 `docs/superpowers/specs/YYYY-MM-DD--design.md`
7. **设计自检** — 扫描占位符、矛盾、歧义、范围
8. **用户审阅规格** — 用户确认规格后方可继续
9. **触发 writing-plans** — 编写实施计划

### 设计文档自检清单

- [ ] 无 "TBD"/"TODO"/不完整段落
- [ ] 各节内部一致，架构与功能描述匹配
- [ ] 范围聚焦，适合单一实施计划
- [ ] 无歧义需求，所有要求明确唯一解释

### 会话日志记录要求

**每轮交互后必须追加记录：**

```markdown
### Round N: [主题]
**时间:** YYYY-MM-DD HH:MM
**用户输入:** [摘要]
**我的动作:** [做了什么]
**输出/结果:** [关键输出或用户反馈]
**决策:** [用户批准/要求修改/待确认]
```

---

## 第二阶段：隔离工作区 (using-git-worktrees)

**触发条件：** 设计批准后，实施开始前。

**核心原则：** 检测现有隔离 → 优先原生工具 → Git worktree 兜底。

### 流程

1. **检测现有隔离**
   ```bash
   GIT_DIR=$(cd "$(git rev-parse --git-dir)" 2>/dev/null && pwd -P)
   GIT_COMMON=$(cd "$(git rev-parse --git-common-dir)" 2>/dev/null && pwd -P)
   ```
   - 若 `GIT_DIR != GIT_COMMON` 且非子模块：已在 worktree 中，跳到项目设置
   - 若 `GIT_DIR == GIT_COMMON`：正常仓库，需创建隔离

2. **征求用户同意**
   > "是否创建隔离 worktree？可保护当前分支不受修改影响。"

3. **创建隔离（优先原生工具）**
   - 优先使用平台原生 worktree 工具（如 `EnterWorktree`）
   - 无原生工具时，使用 Git worktree fallback：
     - 目录优先级：`.worktrees/` > `worktrees/` > `~/.config/superpowers/worktrees/$project/`
     - 项目本地目录必须验证已在 `.gitignore` 中
     - 创建：`git worktree add "$path" -b "$BRANCH_NAME"`

4. **项目设置**
   ```bash
   # Node.js
   if [ -f package.json ]; then npm install; fi
   # Rust
   if [ -f Cargo.toml ]; then cargo build; fi
   # Python
   if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
   if [ -f pyproject.toml ]; then poetry install; fi
   # Go
   if [ -f go.mod ]; then go mod download; fi
   ```

5. **验证干净基线**
   ```bash
   npm test / cargo test / pytest / go test ./...
   ```
   - 若测试失败：报告失败，询问是否继续或调查
   - 若测试通过：报告就绪

### 会话日志记录要求

记录 worktree 路径、分支名、基线测试结果。

---

## 第三阶段：编写实施计划 (writing-plans)

**触发条件：** 设计规格已批准，需转化为可执行任务。

**核心原则：** 假设执行者零上下文、品味可疑、厌恶测试。每一步 2-5 分钟粒度。

### 计划文档头部（强制）

```markdown
# [功能名] 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 按任务实施。步骤使用 checkbox (`- [ ]`) 语法追踪。

**目标:** [一句话描述构建内容]
**架构:** [2-3句方法描述]
**技术栈:** [关键技术/库]

---
```

### 任务结构（每任务）

```markdown
### Task N: [组件名]

**文件:**
- 创建: `exact/path/to/file.py`
- 修改: `exact/path/to/existing.py:123-145`
- 测试: `tests/exact/path/to/test.py`

- [ ] **Step 1: 编写失败测试**
  ```python
  def test_specific_behavior():
      result = function(input)
      assert result == expected
  ```

- [ ] **Step 2: 运行测试确认失败**
  运行: `pytest tests/path/test.py::test_name -v`
  预期: FAIL with "function not defined"

- [ ] **Step 3: 编写最小实现**
  ```python
  def function(input):
      return expected
  ```

- [ ] **Step 4: 运行测试确认通过**
  运行: `pytest tests/path/test.py::test_name -v`
  预期: PASS

- [ ] **Step 5: 提交**
  ```bash
  git add tests/path/test.py src/path/file.py
  git commit -m "feat: add specific feature"
  ```
```

### 禁止的占位符

- "TBD", "TODO", "稍后实现", "补充细节"
- "添加适当的错误处理"/"添加验证"/"处理边界情况"
- "为上述内容编写测试"（无实际测试代码）
- "类似于 Task N"（任务可能被乱序阅读）
- 仅有描述无代码的步骤
- 引用未定义的类型/函数/方法

### 计划自检

1. **规格覆盖**：逐节检查规格，确认每个需求都有对应任务
2. **占位符扫描**：检查上述禁止模式
3. **类型一致性**：跨任务的方法签名、属性名是否一致

### 保存位置

`docs/superpowers/plans/YYYY-MM-DD--plan.md`

### 执行方式选择

计划完成后提供两种方式：
1. **子代理驱动（推荐）** — 每任务派新鲜子代理，任务间审查，快速迭代
2. **本会话执行** — 使用 executing-plans，批量执行带检查点

---

## 第四阶段：执行实施 (subagent-driven-development / executing-plans)

### 4A. 子代理驱动开发（推荐，同一会话）

**触发条件：** 计划已就绪，任务相对独立，当前会话执行。

**核心原则：** 每任务新鲜子代理 + 两阶段审查（规格合规 → 代码质量）。

#### 流程

```
读取计划 → 提取所有任务全文 → 创建 TodoWrite
  → 派发实现子代理（implementer-prompt.md）
    → 子代理提问？ → 回答后重新派发
    → 子代理实现、测试、提交、自检
      → 派发规格审查子代理（spec-reviewer-prompt.md）
        → 不合规？ → 实现子代理修复 → 重新审查
        → 合规？ → 派发代码质量审查子代理（code-quality-reviewer-prompt.md）
          → 不通过？ → 实现子代理修复 → 重新审查
          → 通过？ → 标记任务完成 → 下一任务
  → 所有任务完成 → 最终代码审查 → 触发 finishing-a-development-branch
```

**模型选择策略：**
- 机械实现任务（1-2 文件，完整规格）：用最快最便宜的模型
- 集成与判断任务（多文件协调、调试）：用标准模型
- 架构、设计、审查任务：用最强模型

**子代理状态处理：**
- **DONE**：进入规格审查
- **DONE_WITH_CONCERNS**：阅读关切，若涉及正确性/范围则先处理，若是观察性则记录并继续
- **NEEDS_CONTEXT**：提供缺失上下文，重新派发
- **BLOCKED**：评估阻塞原因（上下文不足→补充后重派；推理不足→换更强模型；任务过大→拆分；计划错误→上报用户）

**禁止：**
- 未获用户明确同意在主分支上开始实现
- 跳过任一阶段审查
- 并行派发多个实现子代理（会冲突）
- 让子代理读取计划文件（应提供全文）
- 跳过场景设置上下文
- 忽略子代理问题
- 规格审查未通过就进入代码质量审查
- 审查发现问题未修复就进入下一任务

### 4B. 计划执行（跨会话）

**触发条件：** 计划在另一会话执行，或平台不支持子代理。

#### 流程

1. **加载并审查计划** — 识别问题，有疑虑先与用户确认
2. **执行任务** — 每项标记 in_progress → 按步骤执行 → 运行验证 → 标记 completed
3. **完成开发** — 触发 finishing-a-development-branch

**立即停止条件：**
- 遇到阻塞（缺失依赖、测试失败、指令不清）
- 计划有关键缺口
- 不理解指令
- 验证反复失败

**禁止猜测，必须询问。**

### 会话日志记录要求

**每任务完成后追加：**

```markdown
### Task N 执行记录
**时间:** YYYY-MM-DD HH:MM
**子代理/执行方式:** [模型/本会话]
**实现摘要:** [做了什么]
**测试状态:** [通过/失败/阻塞]
**审查结果:**
  - 规格合规: [通过/问题列表]
  - 代码质量: [通过/问题列表]
**提交SHA:** [git SHA]
**技术资产沉淀:**
  - 踩坑: [如有]
  - 可复用代码: [代码片段或文件路径]
  - 模式/技巧: [如有]
```

---

## 第五阶段：测试驱动开发 (test-driven-development)

**触发条件：** 实现任何功能或修复任何 Bug 时，写实现代码之前。

**核心原则：** 先写测试，看它失败，写最小代码通过，重构。

### 铁律

```
NO PRODUCTION CODE WITHOUT A FAILING TEST FIRST
```

先写代码后补测试？**删除它，重新开始。**

### RED-GREEN-REFACTOR 循环

1. **RED** — 编写一个最小测试，展示期望行为
   - 单一行为、名称清晰、使用真实代码（非 mock，除非不可避免）

2. **验证 RED** — 运行测试，确认它**正确失败**
   - 禁止跳过此步骤
   - 若测试通过：说明在测已有行为，修复测试
   - 若测试报错：修复错误，重跑直到正确失败

3. **GREEN** — 编写最小代码使测试通过
   - 不添加功能、不重构其他代码、不"改进"超测试范围

4. **验证 GREEN** — 运行测试，确认通过，其他测试仍通过

5. **REFACTOR** — 仅在绿灯后清理
   - 消除重复、改进命名、提取辅助函数
   - 保持测试通过，不添加行为

### 常见借口与真相

| 借口 | 真相 |
|------|------|
| "太简单不用测" | 简单代码也会坏。测试只需30秒。 |
| "稍后补测试" | 事后测试立即通过，证明不了什么。 |
| "事后测试目标一样" | 事后="这做了什么"；事前="这应该做什么"。 |
| "已经手动测过所有边界" | 手动≠系统。无记录，不可重跑。 |
| "删了X小时工作太浪费" | 沉没成本谬误。无验证代码是技术债。 |
| "TDD太教条，我务实" | TDD就是务实：提前发现Bug、防回归、文档化行为。 |

### 验证清单

- [ ] 每个新函数/方法都有测试
- [ ] 每个测试在实现前都看过它失败
- [ ] 每个测试因正确原因失败（功能缺失，非拼写错误）
- [ ] 写了最小通过代码
- [ ] 所有测试通过
- [ ] 输出干净（无错误、警告）
- [ ] 测试使用真实代码（mock 仅不可避免时）
- [ ] 覆盖边界情况和错误路径

---

## 第六阶段：系统调试 (systematic-debugging)

**触发条件：** 任何 Bug、测试失败、意外行为、性能问题、构建失败、集成问题。

**核心原则：** 无根因调查，禁止提修复方案。症状修复=失败。

### 铁律

```
NO FIXES WITHOUT ROOT CAUSE INVESTIGATION FIRST
```

### 四阶段流程

#### Phase 1: 根因调查（修复前必须完成）

1. **仔细阅读错误信息** — 不跳过错误或警告，它们常含精确解决方案
2. **稳定复现** — 能否可靠触发？ exact steps？每次发生？
3. **检查近期变更** — git diff、近期提交、新依赖、配置变更、环境差异
4. **多组件系统取证** — 在组件边界添加诊断日志：
   ```bash
   # 每层：记录进入数据、离开数据、环境/配置传播、每层状态
   echo "=== Layer X: Input ==="
   echo "$input_data"
   echo "=== Layer X: Output ==="
   echo "$output_data"
   ```
5. **追踪数据流** — 错误值从哪来？谁用坏值调用了它？逐层向上直到源头

#### Phase 2: 模式分析

1. **找到工作示例** — 同类正常代码在哪？
2. **对照参考实现** — 完整阅读，不略读
3. **识别差异** — 正常与异常之间每个差异，不论多小
4. **理解依赖** — 需要什么组件、设置、配置、环境？

#### Phase 3: 假设与测试

1. **形成单一假设** — 清晰陈述"我认为X是根因，因为Y"
2. **最小化测试** — 最小改动验证假设，一次一个变量
3. **验证后继续** — 有效→Phase 4；无效→新假设
4. **不懂就说** — "我不懂X"，不假装懂，求助或研究

#### Phase 4: 实现修复

1. **创建失败测试** — 最简复现，自动化或一次性脚本
2. **单一修复** — 针对识别的根因，一处改动，不"顺手"重构
3. **验证修复** — 测试通过？其他测试未坏？问题真解决？
4. **修复无效？**
   - 尝试次数 < 3：回到 Phase 1，用新信息重新分析
   - 尝试次数 ≥ 3：**停止，质疑架构** — 可能是架构问题，与用户讨论

### 调试技术资产沉淀

**每次调试完成后，强制追加到技术资产文档：**

```markdown
### Debug Record: [简短描述]
**时间:** YYYY-MM-DD HH:MM
**症状:** [错误信息/现象]
**根因:** [真正原因]
**修复:** [代码/配置变更]
**教训:** [可复用的排查思路或模式]
**关联文件:** [相关文件路径]
```

---

## 第七阶段：代码审查 (requesting-code-review / receiving-code-review)

### 7A. 请求代码审查

**触发条件：** 每任务完成（子代理驱动中强制）、大功能完成、合并前。

**核心原则：** 早审查、常审查。派审查子代理，不给它你的会话历史。

#### 流程

1. **获取 git SHAs**
   ```bash
   BASE_SHA=$(git rev-parse HEAD~1)  # 或 origin/main
   HEAD_SHA=$(git rev-parse HEAD)
   ```

2. **派发审查子代理** — 使用 `general-purpose` 类型，填充模板 `code-reviewer.md`
   - 占位符：`{DESCRIPTION}`、`{PLAN_OR_REQUIREMENTS}`、`{BASE_SHA}`、`{HEAD_SHA}`

3. **处理反馈**
   - Critical：立即修复
   - Important：继续前修复
   - Minor：记录稍后处理
   - 审查错误：有理有据地反驳

### 7B. 接收代码审查

**触发条件：** 收到审查反馈，实施建议前。

**核心原则：** 技术评估，非情绪表演。验证后实施，有理有据地反驳。

#### 响应模式

```
1. READ: 完整阅读反馈，不立即反应
2. UNDERSTAND: 用自己的话重述需求（或提问）
3. VERIFY: 对照代码库现实检查
4. EVALUATE: 对此代码库技术上正确？
5. RESPOND: 技术确认或有理有据的反驳
6. IMPLEMENT: 一项一项来，每项测试
```

#### 禁止的回应

- ❌ "You're absolutely right!"
- ❌ "Great point!" / "Excellent feedback!"
- ❌ "Let me implement that now"（验证前）

#### 正确回应

- ✅ 重述技术要求
- ✅ 提问澄清
- ✅ 技术上有误则反驳
- ✅ 直接开始工作（行动 > 言语）

#### 实施顺序

1. 先澄清所有不清楚的项
2. 按此顺序实施：
   - 阻塞问题（崩溃、安全）
   - 简单修复（拼写、导入）
   - 复杂修复（重构、逻辑）
3. 每项单独测试
4. 验证无回归

#### YAGNI 检查

若审查建议"正确实现"某功能：
```bash
grep codebase for actual usage
```
- 未使用："此端点未被调用。删除它（YAGNI）？"
- 已使用：则正确实现

---

## 第八阶段：完成前验证 (verification-before-completion)

**触发条件：** 即将声明工作完成、修复、通过，或提交/创建 PR 前。

**核心原则：** 无新鲜验证证据，禁止任何完成声明。

### 铁律

```
NO COMPLETION CLAIMS WITHOUT FRESH VERIFICATION EVIDENCE
```

### 验证门函数

```
声明任何状态前：
1. IDENTIFY: 什么命令能证明此声明？
2. RUN: 执行完整命令（新鲜、完整）
3. READ: 阅读完整输出，检查退出码，统计失败数
4. VERIFY: 输出是否确认声明？
   - 否：陈述实际状态+证据
   - 是：带着证据陈述声明
5. ONLY THEN: 做出声明

跳过任何一步 = 撒谎，非验证
```

### 常见声明与所需证据

| 声明 | 需要 | 不足够 |
|------|------|--------|
| 测试通过 | 测试命令输出：0 失败 | 上次运行、"应该通过" |
| Linter 干净 | Linter 输出：0 错误 | 部分检查、推断 |
| 构建成功 | 构建命令：exit 0 | Linter 通过、日志看起来正常 |
| Bug 修复 | 原症状测试：通过 | 代码改了、假设已修复 |
| 回归测试有效 | 红绿循环验证 | 测试通过一次 |
| 代理完成 | VCS diff 显示变更 | 代理报告"成功" |
| 需求满足 | 逐条核对清单 | 测试通过 |

### 禁止的措辞

- "should work now"
- "probably passes"
- "seems correct"
- "Great!" / "Perfect!" / "Done!"（验证前）
- 任何暗示成功但未运行验证的措辞

---

## 第九阶段：分支收尾 (finishing-a-development-branch)

**触发条件：** 实现完成，所有测试通过，需决定如何集成。

**核心原则：** 验证测试 → 检测环境 → 展示选项 → 执行选择 → 清理。

### 流程

1. **验证测试** — 运行完整测试套件，失败则停止
2. **检测环境** — 判断 workspace 状态：
   - `GIT_DIR == GIT_COMMON`：正常仓库
   - `GIT_DIR != GIT_COMMON`，命名分支：标准选项
   - `GIT_DIR != GIT_COMMON`，detached HEAD：缩减选项（无 merge）

3. **确定基分支**
   ```bash
   git merge-base HEAD main 2>/dev/null || git merge-base HEAD master 2>/dev/null
   ```

4. **展示选项**

   **正常仓库/命名分支 worktree（4 选项）：**
   ```
   实现完成。请选择：
   1. 本地合并回 [base-branch]
   2. 推送并创建 Pull Request
   3. 保持分支现状（稍后处理）
   4. 丢弃此工作
   ```

   **Detached HEAD（3 选项）：**
   ```
   实现完成。当前为 detached HEAD（外部管理）。
   1. 推送为新分支并创建 Pull Request
   2. 保持现状（稍后处理）
   3. 丢弃此工作
   ```

5. **执行选择**
   - **选项1 本地合并**：checkout base → pull → merge → 验证测试 → 清理 worktree → 删除分支
   - **选项2 创建 PR**：推送分支 → `gh pr create` → **不清理 worktree**（用户需迭代）
   - **选项3 保持**：报告分支名和 worktree 路径
   - **选项4 丢弃**：要求输入"discard"确认 → 清理 worktree → 强制删除分支

6. **清理 Workspace**（仅选项1和4）
   - 正常仓库：无需清理
   - Superpowers 创建的 worktree（路径含 `.worktrees/`、`worktrees/`、`~/.config/superpowers/worktrees/`）：`git worktree remove` + `git worktree prune`
   - 其他 worktree：不删除，归宿主环境管理

---

## 第十阶段：并行代理调度 (dispatching-parallel-agents)

**触发条件：** 面临 2+ 独立任务，无共享状态或顺序依赖。

**核心原则：** 每个独立问题域一个代理，并行工作。

### 使用时机

- 3+ 测试文件因不同根因失败
- 多个子系统独立损坏
- 每个问题无需其他问题上下文即可理解
- 调查间无共享状态

### 不使用时机

- 失败相关（修复一个可能修复其他）
- 需要理解完整系统状态
- 代理会互相干扰（编辑同一文件、使用同一资源）

### 代理提示结构

1. **聚焦** — 单一清晰问题域
2. **自包含** — 理解问题所需的全部上下文
3. **输出具体** — 代理应返回什么？

---

## 文档规范：会话日志与技术资产

### A. 会话日志 (`docs/superpowers/session-logs/`)

**每轮会话必须创建或追加的文件：**

```
docs/superpowers/session-logs/
  └── YYYY-MM-DD-HH-MM--session.md   # 新会话新建文件
```

**文件结构：**

```markdown
# Session Log: [主题/任务名]

**会话ID:** [唯一标识]
**开始时间:** YYYY-MM-DD HH:MM
**关联设计:** [design.md 路径]
**关联计划:** [plan.md 路径]
**工作分支:** [branch name]
**Worktree路径:** [path]

---

## Round 1: [主题]
**时间:** HH:MM
**用户输入:** [原文或摘要]
**触发Skill:** [skill name]
**执行动作:** [具体做了什么]
**命令输出:** [关键命令及其输出（精简）]
**结果/决策:** [用户反馈或自动结论]
**提交SHA:** [如有]

## Round 2: [主题]
...

## 会话结束
**结束时间:** HH:MM
**最终状态:** [完成/阻塞/待续]
**交付物:**
  - [文件路径或功能]
**待办:**
  - [如有]
```

**记录原则：**
- 每轮用户消息+我的响应构成一个 Round
- 关键命令必须记录（含输出摘要）
- 用户决策/批准必须明确记录
- 提交 SHA 必须记录
- 阻塞/失败必须记录原因

### B. 技术资产总表 (`docs/superpowers/tech-assets/ASSETS.md`)

**持续追加的单一文件：**

```markdown
# 技术资产沉淀总表

> 本文件汇总项目开发过程中验证过的模式、踩过的坑、可复用的代码片段。
> 按类别组织，便于未来检索和复用。

---

## 1. 已验证的设计模式

### [模式名]
**来源:** [Task/会话]
**场景:** [何时使用]
**实现:** [代码片段或文件路径]
**验证状态:** [已验证/待验证]
**注意事项:** [边界条件、限制]

---

## 2. 踩坑记录 (Pitfalls)

### [问题简述]
**时间:** YYYY-MM-DD
**症状:** [错误信息/现象]
**根因:** [真正原因]
**修复:** [解决方案]
**预防:** [如何避免]
**关联文件:** [相关路径]

---

## 3. 可复用代码片段

### [片段名]
**来源:** [Task/文件]
**语言:** [Python/TS/Shell...]
**用途:** [一句话]
```[代码块]```
**依赖:** [需要什么库/环境]
**验证状态:** [已验证/待验证]

---

## 4. 工具/命令备忘

### [命令/工具名]
**场景:** [何时使用]
**用法:** ```[命令]```
**输出解读:** [关键字段含义]
**常见错误:** [及解决]

---

## 5. 架构决策记录 (ADR)

### [决策名]
**时间:** YYYY-MM-DD
**上下文:** [决策背景]
**决策:** [选择了什么]
**备选:** [考虑过什么]
**理由:** [为什么]
**后果:** [正面/负面]
```

**沉淀时机：**
- 每次调试完成后 → 踩坑记录
- 每次任务完成后 → 可复用代码/模式
- 每次架构选择后 → ADR
- 发现有用工具/命令 → 工具备忘

---

## 快速参考：技能优先级与触发

| 优先级 | Skill | 触发条件 | 关键产出 |
|--------|-------|----------|----------|
| 1 | using-superpowers | 任何消息 | 技能加载决策 |
| 2 | brainstorming | 任何创造性工作 | design.md + 日志 |
| 3 | using-git-worktrees | 设计批准后 | 隔离工作区 + 日志 |
| 4 | writing-plans | 规格批准后 | plan.md + 日志 |
| 5 | subagent-driven-development | 计划就绪，同会话 | 代码 + 审查记录 + 日志 |
| 5 | executing-plans | 计划就绪，跨会话 | 代码 + 日志 |
| 6 | test-driven-development | 写实现代码前 | 测试 + 实现 + 日志 |
| 7 | systematic-debugging | 任何 Bug/失败 | 根因 + 修复 + 技术资产 |
| 8 | requesting-code-review | 任务完成/合并前 | 审查报告 + 日志 |
| 8 | receiving-code-review | 收到反馈 | 修复 + 日志 |
| 9 | verification-before-completion | 任何完成声明前 | 验证证据 + 日志 |
| 10 | finishing-a-development-branch | 全部完成 | 集成 + 清理 + 日志 |
| - | dispatching-parallel-agents | 2+ 独立任务 | 并行结果 + 日志 |
| - | writing-skills | 创建新 skill | 新 skill + 测试 + 日志 |

---

## 常见错误与纠正

| 错误 | 纠正 |
|------|------|
| "这个很简单，不用设计" | 简单项目未经审视的假设造成最大浪费。设计可短，但必须呈现并获批。 |
| "紧急，没时间走流程" | 系统方法比 guess-and-check 更快。流程是加速剂，非阻碍。 |
| "我已经手动测过了" | 手动≠系统。无记录，不可重跑，压力下易遗漏。 |
| "删了X小时工作太浪费" | 沉没成本谬误。保留无验证代码是技术债。 |
| "先修复再调查" | 无根因的修复制造新 Bug。Phase 1 完成前禁止提方案。 |
| "应该通过了" | 运行命令。阅读输出。然后声明。 |
| "跳过日志，我记得" | 上下文会丢失、会话会重置。日志是唯一的真相来源。 |
| "这个坑我记住了" | 人脑不可靠。写入 ASSETS.md 才是沉淀。 |
| "事后补测试一样" | 事后测试验证的是"你写了什么"；事前测试定义的是"应该做什么"。 |
| "子代理说成功了" | 验证 VCS diff，独立确认，再声明。 |

---

## 哲学

- **测试驱动开发** — 始终先写测试
- **系统优先于随意** — 流程胜过猜测
- **复杂度削减** — 简洁是首要目标
- **证据优于主张** — 声明成功前先验证
- **日志优于记忆** — 落盘记录是不可省略的步骤
- **沉淀优于遗忘** — 技术资产是项目的复利
