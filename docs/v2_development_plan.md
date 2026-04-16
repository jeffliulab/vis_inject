# VisInject v2.0→v2.5 开发计划与进度管理

> 本文档是 v2.0 开发的**执行计划**，按 agent-rules 的多阶段进度管理方式组织。每个版本有明确的交付物、验收标准和执行步骤。

---

## 版本总览

```
已完成 (Completed)
  V1.0   三阶段攻击流水线 + 双维度评估 + 迁移性测试
         （2026-04-16 归档，tag v1.0，branch v1.0-release）

当前开发 (Current Development)
  V2.0   五类攻击实现 + 无防御基线评估（进行中）
         详细任务 → 下方 V2.0 章节

计划中 (Planned)
  V2.1P  三层防御实现 + 攻防交叉评估
  V2.2P  闭源模型迁移性测试（ChatGPT / Gemini 手动测试）
  V2.3P  综合攻防矩阵 + 隐蔽性分析
  V2.4P  实验报告 + PPT + Demo 全面更新
  V2.5P  封版 + HuggingFace 同步 + 课程提交
```

---

## V2.0 — 五类攻击实现 + 无防御基线

### 目标
实现 5 种攻击方法，在本地 4090 上对 3 个开源 VLM 跑无防御基线评估。

### 交付物
- [ ] `attack/typographic.py` — C2 排版注入
- [ ] `attack/steganography.py` — C3 隐写注入
- [ ] `attack/crossmodal.py` — C4 跨模态攻击
- [ ] `attack/spoofing.py` — C5 场景伪装
- [ ] 无防御基线数据：5 攻击 × 3 VLM × 7 图片 × 15 问题

### 执行步骤

**Step 2.0.1**: 搭建 v2.0 评估框架
```
做什么：
  - 更新 src/config.py 添加 V2_ATTACK_CONFIG（各攻击参数）
  - 编写统一的攻击接口基类 attack/base.py
  - 更新 evaluate/judge.py 支持批量评估多种攻击
  
产出：attack/base.py, 更新的 config.py
验证：python -c "from attack.base import AttackBase; print('OK')"
```

**Step 2.0.2**: 实现 C2 排版攻击
```
做什么：
  - 实现 attack/typographic.py（三个子变体：高/低可见度 + 场景融合）
  - 用 PIL 渲染文本，支持 alpha/颜色/位置/字号参数
  - 生成 7 张图 × 7 个 prompt = 49 组攻击图片

产出：attack/typographic.py + 49 张攻击图片
验证：PSNR 测量 + 人工目检
```

**Step 2.0.3**: 实现 C3 隐写攻击
```
做什么：
  - 实现 attack/steganography.py（LSB + DCT 两种方法）
  - LSB：自适应位置选择 + 多级嵌入深度
  - DCT：8×8 块变换 + 中频系数嵌入

产出：attack/steganography.py + 49 张攻击图片
验证：PSNR > 35dB + steganalysis 工具检测
```

**Step 2.0.4**: 实现 C4 跨模态攻击
```
做什么：
  - 实现 attack/crossmodal.py（简化版 JPS）
  - 图片通道：嵌入部分目标内容（用排版或像素）
  - 文本通道：配套 prompt 引导 VLM 补全

产出：attack/crossmodal.py + 49 组 (图片, prompt) 对
验证：手动检查几个案例的攻击逻辑
```

**Step 2.0.5**: 实现 C5 场景伪装攻击
```
做什么：
  - 实现 attack/spoofing.py（通知/弹窗/水印三种场景）
  - 用 PIL 绘制 UI 元素模板
  - 支持 iOS/Android/浏览器风格

产出：attack/spoofing.py + 49 张攻击图片
验证：目检 UI 元素是否逼真
```

**Step 2.0.6**: 本地 4090 全量评估（无防御基线）
```
做什么：
  - 加载 Qwen2.5-VL-3B + BLIP-2 + DeepSeek-VL-1.3B（共 ~15GB）
  - 对每种攻击生成 response pairs
  - 用 v2 judge 评估 affected + injected

测试范围：
  5 攻击 × 3 VLM × 7 图片 × 15 问题 = 1,575 pairs
  （C1 直接复用 v1.0 数据）

产出：outputs/v2_experiments/ 下的 response_pairs + judge_results
验证：每种攻击至少有 1 个 VLM 的 ASR 数据
```

### 验收标准
- [ ] 5 个 attack/*.py 文件都能运行
- [ ] 无防御基线数据表：

|  | Qwen2.5 | BLIP-2 | DeepSeek |
|--|---------|--------|----------|
| C1 像素 | v1.0 数据 | v1.0 数据 | v1.0 数据 |
| C2 排版 | ?% | ?% | ?% |
| C3 隐写 | ?% | ?% | ?% |
| C4 跨模态 | ?% | ?% | ?% |
| C5 伪装 | ?% | ?% | ?% |

---

## V2.1P — 三层防御实现

### 目标
实现 3 种防御方案，每种有代码 + 理论 + 量化效果。

### 交付物
- [ ] `defense/preprocessing.py` — D1 输入预处理
- [ ] `defense/bottleneck.py` — D2 Q-Former 瓶颈
- [ ] `defense/ensemble.py` — D3 共识投票
- [ ] 每种防御对每种攻击的效果数据

### 执行步骤

**Step 2.1.1**: D1 输入预处理
```
做什么：
  - JPEG 压缩（quality=75）
  - 高斯模糊（sigma=1.0）
  - OCR 检测（pytesseract）
  - 频率域异常检测

产出：defense/preprocessing.py
验证：对 C2 排版攻击图片处理后，人工判断文字是否被消除
```

**Step 2.1.2**: D2 Q-Former 瓶颈
```
做什么：
  - 分析 BLIP-2 Q-Former 为什么能免疫（v1.0 发现）
  - 实现简化版信息瓶颈层（32 query tokens + cross-attention）
  - 集成到 Qwen wrapper 作为可选防御层

产出：defense/bottleneck.py + 理论分析文档
验证：Qwen + 瓶颈层 vs 原始 Qwen，对 C1 和 C2 的 ASR 对比
```

**Step 2.1.3**: D3 共识投票
```
做什么：
  - 多 VLM 并行推理 + 输出一致性检测
  - 阈值策略：文本相似度 < 0.5 → 告警

产出：defense/ensemble.py
验证：对已知攻击图片的检测准确率
```

### 验收标准
- [ ] 3 个 defense/*.py 都能运行
- [ ] 每种防御有理论分析（为什么能/不能防住某类攻击）

---

## V2.2P — 闭源模型迁移性测试

### 目标
测试 5 种攻击在 ChatGPT / Gemini 上的迁移效果。

### 执行步骤
```
做什么：
  - 每种攻击选 3 个最高 ASR 案例
  - 手动发给 ChatGPT + Gemini
  - 记录回答，用 v2 judge 评估

测试量：5 攻击 × 3 案例 × 2 闭源模型 = 30 组手动测试
```

### 验收标准
- [ ] 闭源模型测试数据表完成
- [ ] 至少 1 种攻击在闭源模型上有可量化的效果

---

## V2.3P — 综合攻防矩阵 + 隐蔽性分析

### 目标
生成完整的 5×4 攻防交叉评估矩阵 + 隐蔽性对比。

### 执行步骤

**Step 2.3.1**: 攻防交叉评估
```
对每种攻击 × 每种防御：
  - 先施加防御
  - 再用防御后的图片跑 VLM
  - 用 judge 评估 affected + injected
  
总量：4 攻击(C2-C5) × 3 防御 × 3 VLM × 7 图片 = 252 组
      （C1 数据已有，只需补防御后的数据）
```

**Step 2.3.2**: 隐蔽性分析
```
对每种攻击：
  - 自动度量：PSNR, SSIM, LPIPS
  - 人工评估：10 张图盲测（看出是否被修改？）
```

### 验收标准
- [ ] 5×4 矩阵所有格有数据
- [ ] 隐蔽性对比表完成

---

## V2.4P — 报告 + PPT + Demo 更新

### 交付物
- [ ] `docs/experiment_report.md` 添加 v2.0 章节（攻击分类、防御方案、攻防矩阵、结论）
- [ ] PPT 更新（`docs/reports/visinject_experiment_report.pptx`）
- [ ] Demo 更新：添加 5 种攻击展示 + 防御效果对比 tab
- [ ] README.md / README_zh.md 更新为 v2.5 内容

---

## V2.5P — 封版 + 提交

### 执行步骤
```
1. 代码清理 + .gitignore 更新
2. 全部 commit
3. git tag v2.5
4. git branch v2.5-release
5. HuggingFace Dataset + Space 同步
6. 课程提交
```

---

## 进度跟踪规则

1. **每完成一个 Step**：在对应的 `[ ]` 改为 `[x]`，附注完成日期
2. **每完成一个版本**：从"当前开发"移到"已完成"，打 git tag
3. **遇到阻塞**：在 Step 下方加 `⚠️ BLOCKED: <原因>`
4. **决策变更**：在 Step 下方加 `📝 DECISION: <变更内容和原因>`
5. **此文档由 Claude 和用户共同维护**，每次会话开始时读取最新进度
