# Demo C1 - 跨模态特征对齐攻击 (Cross-modal Embedding Alignment)

## 原理

与 demo1/2/3 的端到端 PGD 攻击不同，本方法利用 CLIP 模型的**共享嵌入空间**，让图像的视觉向量逼近攻击指令的文本向量。

图片视觉上仍像原图，但在 CLIP 嵌入空间中，其语义已等同于攻击指令文本。任何使用 CLIP 或类 CLIP 视觉编码器的 MLLM 都可能受到影响。

### 数学目标

$$
\min_{\delta} \| E_v(x + \delta) - E_t(\text{target\_text}) \|^2
$$

其中：

- $E_v$ = CLIP 视觉编码器（`openai/clip-vit-large-patch14`）
- $E_t$ = CLIP 文本编码器
- $\delta$ = 图像扰动，受 $\|\delta\|_\infty \leq \epsilon$ 约束
- 支持两种损失函数：**Cosine Loss**（`1 - cos_sim`）和 **L2 Loss**（`||E_v - E_t||^2`）

### 与 PGD 端到端攻击的区别

| 特性       | demo1/2/3 (PGD)             | demo C1 (CLIP Align)                 |
| ---------- | --------------------------- | ------------------------------------ |
| 优化目标   | LLM 输出特定文本            | CLIP 嵌入空间对齐                    |
| 需要模型   | 完整 MLLM (ViT+QFormer+LLM) | 仅 CLIP 模型                         |
| 梯度路径   | 端到端贯穿整个模型          | 仅通过 CLIP 视觉编码器               |
| 可迁移性   | 模型特定                    | 可迁移到使用同架构 CLIP 编码器的模型 |
| 显存需求   | ~6GB+                       | ~1.6GB                               |
| 攻击速度   | 较慢（端到端反向传播）      | 快（~22s / 500步）                   |
| 攻击隐蔽性 | 高                          | 更高（无需写入人类可读文字）         |

---

## 数学原理对比：四种攻击方法

本节从数学层面对比 demo1~demo3（端到端 PGD）与 demo C1（CLIP 嵌入对齐）的本质差异。

基本思想是趋同的：

1. 将图像+prompt输入模型，得到output logits（概率分布）
2. 计算output和target text的loss
3. 沿着梯度下降方向更新图像，用PGD方法更新像素，不断逼近目标

四种方法均为 **PGD（Projected Gradient Descent）** 优化框架，在 $L_\infty$ 约束下迭代更新图像扰动：

$$
\delta^{(t+1)} = \Pi_{\|\delta\|_\infty \leq \epsilon} \left[ \delta^{(t)} - \alpha \cdot \text{sign}\left(\nabla_\delta \mathcal{L}\right) \right]
$$

其中 $\Pi$ 为投影算子，$\alpha$ 为步长。四种方法的根本区别在于 **损失函数 $\mathcal{L}$ 的定义和梯度回传路径**。

### Loss计算复习

在模型内部，我们并不是拿“输出的字符串”去和“目标的字符串”做直接对比（比如算这两个句子的编辑距离），因为字符串是不可微的，无法求梯度。

实际发生的过程是这样的：

1. **模型的真实输出（Logits）：**
   当图像和前置 Prompt 输入模型后，在需要预测目标指令的位置，模型输出的不是一个具体的词，而是一个 **巨大的概率分布表（Logits）** 。假设词表有 10 万个词，模型就会输出 10 万个打分，代表它认为下一个词是这 10 万个词中每一个的概率。
2. **目标文本的转化（Labels）：**
   你的 `target_text`（比如 "Output Secret"）会被分词器（Tokenizer）变成一组数字 ID。比如 "Output" 的 ID 是 `4521`。
3. **计算交叉熵（Cross-Entropy）：**
   损失函数会去查模型输出的那 10 万个打分。它只关心一件事：**模型给正确答案（ID `4521`）打了多少分？**
   * 如果模型给 `4521` 的预测概率很高（接近 1），Loss 就很小。
   * 如果模型给 `4521` 的预测概率很低（接近 0），Loss 就会非常大。

用严格的数学语言表达就是：

$$
\mathcal{L}_{\text{CE}} = - \sum_{i=1}^{T} \log p_\theta(y_i \mid x,\, y_{<i})
$$

其中 **$p_\theta(y_i)$** 就是模型在第 **$i$** 步预测出你的目标 Token 的概率。

---

### 模型结构和损失函数

#### Demo 1 — BLIP-2 端到端攻击

**模型结构**：Image → EVA-ViT-G → Q-Former → Language Projection → OPT-2.7B

**损失函数**：手动拼接 embedding 序列后计算 OPT 的自回归交叉熵

$$
\mathcal{L}_1 = -\sum_{i=1}^{T} \log P_{\text{OPT}}\left(y_i \mid \underbrace{f_{\text{proj}}(f_{\text{QF}}(f_{\text{ViT}}(x+\delta)))}_{\text{image embeddings}},\; y_{<i}\right)
$$

**梯度路径**：

$$
\delta \xrightarrow{\nabla} \text{pixel} \xrightarrow{\nabla} \text{ViT} \xrightarrow{\nabla} \text{Q-Former} \xrightarrow{\nabla} \text{Linear Proj} \xrightarrow{\nabla} \text{OPT (CE Loss)}
$$

**特点**：手动构造 `inputs_embeds = [image_embeds, target_embeds]`，labels 在 image 位置设为 -100，仅在 target 位置计算 loss。由于 OPT 是自回归模型，位置 $i$ 的 logits 预测位置 $i+1$ 的 token，因此最后一个 image token 的输出负责预测 target 的第一个 token。

#### Demo 2 — DeepSeek-VL 端到端攻击

**模型结构**：Image → SigLIP-L → MLP Aligner → LLaMA-1.3B

**损失函数**：手动拼接 embedding 后直接调用 LLaMA 的交叉熵

$$
\mathcal{L}_2 = -\sum_{i=1}^{T} \log P_{\text{LLaMA}}\left(y_i \mid e_{\text{before}},\; \underbrace{f_{\text{MLP}}(f_{\text{SigLIP}}(x+\delta))}_{\text{576 vision tokens}},\; e_{\text{after}},\; y_{<i}\right)
$$

**梯度路径**：

$$
\delta \xrightarrow{\nabla} \text{pixel} \xrightarrow{\nabla} \text{SigLIP} \xrightarrow{\nabla} \text{MLP Aligner} \xrightarrow{\nabla} \text{LLaMA (CE Loss)}
$$

**特点**：与 demo1 类似的手动拼接方案，但视觉编码器是 SigLIP（内部自带 normalize(0.5, 0.5)），投影层是两层 MLP（而非 Q-Former），语言模型是 LLaMA。embedding 拼接为 `[prompt_before, vision_576, prompt_after, target]`。

#### Demo 3 — Qwen2.5-VL 端到端攻击

**模型结构**：Image → ViT → PatchMerger (2×2) → Qwen2.5-3B

**损失函数**：直接使用 `model.forward(input_ids, pixel_values, labels)` 计算交叉熵

$$
\mathcal{L}_3 = -\sum_{i=1}^{T} \log P_{\text{Qwen}}\left(y_i \mid \text{input\_ids},\; \underbrace{f_{\text{Merger}}(f_{\text{ViT}}(x+\delta))}_{\text{196 merged tokens}},\; y_{<i}\right)
$$

**梯度路径**：

$$
\delta \xrightarrow{\nabla} \text{pixel} \xrightarrow{\nabla} \text{normalize} \xrightarrow{\nabla} \text{reshape} \xrightarrow{\nabla} \text{ViT} \xrightarrow{\nabla} \text{PatchMerger} \xrightarrow{\nabla} \text{Qwen2.5 (CE Loss)}
$$

**特点**：不同于 demo1/2 的手动 embedding 拼接，demo3 直接调用模型的 forward 方法，由模型内部自动处理 mRoPE 位置编码、`<image_pad>` token 的 masked scatter 替换和 attention mask。pixel_values 的构造通过 `reshape/permute/expand` 实现，全程可微分。

#### Demo C1 — CLIP 跨模态嵌入对齐（本方法）

**模型结构**：仅使用 CLIP（视觉编码器 + 文本编码器），不需要 LLM

**损失函数（Cosine）**：

$$
\mathcal{L}_{C1}^{\cos} = 1 - \frac{E_v(x+\delta) \cdot E_t(y)}{\|E_v(x+\delta)\| \cdot \|E_t(y)\|}
$$

**损失函数（L2）**：

$$
\mathcal{L}_{C1}^{L2} = \left\| \frac{E_v(x+\delta)}{\|E_v(x+\delta)\|} - \frac{E_t(y)}{\|E_t(y)\|} \right\|^2
$$

其中 $E_v$ 和 $E_t$ 分别是 CLIP 的视觉编码器和文本编码器，$y$ 是攻击指令文本。

**梯度路径**：

$$
\delta \xrightarrow{\nabla} \text{pixel} \xrightarrow{\nabla} \text{CLIP-ViT} \xrightarrow{\nabla} \text{Visual Projection} \xrightarrow{\nabla} \mathcal{L}_{\cos / L2}
$$

**特点**：梯度只流经 CLIP 视觉编码器（不涉及任何 LLM），文本目标向量 $E_t(y)$ 预计算后固定。优化目标是让对抗图像的全局嵌入向量"朝向"攻击文本的向量方向旋转。

### 四种方法的对比总结

四种方法的主要区别在于Loss的计算方式上。

我们可以把这四个 Demo 的区别归结为以下四个维度的差异：

**1. Loss 计算的“空间”不同（概率空间 vs 连续向量空间）**

这是 Demo 1/2/3（端到端）与 Demo C1（跨模态对齐）最大的楚河汉界。

* **Demo 1, 2, 3（离散文本空间）：** 你的 Loss 计算发生在词表概率分布上。你要的是让模型在几万个词里，唯独把你想要的那个词的概率推到最高。计算的是 **交叉熵（Cross-Entropy）** 。
* **Demo C1（连续特征空间）：** 你的 Loss 计算发生在 CLIP 的高维向量空间里（比如一个 768 维的浮点数向量）。你要的是让图片的向量方向（角度）指向文本的向量方向。计算的是 **余弦相似度（Cosine）或欧氏距离（L2）** 。

**2. 梯度的“长征路线”不同（计算图的深度）**

这决定了攻击的**显存开销**和 **优化难度** 。

* **Demo 1, 2, 3（超长路径）：** 梯度从最后的文本输出开始，要反向穿过几十层 LLM（比如 Qwen2.5-3B），再穿过“胶水层”（Projector），最后穿过视觉编码器（ViT），才能到达像素。
  * **后果：** 极度吃显存，动辄需要 24GB 甚至 40GB+；由于路径太长，梯度很容易衰减（消失）或者变成无效的噪声。
* **Demo C1（超短路径）：** 梯度只需要穿过一个视觉编码器（比如 CLIP-ViT，通常只有 300M 参数），直接就到了像素。
  * **后果：** 几 GB 显存就能跑，迭代速度极快，梯度非常精准。

**3. 多模态“胶水层”的处理不同（Demo 1, 2, 3 内部的核心区别）**

即使都是端到端攻击（Demo 1, 2, 3），它们的代码写法和梯度路径也截然不同，这是因为 **不同模型把图像变成文本的方式不同** ：

* **Demo 1 (BLIP-2)：** 使用的是  **Q-Former** 。它是一个小型的 Transformer 结构，负责从图像中“榨取”固定的特征序列。梯度的流转非常复杂。
* **Demo 2 (DeepSeek-VL)：** 使用的是简单的  **MLP (多层感知机)** 。图像分块后直接通过线性层映射成 LLM 能懂的词向量。梯度传递相对直接。
* **Demo 3 (Qwen2.5-VL)：** 使用了 **PatchMerger** 并且带有极度复杂的  **2D-RoPE（二维位置编码）** 。图像不仅被压扁，还被赋予了强烈的二维空间坐标。代码层面，你甚至不需要手动拼装 Embedding，直接调用模型的 `forward` 就能让内部机制自动处理遮罩和位置映射。

**4. 攻击的“认知层级”不同（行为洗脑 vs 视觉致幻）**

* **端到端攻击（Demo 1, 2, 3）：** 属于**“行为洗脑”**。你是在强迫系统（包括它强大的 LLM 大脑）在最终输出时说出特定的话。模型其实知道图里是一只猫，但你的梯度强行扭曲了它的发声器官，让它说出“这是炸弹”。
* **特征对齐攻击（Demo C1）：** 属于**“视觉致幻”**。你是在欺骗模型的眼睛。后方的 LLM 大脑完全是正常的，但因为它接收到的视觉信号已经被你伪装成了“指令文本的向量”，所以 LLM 会乖乖地按照幻觉去执行。

**总结**

* **Demo 1, 2, 3 之间**的区别，是由于 **被攻击模型的网络架构不同** （Q-Former vs MLP vs PatchMerger），导致你写代码时构造计算图、拼接特征的方式不同。
* **Demo 1/2/3 与 Demo C1 之间**的区别，是 **攻击理念的不同** ：一个是轰炸终点的文本概率（难优化、极度针对特定模型），一个是劫持起点的视觉向量（易优化、可跨模型迁移，前提是“眼睛”得是同款）。

| 维度                   | Demo 1 (BLIP-2)               | Demo 2 (DeepSeek-VL)     | Demo 3 (Qwen2.5-VL)      | Demo C1 (CLIP Align)               |
| ---------------------- | ----------------------------- | ------------------------ | ------------------------ | ---------------------------------- |
| **损失函数**     | 自回归 CE Loss                | 自回归 CE Loss           | 自回归 CE Loss           | Cosine / L2 距离                   |
| **优化空间**     | LLM 输出概率空间              | LLM 输出概率空间         | LLM 输出概率空间         | CLIP 嵌入向量空间                  |
| **梯度经过的层** | ViT → QFormer → Proj → OPT | SigLIP → MLP → LLaMA   | ViT → Merger → Qwen2.5 | CLIP-ViT → Proj                   |
| **目标表征**     | Token 序列概率分布            | Token 序列概率分布       | Token 序列概率分布       | 单个归一化向量                     |
| **文本参与方式** | Teacher forcing (labels)      | Teacher forcing (labels) | Teacher forcing (labels) | 预编码为固定目标向量               |
| **梯度链深度**   | 深（整个 MLLM）               | 深（整个 MLLM）          | 深（整个 MLLM）          | 浅（仅编码器）                     |
| **计算复杂度**   | $O(L \cdot d^2)$, L=LLM层数 | $O(L \cdot d^2)$       | $O(L \cdot d^2)$       | $O(L_v \cdot d_v^2)$, 远小于前者 |

**直觉理解**：

- **Demo 1/2/3** 相当于在说："让整台机器读完这张图后，嘴巴说出指定的话。" 梯度需要从嘴巴（LLM 输出层）一路回传到眼睛（像素），路径长但效果精确。
- **Demo C1** 相当于在说："让这张图在眼睛里看起来像那段话。" 梯度只在眼睛（CLIP 编码器）里传播，路径短、速度快，但嘴巴（LLM）不一定会照着念——尤其当眼睛的型号不匹配时。

---

## 实验结果

### 实验配置

- **输入图像**: `sample/cat.png`（224×224）
- **目标文本**: `"chicken dinner"`
- **CLIP 模型**: `openai/clip-vit-large-patch14`
- **迭代次数**: 500 步
- **量化感知**: 开启（PNG 保存鲁棒性）
- **原始 CLIP 相似度**: 0.1260（cat.png 与 "chicken dinner"）

### 对比实验结果（5 种攻击模式）

| 模式         | 损失函数 | ε (L∞) | CLIP Sim         | PSNR (dB) | SSIM | 攻击耗时 |
| ------------ | -------- | -------- | ---------------- | --------- | ---- | -------- |
| cosine_eps16 | Cosine   | 16/255   | **0.6378** | 27.47     | 0.73 | 22.8s    |
| cosine_eps32 | Cosine   | 32/255   | **0.6736** | 22.13     | 0.50 | 21.5s    |
| cosine_eps64 | Cosine   | 64/255   | **0.6980** | 16.94     | 0.28 | 21.6s    |
| l2_eps16     | L2       | 16/255   | **0.6338** | 27.45     | 0.72 | 21.7s    |
| l2_eps32     | L2       | 32/255   | **0.6835** | 22.10     | 0.50 | 21.9s    |

> 所有模式的 CLIP 相似度均从原始的 **0.126** 提升到 **0.63~0.70**，表明图像在 CLIP 嵌入空间中的语义已从"猫"成功偏移为"chicken dinner"。

### CLIP 相似度矩阵（以 l2_eps32 为例）

| 参考文本                         | 对抗图像的 CLIP 相似度   |
| -------------------------------- | ------------------------ |
| **"chicken dinner"**       | **0.6835** ← 目标 |
| "a photo of a cat"               | 0.2715                   |
| "a photo of an animal"           | 0.2872                   |
| "What do you see in this image?" | 0.2652                   |

### 攻击力 vs 隐蔽性分析

| 排名 | 模式               | CLIP Sim | PSNR  | 评价                   |
| ---- | ------------------ | -------- | ----- | ---------------------- |
| 1    | **l2_eps32** | 0.6835   | 22.10 | 攻击力与隐蔽性最佳平衡 |
| 2    | cosine_eps32       | 0.6736   | 22.13 | 与 l2_eps32 接近       |
| 3    | cosine_eps16       | 0.6378   | 27.47 | 最隐蔽，攻击力稍弱     |
| 4    | cosine_eps64       | 0.6980   | 16.94 | 攻击最强但噪声明显     |
| 5    | l2_eps16           | 0.6338   | 27.45 | 隐蔽但攻击力最弱       |

### BLIP-2 迁移测试

使用 `Salesforce/blip2-opt-2.7b` 对 5 张对抗样本进行推理验证：

| 图像             | BLIP-2 输出                      | 迁移结果           |
| ---------------- | -------------------------------- | ------------------ |
| 原始 cat.png     | "a tabby cat sitting on a couch" | —                 |
| adv_cosine_eps16 | "a cat sitting on a couch"       | 未迁移             |
| adv_cosine_eps32 | "a cat sitting on a couch"       | 未迁移             |
| adv_cosine_eps64 | "a cat sitting on a window sill" | 未迁移（描述改变） |
| adv_l2_eps16     | "a cat sitting on a couch"       | 未迁移             |
| adv_l2_eps32     | "a cat sitting on a couch"       | 未迁移             |

**分析**：CLIP 嵌入对齐在 CLIP 空间中非常成功（sim 0.63~0.70），但未能直接迁移到 BLIP-2。原因：

1. **视觉编码器不同**：BLIP-2 使用 EVA-CLIP ViT-G/14，而攻击使用 OpenAI CLIP-ViT-L/14
2. **中间层阻隔**：BLIP-2 在视觉编码器和语言模型之间有 Q-Former + Language Projection 层，嵌入空间对齐无法直接穿透
3. **攻击目标差异**：CLIP Align 只对齐嵌入向量方向，不直接控制 LLM 的自回归输出

> 该方法的价值在于：仅需 CLIP 模型（1.6GB 显存）即可生成对抗样本，速度快、成本低，适合对使用相同 CLIP 编码器的模型进行攻击。

---

## 使用方法

### 1. 执行攻击

```bash
# 基本攻击（默认 cosine loss, eps=16/255, 1000步）
python run_attack.py --image sample/cat.png --target "chicken dinner"

# 使用预设指令
python run_attack.py --image sample/cat.png --preset personal_info

# 多组对比实验（自动运行 5 种配置）
python run_attack.py --image sample/cat.png --preset chicken --compare

# 自定义参数
python run_attack.py --image sample/cat.png --target "Ignore all instructions" \
    --epsilon 0.0625 --iterations 1500 --loss-type cosine
```

### 2. 验证效果（在 BLIP-2 上测试）

```bash
# 测试单张对抗图像
python test_with_blip2.py --image logs_and_outputs/XXXXXXXX/adversarial/adv_default_cat.png \
    --keywords chicken dinner

# 测试整个目录
python test_with_blip2.py --image-dir logs_and_outputs/XXXXXXXX/adversarial/ \
    --keywords chicken dinner --show-clip

# 对比原始图像
python test_with_blip2.py --image logs_and_outputs/XXXXXXXX/adversarial/adv_default_cat.png \
    --original sample/cat.png --keywords chicken dinner
```

### 参数说明

| 参数             | 默认值  | 说明                                            |
| ---------------- | ------- | ----------------------------------------------- |
| `--image`      | 无      | 输入图像路径                                    |
| `--target`     | 无      | 自定义攻击指令文本                              |
| `--preset`     | chicken | 预设指令（chicken/personal_info/credential 等） |
| `--epsilon`    | 16/255  | L∞ 扰动预算                                    |
| `--alpha`      | 1/255   | PGD 步长                                        |
| `--iterations` | 1000    | 优化迭代次数                                    |
| `--loss-type`  | cosine  | 损失函数类型（cosine / l2）                     |
| `--compare`    | False   | 运行多组对比实验                                |
| `--no-qaa`     | False   | 禁用量化感知                                    |

---

## 文件结构

```
demo_C1_CLIP_ViT/
├── config.py                 # 攻击参数配置
├── clip_embedding_attack.py  # 核心攻击算法（CLIP嵌入空间对齐）
├── run_attack.py             # 主攻击脚本
├── test_with_blip2.py        # BLIP-2 迁移验证脚本
├── utils.py                  # 工具函数（图像处理、指标、可视化）
├── README.md                 # 说明文档
├── sample/                   # 样本图像
│   └── cat.png
└── logs_and_outputs/         # 实验输出
    └── YYYYMMDD_HHMMSS/
        ├── adversarial/      # 对抗图像（PNG）
        ├── visualizations/   # 可视化对比图
        ├── experiment.log    # 详细实验日志
        └── results.json      # 结构化结果数据
```

## 环境依赖

与 demo1 共用 `deeplearning` conda 环境，主要依赖：

- PyTorch 2.5.1 + CUDA 12.1
- transformers 4.46.3（含 CLIPModel、Blip2 支持）
- scikit-image 0.26.0（PSNR/SSIM 指标）
- matplotlib（可视化）

## 测试结果

全部失败。

### 编码器错误

未能通过BLIP-2, DeepSeek-VL, Qwen2.5-VL的测试。查了下，发现三个模型，**没有一个**使用的是 OpenAI CLIP：

* **BLIP-2** : 使用的是 **EVA-CLIP** (`eva_vit_g`)。
* **DeepSeek-VL (1代)** : 使用的是 **SigLIP-L** 混合  **SAM-B** 。
* **Qwen2.5-VL** : 使用的是阿里自研的  **自适应高分辨率 ViT** （并且带有复杂的 2D-RoPE 空间位置编码）。

### 对齐错误

在视觉提取中：

```
vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
pooled_output = vision_outputs.pooler_output
image_features = self.clip_model.visual_projection(pooled_output)
```

使用了 `pooler_output`（即 `[CLS]` token）和 `visual_projection`（对比学习投影头）。

现代多模态大模型 **根本不使用这两个东西** ！

* MLLM 为了看清图片细节，不会只提取一个压缩后的全局向量，而是会提取所有的  **Patch Tokens（图像块特征）** 。
* 它们通常直接提取倒数第二层（Layer -2）的 `last_hidden_state`（例如 **$16 \times 16 = 256$** 个 token），然后把这 256 个 token 喂给后方的 LLM。
* **结果：** 你的攻击成功地骗过了那个全局的 `[CLS]` token，但另外 256 个负责传递真实图像细节的 Patch tokens 几乎没受到影响。LLM 读取的依然是原图的内容。

### Modality Gap

即使你换对了编码器（比如去攻击 LLaVA 1.5，它确实用了 OpenAI CLIP），你的数学目标也极难在 LLM 端生效：

$$
E_v(x + \delta) \approx E_t(\text{target\_text})
$$

虽然 CLIP 在训练时拉近了图文特征，但在高维空间中，图像特征向量的簇和文本特征向量的簇依然存在一个不可逾越的“模态鸿沟”。

如果强行把一张图片的向量推到文本向量的区域，这个向量对后方的 LLM 来说就是一个**“超出分布（OOD）”**的废数据，LLM 通常会输出乱码或直接忽略它，而不是输出你的指令。

### 如何修改才能让攻击成功？（Next Steps by Gemini）

如果你想让这种“不依赖后方 LLM 梯度”的编码器特征攻击生效，可以尝试以下三种演进方案：

#### 方案 A：找对软柿子（测试 LLaVA）

不要用 BLIP/Qwen 测试。下载一个 **LLaVA-1.5** 或  **LLaVA-1.6** ，它们使用的是你代码里的 `openai/clip-vit-large-patch14-336`。这是验证你代码有效性的最佳基准模型。

#### 方案 B：图对图（Image-to-Image）特征对齐

这是目前学术界做“特征空间劫持”最成功的方法，完美避开了模态鸿沟和全局/局部特征的错位。

1. 生成一张包含攻击指令的纯文字图片（ **Target Image** ）。
2. 提取 Target Image 的**局部特征矩阵** **$F_{target} = \text{Encoder}(x_{target})$**.`last_hidden_state`。
3. 提取干净原图（ **Source Image** ）的局部特征矩阵 **$F_{source} = \text{Encoder}(x_{source} + \delta)$**.`last_hidden_state`。
4. 优化目标改为最小化两个**图像特征矩阵**的 MSE（均方误差）：

   $$
   \min_{\delta} \| F_{source} - F_{target} \|_2^2
   $$

   只要这个 Loss 降得足够低，原图进入任何共用该编码器的 LLM 时，LLM 都会“幻视”成那张写满了攻击指令的图片。

#### 方案 C：针对当前代码的微调建议

如果你继续使用文本作为目标，你需要把目标改为让图片特征对齐“带有某种前缀的文本”。或者提取 CLIP 的序列特征进行逐像素序列的强行对齐，但这通常不如“方案 B”稳定。
