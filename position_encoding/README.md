# Positional Encoding (位置编码)

## 类别概述

Transformer 的自注意力机制具有置换等变性——打乱输入 token 的顺序，输出也会相应打乱。为了让模型理解序列中 token 的先后顺序，必须显式注入位置信息。**位置编码**就是解决这一问题的关键技术。

本目录实现了 4 种主流的位置编码方案，覆盖从 Transformer 诞生到现代大语言模型（LLM）的演进路径：

| 方案 | 类型 | 核心机制 | 代表模型 | 外推能力 |
|------|------|----------|----------|----------|
| [Sinusoidal](sinusoidal/sinusoidal.md) | 绝对位置编码 | 正弦/余弦函数生成确定性位置向量 | Transformer (原版) | 弱 |
| [Learnable](learnable/learnable.md) | 绝对位置编码 | 每个位置是可训练参数 | BERT, GPT-2, ViT | 无 |
| [RoPE](rope/rope.md) | 相对位置编码 | 旋转矩阵变换 Q/K，内积仅依赖相对距离 | LLaMA, Mistral, Qwen | 强（配合 NTK） |
| [ALiBi](alibi/alibi.md) | 相对位置编码 | 在 attention score 上添加距离相关的线性负偏置 | MPT, BLOOM, Baichuan | 很强 |

## 演进脉络

```
2017  Transformer (Sinusoidal) ──► 手工设计，确定性函数
   │
   ▼
2018  BERT / GPT (Learnable) ────► 数据驱动，更灵活但无外推
   │
   ▼
2021  RoPE / ALiBi ──────────────► 显式建模相对位置，现代 LLM 标配
   │
   ▼
2023+ NTK-RoPE / YaRN / Dynamic NTK ──► 长上下文外推技术
```

## 面试聚焦点

### 1. 绝对 vs 相对位置编码

- **绝对编码**：每个位置有唯一的编码向量，模型直接感知绝对位置。代表：Sinusoidal、Learnable。
- **相对编码**：模型感知的是两个 token 之间的距离，而非绝对坐标。代表：RoPE、ALiBi。
- **面试常问**：为什么现代 LLM 普遍采用相对位置编码？
  - 答案：相对位置编码具有更好的长度外推能力，且更符合自然语言的平移不变性（句子的整体偏移不应改变语义）。

### 2. RoPE 为什么是面试必考？

RoPE 是 LLaMA 系列模型的核心组件，面试中几乎必考：
- **数学原理**：旋转矩阵、复数乘法、相对位置不变性。
- **工程实现**：`precompute_freqs_cis`、`apply_rotary_emb`、`rotate_half`。
- **长文本优化**：NTK-aware 缩放、Dynamic NTK、YaRN。
- **高频问题**：为什么只旋转 Q/K 不旋转 V？base 值的作用？

### 3. ALiBi 的外推优势

ALiBi 以"Train Short, Test Long"著称，是外推能力考点：
- **核心思想**：不修改嵌入，直接在 attention score 上加距离偏置。
- **斜率设计**：等比数列覆盖多尺度位置模式。
- **高频问题**：ALiBi 与 RoPE 的本质区别？为什么外推能力强？

### 4. 正弦编码的"遗产"

尽管现代 LLM 很少直接使用正弦编码，但其设计思想仍值得学习：
- **多尺度频率**：波长覆盖多个数量级，是 RoPE 频率设计的灵感来源。
- **相对位置线性**：三角和角公式体现的数学美感。
- **高频问题**：正弦编码的公式推导？为什么用 10000 作为底数？

### 5. 可学习嵌入的局限

- **无法外推**：超出 max_len 的位置没有参数。
- **参数量**：$\text{max_len} \cdot d_{\text{model}}$，对于超长上下文不友好。
- **补救方法**：Positional Interpolation、动态扩展等。

## 文件说明

```
position_encoding/
├── sinusoidal.py      # 正弦位置编码实现
├── sinusoidal.md      # 正弦位置编码文档
├── rope.py            # RoPE 实现（LLaMA 风格）
├── rope.md            # RoPE 文档
├── alibi.py           # ALiBi 实现
├── alibi.md           # ALiBi 文档
├── learnable.py       # 可学习位置嵌入实现
├── learnable.md       # 可学习位置嵌入文档
└── README.md          # 本文件
```

## 快速对比

| 特性 | Sinusoidal | Learnable | RoPE | ALiBi |
|------|------------|-----------|------|-------|
| 可训练参数 | 无 | max_len * d_model | 无 | 无 |
| 相对位置建模 | 隐式（线性性质） | 无 | 显式 | 显式 |
| 修改对象 | 输入嵌入 | 输入嵌入 | Q/K 向量 | Attention score |
| 长度外推 | 弱 | 无 | 强（配合 NTK） | 很强 |
| 实现复杂度 | 低 | 低 | 中 | 低 |
| 现代采用度 | 低 | 中 | 很高 | 高 |

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Train Short, Test Long: Attention with Linear Biases](https://arxiv.org/abs/2108.12409)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
