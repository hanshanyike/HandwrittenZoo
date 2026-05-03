# Transformer 架构全系列

本目录系统覆盖 Transformer 及其核心变体的从零实现，是算法/大模型工程师面试的**绝对核心考点**。

---

## 文件清单

| 文件 | 内容 | 面试重点 |
|------|------|----------|
| `transformer_full.py` + `.md` | 完整 Encoder-Decoder Transformer | Attention 计算、Mask 机制、参数量估算 |
| `bert.py` + `.md` | BERT（Encoder-Only + MLM/NSP） | 双向表示、MLM 策略、预训练 vs 微调 |
| `decoder_only.py` + `.md` | GPT/LLaMA 风格 Decoder-Only | Pre-Norm、RMSNorm、RoPE、SwiGLU、GQA |

---

## 面试焦点速查

### 1. 三种架构对比

| 特性 | Encoder-Decoder (原始 Transformer) | Encoder-Only (BERT) | Decoder-Only (GPT/LLaMA) |
|------|-----------------------------------|---------------------|--------------------------|
| Attention | 双向 + Cross | 双向 | 单向（Causal） |
| Mask | Padding Mask | Padding Mask | Causal Mask |
| 典型任务 | 机器翻译 | 理解任务（分类/NER） | 生成任务（文本续写） |
| 预训练目标 | 监督/自监督 | MLM + NSP | 自回归 LM |
| 代表模型 | T5, BART | BERT, RoBERTa | GPT-4, LLaMA, Qwen |

### 2. 核心组件演进

```
原始 Transformer (2017)
    ├── LayerNorm (Post-Norm)
    ├── Sinusoidal Positional Encoding
    └── ReLU FFN

BERT (2018)
    ├── Learned Positional Embedding
    ├── GELU FFN
    └── MLM + NSP

GPT-3 (2020)
    ├── Decoder-Only
    └── Causal Mask

LLaMA (2023)
    ├── Pre-Norm
    ├── RMSNorm
    ├── RoPE
    ├── SwiGLU
    └── GQA (LLaMA-2/3)
```

### 3. 高频计算题

**Q：BERT-Base 的参数量是多少？**

- $V = 30522, d = 768, N = 12, d_{ff} = 3072$
- Embedding：$30522 \times 768 \approx 23.4$ M
- Position：$512 \times 768 \approx 0.39$ M
- Segment：$2 \times 768 \approx 0.0015$ M
- Attention（每层）：$4 \times 768^2 \approx 2.36$ M
- FFN（每层）：$2 \times 768 \times 3072 \approx 4.72$ M
- 总参数量：$\approx 23.4 + 12 \times (2.36 + 4.72) \approx 110$ M

**Q：为什么 LLaMA 使用 SwiGLU 时中间维度要乘以 2/3？**

- 标准 FFN：$2$ 个矩阵，参数量 $2 \cdot d \cdot d_{ff}$
- SwiGLU：$3$ 个矩阵，参数量 $3 \cdot d \cdot d_{hidden}$
- 令 $3 \cdot d \cdot d_{hidden} = 2 \cdot d \cdot d_{ff}$，得 $d_{hidden} = \frac{2}{3} d_{ff}$

### 4. 关键面试陷阱

1. **"Transformer 的 Attention 复杂度是 $O(n^2)$"** —— 准确说是 $O(n^2 \cdot d)$，当 $d$ 很大时不能忽略。
2. **"BERT 可以做生成"** —— BERT 是双向的，没有 Causal Mask，不能直接自回归生成。
3. **"Pre-Norm 一定比 Post-Norm 好"** —— Pre-Norm 训练更稳定，但表示能力略弱，需通过增加层数弥补。
4. **"RoPE 只编码相对位置"** —— RoPE 同时编码绝对位置（通过旋转角度）和相对位置（通过内积性质）。

---

## 学习路线建议

1. **入门**：先阅读 `transformer_full.md`，理解 Attention、Mask、残差连接的数学原理。
2. **进阶**：学习 `bert.md`，掌握 MLM 的 80/10/10 策略和双向表示的意义。
3. **前沿**：精读 `decoder_only.md`，这是 2024-2025 面试的核心战场，Pre-Norm/RMSNorm/RoPE/SwiGLU/GQA 几乎必考。
4. **实战**：运行三个 `.py` 文件的 `__main__` 自测块，观察输出 shape 和 loss 变化。

---

*持续更新中，欢迎补充更多变体（如 T5、BART、Mixtral MoE 等）。*
