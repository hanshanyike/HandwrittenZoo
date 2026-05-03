# Decoder-Only Transformer (GPT / LLaMA Style)

## 算法简介

Decoder-Only Transformer 是现代大语言模型（LLM）的主流架构，代表性模型包括 GPT 系列、LLaMA、Qwen、Baichuan 等。与原始 Encoder-Decoder Transformer 不同，它仅保留 Transformer Decoder，去掉 Cross-Attention，通过 Causal Mask 实现自回归（Autoregressive）文本生成。LLaMA 在此基础上引入 Pre-Norm、RMSNorm、RoPE 和 SwiGLU 等改进，成为当前开源模型的标杆设计。

## 核心思想

1. **Decoder-Only 架构**：仅使用 Transformer Decoder 的 Masked Self-Attention + FFN，去掉 Encoder 和 Cross-Attention。简化架构的同时保持强大的生成能力。
2. **Pre-Norm vs Post-Norm**：原始 Transformer 使用 Post-Norm（子层后归一化），LLaMA 使用 Pre-Norm（子层前归一化）。Pre-Norm 形成更干净的残差流，使深层模型（32+ 层）训练更稳定，不易出现梯度爆炸或消失。
3. **RMSNorm**：去掉 LayerNorm 的均值中心化，只保留缩放操作。参数量减半，计算更快，且在大模型上效果相当甚至更优。
4. **RoPE（旋转位置编码）**：通过旋转矩阵将位置信息注入 Q/K 向量，使得 Attention 分数天然包含相对位置信息。支持长度外推，是现代 LLM 的首选位置编码。
5. **SwiGLU**：在 FFN 中引入门控机制（SiLU 门控），表达能力显著强于标准 ReLU/GELU FFN。LLaMA 使用 SwiGLU 并调整中间维度以控制参数量。
6. **GQA（Grouped Query Attention）**：将多个 Query 头分组共享 K/V 头，在保持质量的同时大幅减少 KV Cache 内存，是现代推理优化的关键设计。

## 数学公式

### RMSNorm

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \varepsilon}} \cdot \gamma
$$

与 LayerNorm 相比，没有减去均值的操作：$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta$

### RoPE（旋转位置编码）

对于位置 $m$ 的二维向量 $(x_1, x_2)$，应用旋转矩阵：

$$
R_{\Theta, m} \cdot \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
$$

其中 $\theta_i = \text{base}^{-2i/d}$。推广到 $d$ 维时，每两个维度组成一对进行旋转。

### SwiGLU

$$
\text{SwiGLU}(x) = (W_2 \cdot (\text{SiLU}(W_1 x) \odot (W_3 x)))
$$

其中 $\text{SiLU}(x) = x \cdot \sigma(x)$，$\sigma$ 为 sigmoid 函数。

### Causal Mask

$$
M_{ij} = \begin{cases} 1 & \text{if } i \geq j \\ 0 & \text{if } i < j \end{cases}
$$

保证位置 $i$ 只能 attend 到位置 $\leq i$ 的 token。

## 时间/空间复杂度

| 项目 | 复杂度 | 说明 |
|------|--------|------|
| Self-Attention（训练） | $O(n^2 \cdot d)$ | 序列长度的平方是主要瓶颈 |
| Self-Attention（推理，无 KV Cache） | $O(n^2 \cdot d)$ | 每步重新计算所有历史 |
| Self-Attention（推理，有 KV Cache） | $O(n \cdot d)$ | 每步只需计算当前 Q 与缓存的 K/V |
| FFN | $O(n \cdot d \cdot d_{ff})$ | SwiGLU 有 3 个矩阵，但中间维度通常缩小 |
| KV Cache 内存 | $O(2 \cdot L \cdot n \cdot d)$ | $L$ 为层数，是长文本推理的关键瓶颈 |

## 面试高频考点

### Q1：Pre-Norm 和 Post-Norm 有什么区别？为什么大模型都用 Pre-Norm？

**答案**：
- **Post-Norm**：$x_{out} = \text{Norm}(x + \text{Sublayer}(x))$。残差连接在归一化之外，深层时梯度需经过 LayerNorm 的逆变换，容易出现梯度消失或爆炸。
- **Pre-Norm**：$x_{out} = x + \text{Sublayer}(\text{Norm}(x))$。归一化在子层之前，残差连接形成"干净"的直接通路，梯度可稳定传播，支持训练上百层的模型。
- 代价：Pre-Norm 的表示能力略弱于 Post-Norm，但通过增加层数可弥补。

### Q2：RMSNorm 为什么比 LayerNorm 更适合大模型？

**答案**：
1. **参数量减半**：RMSNorm 只有可学习的缩放参数 $\gamma$，没有偏移参数 $\beta$。
2. **计算更快**：省去均值计算，只需计算均方根。
3. **效果相当**：实验表明在大模型上 RMSNorm 与 LayerNorm 效果相当甚至更优，因为深层网络中均值中心化带来的收益递减。
4. **与 Pre-Norm 配合**：Pre-Norm 架构下输入分布相对稳定，RMSNorm 的简化设计已足够。

### Q3：RoPE 相比传统位置编码的优势是什么？

**答案**：
1. **相对位置显式表达**：RoPE 通过旋转矩阵编码，使得 $q_m^T k_n$ 的结果天然包含 $(m-n)$ 的相对位置信息，无需额外学习。
2. **长度外推性**：训练时的旋转角度可外推到更长的未见序列，支持动态扩展上下文（如从 4K 扩展到 128K）。
3. **与 Attention 融合**：位置信息直接编码在 Q/K 中，不修改输入嵌入，保持模型结构的简洁性。

### Q4：SwiGLU 比标准 FFN 强在哪里？为什么要调整中间维度？

**答案**：
- **门控机制**：SwiGLU 引入 SiLU 门控（$\text{SiLU}(x) = x \cdot \sigma(x)$），可根据输入动态调节信息流，表达能力更强。
- **参数量控制**：SwiGLU 有 3 个权重矩阵（$W_1, W_2, W_3$），而标准 FFN 只有 2 个。为保持总参数量相近，LLaMA 将中间维度调整为 $\frac{2}{3} d_{ff}$，使得总参数量约为 $3 \times d \times \frac{2}{3}d_{ff} = 2 d \cdot d_{ff}$，与标准 FFN 相同。

### Q5：GQA（Grouped Query Attention）解决了什么问题？

**答案**：
- **问题**：标准 MHA 中每个头有独立的 K/V，推理时 KV Cache 内存占用为 $2 \cdot L \cdot n \cdot d$，随头数线性增长，成为长文本推理的瓶颈。
- **MQA（Multi-Query Attention）**：所有头共享同一组 K/V，内存最小但质量下降明显。
- **GQA**：折中方案，将 $n$ 个 Q 头分为 $g$ 组，每组共享一对 K/V。例如 LLaMA-2 70B 使用 8 组，内存减少为 $1/8$，同时保持接近 MHA 的质量。

## 代码解析

### RMSNorm

```python
rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
x_norm = x / rms
return self.weight * x_norm
```

去掉均值计算，直接对输入的均方根进行归一化，再乘以可学习缩放参数。

### RoPE 旋转操作

```python
x1, x2 = x[..., ::2], x[..., 1::2]
x_rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
return x * cos + x_rotated * sin
```

将相邻两个维度视为复数平面上的坐标，应用旋转矩阵。`cos` 和 `sin` 根据位置预计算，实现绝对位置编码与相对位置信息的统一。

### Pre-Norm 残差连接

```python
x = x + self.attn(self.norm1(x), rope, mask)
x = x + self.ffn(self.norm2(x))
```

先对输入做 RMSNorm，再进入 Attention/FFN 子层，最后与原始输入相加。形成干净的残差流，是 LLaMA 训练稳定的关键。

## 参考资料

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) — Touvron et al., 2023
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Su et al., 2021
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — Zhang & Sennrich, 2019
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer, 2020
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
