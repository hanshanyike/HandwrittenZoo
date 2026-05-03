# Transformer (Complete Encoder-Decoder)

## 算法简介

Transformer 是 Vaswani 等人于 2017 年在论文 *Attention Is All You Need* 中提出的完全基于注意力机制的序列到序列（Seq2Seq）模型。它彻底摒弃了 RNN 和 CNN，通过自注意力（Self-Attention）实现全局依赖建模，成为现代 NLP 和大语言模型的基石架构。

## 核心思想

1. **Self-Attention 替代 RNN**：传统 RNN 需逐步计算，难以并行；Transformer 的 Attention 可一次性看到整个序列，天然支持 GPU 并行加速。
2. **Multi-Head Attention**：将 Q/K/V 投影到多个子空间分别计算注意力，让模型在不同语义维度上同时捕捉多种依赖关系。
3. **Positional Encoding**：由于 Attention 本身对位置不敏感，通过正弦/余弦函数注入绝对位置信息，且支持外推到更长序列。
4. **残差连接 + LayerNorm**：解决深层网络梯度消失问题，使模型可堆叠至数十甚至上百层。
5. **Mask 机制**：Decoder 使用 Causal Mask（下三角掩码）保证自回归生成时当前位置只能依赖已生成的 token。

## 数学公式

### Scaled Dot-Product Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

- $Q \in \mathbb{R}^{n \times d_k}$：查询矩阵
- $K \in \mathbb{R}^{m \times d_k}$：键矩阵
- $V \in \mathbb{R}^{m \times d_v}$：值矩阵
- 缩放因子 $\sqrt{d_k}$：防止当 $d_k$ 较大时点积值过大，导致 softmax 进入梯度饱和区。

### Multi-Head Attention

$$
\begin{aligned}
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O \\
\text{where } \text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{aligned}
$$

### Positional Encoding

$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\end{aligned}
$$

### Layer Normalization

$$
\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta
$$

## 时间/空间复杂度

| 项目 | 复杂度 | 说明 |
|------|--------|------|
| Self-Attention | $O(n^2 \cdot d)$ | QK^T 计算量与序列长度平方成正比，是 Transformer 处理长文本的瓶颈 |
| FFN | $O(n \cdot d \cdot d_{ff})$ | 通常 $d_{ff}=4d$，是参数量最大的部分 |
| 每层总时间 | $O(n^2 \cdot d + n \cdot d \cdot d_{ff})$ | 当 $n$ 较小时，FFN 占比更大 |
| 空间（激活） | $O(n \cdot d)$ | 每层激活值存储，训练时还需保存中间结果用于反向传播 |

**与 RNN 对比**：RNN 时间复杂度为 $O(n \cdot d^2)$，但无法并行；Transformer 的 $O(n^2 \cdot d)$ 在 GPU 上可高度并行，实际训练速度远快于 RNN。

## 面试高频考点

### Q1：为什么 Attention 要除以 $\sqrt{d_k}$？

**答案**：当 $d_k$ 较大时，$QK^T$ 中各元素是 $d_k$ 个随机变量（假设均值为 0，方差为 1）的内积，其方差约为 $d_k$，导致数值量级很大。softmax 在输入绝对值很大时会进入梯度极小的饱和区，除以 $\sqrt{d_k}$ 可将方差缩放回 1，保持梯度流动稳定。

### Q2：Decoder 中的 Causal Mask 是怎么实现的？

**答案**：Causal Mask 是一个下三角矩阵（含对角线），通过 `torch.tril` 生成。在计算注意力分数后，将 mask 为 0 的位置（即未来位置）填充极大负值（如 -1e9），使得 softmax 后这些位置的权重接近 0，从而保证当前 token 只能 attend 到自己和之前的 token，实现自回归生成。

### Q3：Transformer 的 Encoder 和 Decoder 有什么区别？

**答案**：
- **Encoder**：由 Multi-Head Self-Attention + FFN 组成，使用 Padding Mask 忽略填充位，可双向看到整个输入序列。
- **Decoder**：除了 Masked Self-Attention 和 FFN 外，还包含 Cross-Attention 层（Q 来自 Decoder，K/V 来自 Encoder 输出），使用联合 Mask（Padding Mask + Causal Mask）。

### Q4：为什么使用 LayerNorm 而不是 BatchNorm？

**答案**：
1. 序列长度可变时，BatchNorm 的均值/方差统计不稳定；LayerNorm 对每个样本独立归一化，不受 batch 内其他样本影响。
2. NLP 中 batch size 通常较小，BatchNorm 统计噪声大；LayerNorm 更适合小 batch 场景。
3. LayerNorm 与 Attention 配合更好，因为不同位置的语义差异大，不应在 batch 维度上做归一化。

### Q5：Transformer 的参数量如何估算？

**答案**：
- Embedding：$V \cdot d$
- Attention（每层）：$4 \cdot d^2$（Q/K/V/O 四个线性层）
- FFN（每层）：$2 \cdot d \cdot d_{ff}$（两个线性层）
- LayerNorm（每层）：$2 \cdot d$（gamma + beta）
- 总参数量（忽略偏置）：$\approx V \cdot d + N \cdot (4d^2 + 2d \cdot d_{ff})$

以 $d=512, d_{ff}=2048, N=6$ 为例，非嵌入参数量约为 $6 \times (4 \times 512^2 + 2 \times 512 \times 2048) \approx 18.9$ M。

## 代码解析

### MultiHeadAttention

```python
# 将 Q/K/V 投影后 reshape 为 (batch, heads, seq, d_k)
Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
```

通过 `view` + `transpose` 实现分头，不需要显式循环，充分利用 PyTorch 的并行计算。

### Mask 应用

```python
scores = scores.masked_fill(mask == 0, -1e9)
```

`masked_fill` 将 mask 为 False（0）的位置替换为 -1e9，softmax 后这些位置的注意力权重趋近于 0。

### Decoder 的 Causal Mask

```python
causal_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool()
return tgt_pad_mask & causal_mask
```

下三角矩阵保证每个位置只能看到自己和之前的位置，与 Padding Mask 做逻辑与，同时屏蔽填充位和未来位。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., NeurIPS 2017
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar
- [PyTorch Transformer Documentation](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP
