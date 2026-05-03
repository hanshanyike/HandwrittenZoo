# Multi-Head Attention (MHA)

## 算法简介
多头注意力（Multi-Head Attention, MHA）是 Transformer 架构的核心组件，最早由 Vaswani 等人在 2017 年提出。它通过将查询（Query）、键（Key）、值（Value）投影到多个子空间，实现并行关注不同位置的信息，从而显著增强模型的表达能力。

## 核心思想
单一注意力头只能捕捉一种语义关系，而多头机制将输入切分为 $h$ 个独立的注意力头，每个头学习不同的子空间表示。最终将所有头的输出拼接并投影回原始维度，实现“多视角”信息融合。

## 数学公式
对于输入 $\mathbf{X} \in \mathbb{R}^{n \times d_{model}}$，先进行线性投影：

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V
$$

然后将 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ 按头切分，对第 $i$ 个头计算缩放点积注意力：

$$
\text{head}_i = \text{Attention}(\mathbf{Q}_i, \mathbf{K}_i, \mathbf{V}_i) = \text{softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_i^T}{\sqrt{d_k}}\right) \mathbf{V}_i
$$

最后拼接并投影：

$$
\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O
$$

其中 $d_k = d_{model} / h$ 为每个头的维度。

## 时间/空间复杂度
- **时间复杂度**: $O(b \cdot n^2 \cdot d_{model})$，其中 $b$ 为 batch size，$n$ 为序列长度。
- **空间复杂度**: $O(b \cdot n^2 \cdot h)$，主要来自于显式存储的注意力分数矩阵。
- **与替代方案对比**: MHA 表达能力最强，但推理时 KV Cache 占用最大（$2 \cdot n_h \cdot d_h \cdot l$），是后续 MQA/GQA/MLA 等压缩方案的起点。

## 面试高频考点
1. **为什么需要除以 $\sqrt{d_k}$？**
   **答案**: 当 $d_k$ 较大时，$QK^T$ 的点积值域会变大，导致 softmax 进入梯度极小的饱和区。缩放因子 $\sqrt{d_k}$ 将方差稳定到约 1，保证梯度流动。

2. **MHA 中 Q、K、V 有什么区别？**
   **答案**: Q 是当前 token 的“查询”，K 是所有 token 的“索引键”，V 是所有 token 的“内容值”。注意力权重由 Q 与 K 的相似度决定，再用权重对 V 加权求和。

3. **多头注意力的优势是什么？**
   **答案**: (1) 不同头可学习不同的语义关系（如语法依赖、指代消解）；(2) 并行计算效率高；(3) 增加模型容量而不显著增加参数量（相比单个大矩阵）。

4. **为什么 Transformer 使用自注意力而不是 RNN？**
   **答案**: 自注意力可并行计算整个序列，时间复杂度为 $O(1)$ 层内并行度；RNN 为 $O(n)$ 顺序依赖。同时自注意力的长距离依赖路径长度为 $O(1)$，而 RNN 为 $O(n)$。

## 代码解析
- `ScaledDotProductAttention`: 实现单头的缩放点积注意力，包含 mask 支持。
- `MultiHeadAttention`:
  - `w_q`, `w_k`, `w_v`: 分别将输入投影到 Q、K、V。
  - `view + transpose`: 将 `(batch, seq, d_model)` 重塑为 `(batch, heads, seq, head_dim)`，以便并行计算多头。
  - `w_o`: 将拼接后的多头输出映射回 `d_model`。

## 参考资料
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (NeurIPS 2017)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [PyTorch MultiheadAttention 文档](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
