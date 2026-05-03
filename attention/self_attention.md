# Self-Attention (Standalone)

## 算法简介
自注意力（Self-Attention）是 Transformer 架构的基石机制。它允许序列中的每个位置都能“看到”序列中的所有其他位置，并根据它们之间的相关性动态地构建自身的上下文表示。与 RNN 的顺序处理不同，自注意力可以并行计算整个序列。

## 核心思想
对于序列中的每个 token，自注意力做三件事：
1. **提问（Query）**: 当前 token 想从其他 token 获取什么信息？
2. **索引（Key）**: 其他 token 提供了哪些“标签”供匹配？
3. **取值（Value）**: 其他 token 实际携带了什么内容？

通过计算 Query 与所有 Key 的相似度，得到注意力权重，再用权重对所有 Value 加权求和，即得到当前 token 的上下文感知表示。

## 数学公式
给定输入序列 $\mathbf{X} \in \mathbb{R}^{n \times d_{model}}$：

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}^Q, \quad \mathbf{K} = \mathbf{X}\mathbf{W}^K, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^V
$$

自注意力输出：

$$
\text{SelfAttention}(\mathbf{X}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}
$$

其中 $d_k$ 为每个头的维度。若使用多头，则对每个头独立计算后拼接。

## 时间/空间复杂度
- **时间复杂度**: $O(b \cdot n^2 \cdot d_{model})$，序列长度的平方是主要瓶颈。
- **空间复杂度**: $O(b \cdot n^2 \cdot h)$，注意力分数矩阵占主导。
- **与替代方案对比**:
  - RNN: $O(n)$ 时间复杂度（顺序），但长距离依赖路径为 $O(n)$，难以并行。
  - CNN: 局部感受野，长距离依赖需要多层堆叠。
  - Self-Attention: 全局依赖，$O(1)$ 层内距离，高度并行。

## 面试高频考点
1. **自注意力与交叉注意力的区别？**
   **答案**: 自注意力的 Q、K、V 来自同一输入；交叉注意力的 Q 来自解码器，K/V 来自编码器输出，用于对齐源序列与目标序列。

2. **为什么自注意力比 RNN 更适合长序列？**
   **答案**: (1) 任意两 token 的依赖路径长度为 $O(1)$（直接相连），RNN 为 $O(n)$；(2) 可并行计算整个序列，RNN 必须顺序执行；(3) 没有梯度消失/爆炸的递推问题。

3. **自注意力的 $O(N^2)$ 复杂度问题如何解决？**
   **答案**: 常见方案包括：稀疏注意力（Sparse Attention）、线性注意力（Linear Attention）、局部窗口注意力（Sliding Window）、FlashAttention（IO 优化不改变复杂度）、以及降低序列长度的方法（如池化、压缩）。

4. **Mask 在自注意力中的作用？**
   **答案**: Padding Mask 用于忽略填充位置；Causal Mask（Look-ahead Mask）用于 Decoder，防止当前位置看到未来的 token，保证自回归生成的正确性。

## 代码解析
- `w_q`, `w_k`, `w_v`: 将同一输入投影为 Q、K、V。
- `num_heads`: 支持单头（`num_heads=1`）和多头（`num_heads>1`）两种模式。
- `view + transpose`: 多头模式下将投影结果按头切分。
- `mask` 处理: 兼容 `(batch, seq, seq)` 和 `(batch, 1, seq, seq)` 两种形状，通过 `unsqueeze` 统一。
- 单头模式下通过 `squeeze(1)` 去除 head 维度，保持输出一致性。

## 参考资料
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (NeurIPS 2017)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Self-Attention 详解](https://zh.d2l.ai/chapter_attention-mechanisms/self-attention-and-positional-encoding.html)
