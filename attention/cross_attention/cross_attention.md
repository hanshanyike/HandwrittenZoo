# Cross-Attention (Standalone)

## 算法简介
交叉注意力（Cross-Attention）是原始 Transformer 编码器-解码器架构的关键组件。它允许解码器（Decoder）在生成每个目标 token 时，主动从编码器（Encoder）的输出中“检索”相关信息。与自注意力不同，交叉注意力的 Query 和 Key/Value 来自两个不同的序列。

## 核心思想
在序列到序列（Seq2Seq）任务中，编码器将源序列压缩为上下文表示，解码器需要利用这些表示来生成目标序列。交叉注意力通过以下方式实现：
- **Query**: 来自解码器的当前隐藏状态（代表“我想生成什么”）。
- **Key**: 来自编码器的输出（代表“源序列每个位置提供了什么索引”）。
- **Value**: 来自编码器的输出（代表“源序列每个位置的实际内容”）。

解码器通过 Q 与 K 的匹配，决定关注源序列的哪些位置，再用 V 加权求和得到上下文向量。

## 数学公式
设目标序列（Decoder）隐藏状态为 $\mathbf{H}_{\text{dec}} \in \mathbb{R}^{n_{\text{tgt}} \times d}$，源序列（Encoder）输出为 $\mathbf{H}_{\text{enc}} \in \mathbb{R}^{n_{\text{src}} \times d}$：

$$
\mathbf{Q} = \mathbf{H}_{\text{dec}} \mathbf{W}^Q, \quad \mathbf{K} = \mathbf{H}_{\text{enc}} \mathbf{W}^K, \quad \mathbf{V} = \mathbf{H}_{\text{enc}} \mathbf{W}^V
$$

交叉注意力输出：

$$
\text{CrossAttention}(\mathbf{H}_{\text{dec}}, \mathbf{H}_{\text{enc}}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}
$$

## 时间/空间复杂度
- **时间复杂度**: $O(b \cdot n_{\text{tgt}} \cdot n_{\text{src}} \cdot d_{\text{model}})$。
- **空间复杂度**: $O(b \cdot h \cdot n_{\text{tgt}} \cdot n_{\text{src}})$，注意力分数矩阵大小为 tgt_len x src_len。
- **与替代方案对比**:
  - 自注意力: Q/K/V 同源，用于建模序列内部关系。
  - 交叉注意力: Q 来自目标，K/V 来自源，用于源-目标对齐。
  - 在 Decoder-only 模型（如 GPT、LLaMA）中不使用交叉注意力，因为无编码器。

## 面试高频考点
1. **交叉注意力与自注意力的本质区别？**
   **答案**: 自注意力的 Q、K、V 来自同一输入，计算序列内部token之间的关系；交叉注意力的 Q 来自一个序列（Decoder），K/V 来自另一个序列（Encoder），用于建立两个序列之间的映射关系。

2. **交叉注意力在 Transformer 中的位置？**
   **答案**: 在原始 Transformer 的每个 Decoder Layer 中，交叉注意力位于 Masked Self-Attention 之后、Feed-Forward 之前。它接收 Encoder 的最终输出作为 K/V，接收 Decoder 的当前层输入作为 Q。

3. **为什么 Decoder-only 模型（如 GPT）不需要交叉注意力？**
   **答案**: Decoder-only 模型采用自回归生成，所有信息都包含在输入提示（prompt）中，通过因果自注意力即可建模。没有独立的编码器输出需要对齐，因此无需交叉注意力。

4. **交叉注意力中的 mask 通常是什么？**
   **答案**: 通常是 Padding Mask，用于忽略源序列中的填充位置。在 Decoder 自注意力中还有 Causal Mask，但交叉注意力本身不需要因果掩码（因为 Encoder 输出已全部可见）。

5. **T5 和 BART 中交叉注意力的作用？**
   **答案**: T5 和 BART 都是 Encoder-Decoder 架构。编码器处理输入文本，解码器通过交叉注意力从编码器输出中提取相关信息，用于生成摘要、翻译等序列到序列任务。

## 代码解析
- `w_q`: 对目标序列（query）进行投影。
- `w_k`, `w_v`: 对源序列（key/value）进行投影。
- `forward` 参数明确区分 `query`（tgt）和 `key/value`（src），体现交叉注意力的语义。
- 分头逻辑与 Self-Attention 相同，支持单头和多头模式。
- Mask 形状为 `(batch, tgt_len, src_len)`，用于屏蔽源序列中的填充位置。

## 参考资料
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (NeurIPS 2017)
- [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) (Meta, 2019)
- [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683) (Google, 2019)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
