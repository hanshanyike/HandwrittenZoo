# Multi-Query Attention (MQA)

## 算法简介
多查询注意力（Multi-Query Attention, MQA）是一种为推理效率优化的注意力变体，最早在 Google 的 PaLM 和 Falcon 等模型中得到广泛应用。它在保持多头查询表达能力的同时，让所有查询头共享同一组 Key 和 Value，从而将 KV Cache 压缩到标准 MHA 的 $1/h$。

## 核心思想
标准 MHA 中，每个头都有独立的 K、V，导致推理时 KV Cache 随头数线性增长。MQA 的核心洞察是：查询需要多视角（多头），但 Key 和 Value 可以作为共享的“记忆索引”被所有查询复用。这样每次生成新 token 时，从显存加载 K/V 的带宽大幅减少。

## 数学公式
Query 仍按头切分：

$$
\mathbf{Q}_i = \mathbf{X}\mathbf{W}^{Q}_i \quad (i = 1, \dots, h)
$$

Key 和 Value 仅保留单组：

$$
\mathbf{K} = \mathbf{X}\mathbf{W}^{K}, \quad \mathbf{V} = \mathbf{X}\mathbf{W}^{V}
$$

每个 query head 与共享的 K、V 计算注意力：

$$
\text{head}_i = \text{softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}
$$

最终拼接并投影：

$$
\text{MQA}(\mathbf{X}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O
$$

## 时间/空间复杂度
- **时间复杂度**: $O(b \cdot n^2 \cdot d_{model})$，与 MHA 训练时相同。
- **空间复杂度（推理 KV Cache）**: $O(b \cdot n \cdot d_h)$，仅为 MHA 的 $1/h$。
- **与替代方案对比**:
  - MHA: KV Cache 最大，能力最强。
  - MQA: KV Cache 最小，但可能损失部分表达能力（所有头共享同一 K/V）。
  - GQA: 介于两者之间，通过分组平衡压缩比与表达能力。

## 面试高频考点
1. **MQA 与 MHA 的核心区别是什么？**
   **答案**: MHA 每个头有独立的 Q、K、V；MQA 只有 Q 是多头的，K 和 V 被所有头共享。这使得 MQA 的 KV Cache 大小为 MHA 的 $1/h$。

2. **MQA 为什么能加速推理？**
   **答案**: 自回归生成时，每步都需要从显存读取所有历史 K/V。MQA 将 K/V 头数减到 1，显存读取量大幅减少，从而缓解内存带宽瓶颈，提升吞吐。

3. **MQA 的潜在缺点是什么？**
   **答案**: 由于所有 query head 共享同一 K/V，模型可能损失捕捉多样化语义关系的能力，某些任务上表现略逊于 MHA。

4. **MQA、GQA、MHA 的 KV Cache 对比？**
   **答案**: MHA 每 token 每层缓存 $2 n_h d_h$；GQA 缓存 $2 n_g d_h$（$n_g$ 为组数）；MQA 缓存 $2 d_h$。MQA 压缩最激进，GQA 是折中方案。

## 代码解析
- `w_q`: 输出维度仍为 `d_model`，后续通过 `view` 切分为多头 Q。
- `w_k`, `w_v`: 输出维度仅为 `head_dim`，即单头 K/V。
- `unsqueeze(1)`: 为单头 K/V 增加 head 维度，使其能广播到所有 query heads。
- 广播机制让 `(batch, heads, seq, seq)` 的 scores 与 `(batch, 1, seq, head_dim)` 的 V 相乘，无需显式复制 K/V。

## 参考资料
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) (Noam Shazeer, 2019)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) (Google, 2022)
- [Falcon-40B 技术博客](https://huggingface.co/blog/falcon)
