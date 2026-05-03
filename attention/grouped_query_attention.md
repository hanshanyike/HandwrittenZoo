# Grouped-Query Attention (GQA)

## 算法简介
分组查询注意力（Grouped-Query Attention, GQA）是 MHA 与 MQA 的折中方案，由 LLaMA-2 70B 和 Mistral 7B 等模型推广。它将查询头分成若干组，每组共享一组 Key 和 Value，既压缩了 KV Cache，又避免了 MQA 表达能力下降过多的问题。

## 核心思想
MHA 的 KV Cache 太大，MQA 的表达能力可能受损。GQA 引入“组”的概念：若 $n_h$ 个 query heads 分成 $n_g$ 组，则每组共享 1 个 K/V 头。当 $n_g = n_h$ 时退化为 MHA，当 $n_g = 1$ 时退化为 MQA。

## 数学公式
Query 仍按头切分：

$$
\mathbf{Q}_i = \mathbf{X}\mathbf{W}^{Q}_i \quad (i = 1, \dots, n_h)
$$

Key / Value 按组切分，第 $g$ 组服务 $n_h / n_g$ 个 query heads：

$$
\mathbf{K}_g = \mathbf{X}\mathbf{W}^{K}_g, \quad \mathbf{V}_g = \mathbf{X}\mathbf{W}^{V}_g \quad (g = 1, \dots, n_g)
$$

每组内的 query heads 共享同一组 K/V：

$$
\text{head}_i = \text{softmax}\left(\frac{\mathbf{Q}_i \mathbf{K}_{g(i)}^T}{\sqrt{d_k}}\right) \mathbf{V}_{g(i)}
$$

其中 $g(i) = \lfloor (i-1) / (n_h / n_g) \rfloor$。

## 时间/空间复杂度
- **时间复杂度**: $O(b \cdot n^2 \cdot d_{model})$，与 MHA 相同。
- **空间复杂度（推理 KV Cache）**: $O(b \cdot n \cdot n_g \cdot d_h)$，介于 MHA（$n_g = n_h$）和 MQA（$n_g = 1$）之间。
- **与替代方案对比**:
  - MHA: $2 n_h d_h$ 每 token 每层。
  - GQA: $2 n_g d_h$ 每 token 每层，通常 $n_g = n_h / 4$ 或 $n_h / 8$。
  - MQA: $2 d_h$ 每 token 每层。

## 面试高频考点
1. **GQA 相比 MQA 的优势是什么？**
   **答案**: GQA 在压缩 KV Cache 的同时保留了多组 K/V，使不同 query head 组仍能关注不同的语义子空间，通常能在推理速度和模型质量之间取得更好的平衡。

2. **GQA 在什么情况下退化为 MHA 或 MQA？**
   **答案**: 当 `num_kv_heads == num_heads` 时，GQA 等价于 MHA；当 `num_kv_heads == 1` 时，GQA 等价于 MQA。

3. **GQA 的 KV Cache 是 MHA 的多少倍？**
   **答案**: 约为 $n_g / n_h$。例如 LLaMA-2 70B 使用 8 组 K/V 头对应 64 个 Q 头，则 KV Cache 为 MHA 的 $1/8$。

4. **为什么现代大模型（如 LLaMA-3、Mistral）普遍采用 GQA？**
   **答案**: 长上下文推理时 KV Cache 是主要瓶颈。GQA 在几乎不损失性能的前提下将 Cache 压缩 4~8 倍，使得更大 batch 或更长序列成为可能。

## 代码解析
- `num_queries_per_kv`: 计算每个 K/V 头需要服务多少个 Q 头。
- `w_k`, `w_v`: 输出维度为 `num_kv_heads * head_dim`，比 MHA 的 `d_model` 小。
- `repeat_interleave`: 将 K/V 的组维度复制扩展到与 Q 的 head 数对齐，便于广播计算注意力。
- 广播后 Q、K、V 的 head 维度均为 `num_heads`，可直接调用标准缩放点积逻辑。

## 参考资料
- [LLaMA 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) (Meta, 2023)
- [Mistral 7B 技术博客](https://mistral.ai/news/announcing-mistral-7b/)
- [GQA 论文: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
