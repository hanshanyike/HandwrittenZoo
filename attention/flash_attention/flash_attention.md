# Simplified FlashAttention (Tiling + Online Softmax)

## 算法简介
FlashAttention 是由 Dao 等人在 2022 年提出的 IO 感知精确注意力算法。它通过**分块计算（Tiling）**和**在线 Softmax（Online Softmax）**两大技术，在不改变注意力数学结果的前提下，将显存占用从 $O(N^2)$ 降低到 $O(N)$，并显著提升实际推理速度。

## 核心思想
1. **分块计算（Tiling）**: 将 Q 分成小块放入高速 SRAM，然后逐块加载 K/V 进行计算。全程不显式构造完整的 $N \times N$ 注意力矩阵。
2. **在线 Softmax**: 传统 softmax 需要整行的最大值和指数和。FlashAttention 维护两个运行变量——当前最大值 $m$ 和指数累加和 $l$，在遍历 K/V 块时逐步更新，无需等所有分数计算完毕。

## 数学公式
### 在线 Softmax 更新
对于每一行（每个 query），维护：
- $m$: 当前已见分数的最大值
- $l$: 当前已见 $\exp(\text{score} - m)$ 的累加和
- $o$: 当前已见的加权输出累加器

当处理新的 K/V 块，得到分数矩阵 $S_{ij}$ 时：

$$
m_{\text{new}} = \max(m_{\text{old}}, \max_j(S_{ij}))
$$

$$
\exp_{\text{diff}} = \exp(m_{\text{old}} - m_{\text{new}})
$$

$$
l_{\text{new}} = l_{\text{old}} \cdot \exp_{\text{diff}} + \sum_j \exp(S_{ij} - m_{\text{new}})
$$

$$
o_{\text{new}} = o_{\text{old}} \cdot \exp_{\text{diff}} + \sum_j \exp(S_{ij} - m_{\text{new}}) \cdot V_j
$$

遍历完所有 K/V 块后，最终输出为：

$$
O = o / l
$$

### 因果掩码优化
在自回归模型中，FlashAttention 可以跳过因果掩码上方的块，减少约一半的访存和计算。

## 时间/空间复杂度
- **时间复杂度**: $O(b \cdot n^2 \cdot d_{model})$，FLOPs 与标准注意力相同。
- **空间复杂度**: $O(b \cdot n \cdot d_{model})$，**线性于序列长度**，不存储 $N \times N$ 矩阵。
- **与替代方案对比**:
  - 标准 Attention: $O(N^2)$ 显存，IO 密集，长序列易 OOM。
  - FlashAttention: $O(N)$ 显存，计算密集型，可支持更长上下文。
  - Sparse Attention: 近似算法，可能损失精度；FlashAttention 是**精确算法**。

## 面试高频考点
1. **FlashAttention 的核心创新是什么？**
   **答案**: (1) 分块计算避免显式存储 $N \times N$ 注意力矩阵；(2) 在线 softmax 允许逐块更新统计量，无需整行数据同时可用。两者结合使显存从 $O(N^2)$ 降到 $O(N)$。

2. **在线 softmax 为什么是正确的？**
   **答案**: 当新的最大值 $m_{\text{new}}$ 出现时，旧的所有值都可以通过乘以 $\exp(m_{\text{old}} - m_{\text{new}})$ 重新缩放，等价于以新最大值为基准重新计算指数。数学上满足结合律，因此分块更新与一次性计算结果完全一致。

3. **FlashAttention 是近似算法吗？**
   **答案**: 不是。FlashAttention 在数学上与标准注意力完全等价，只是通过重排计算顺序和复用 SRAM 减少了 HBM（显存）读写。

4. **FlashAttention 的 backward 如何实现？**
   **答案**: FlashAttention-1 在 backward 时重新计算 forward 中的注意力分数（recomputation），而不是存储它们。FlashAttention-2 进一步优化了 warp 级并行，FlashAttention-3 引入了异步和低精度支持。

5. **因果掩码在 FlashAttention 中如何优化？**
   **答案**: 对于块 $(i, j)$，若该块完全位于因果掩码的上三角（即该 Q 块的所有位置都小于 K 块的所有位置），则直接跳过该块，节省约 50% 计算。

## 代码解析
- `_flash_attention_forward`: 核心分块逻辑。
  - `m`, `l`: 在线 softmax 的运行最大值和指数累加和，形状 `(batch, heads, seq_len)`。
  - `o`: 输出累加器，形状与 Q 相同。
  - 外层循环遍历 K/V 块，`kv_start:kv_end` 为当前块范围。
  - `scores`: 当前 Q 块与 K 块的点积分数。
  - `causal_mask`: 通过广播构造 `row_idx >= col_idx` 的布尔掩码。
  - `m_new`, `l_new`, `o` 的更新严格遵循在线 softmax 公式。
- `forward`: 标准 MHA 的 Q/K/V 投影后，调用 FlashAttention 计算逻辑，再拼接输出。
- 自测块包含与标准注意力的数值对比，验证在线 softmax 的正确性。

## 参考资料
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) (NeurIPS 2022)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) (2023)
- [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) (2024)
- [Online Softmax 讲解](https://blog.csdn.net/weixin_45264425/article/details/156851287)
