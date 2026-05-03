# KV Cache (Key-Value Cache)

## 算法简介

KV Cache 是 Transformer 自回归生成推理中最基础、最核心的 **空间换时间** 优化技术。它在逐 token 生成时缓存历史 Key 和 Value 张量，避免每一步都对完整前缀重新计算 Attention，从而将单步复杂度从 $O(n^2)$ 降至 $O(1)$（相对已生成长度）。

## 核心思想

Transformer 的 Attention 机制在生成第 $n$ 个 token 时，需要让当前 Query 与前面所有 $1 \sim n$ 个 token 的 Key、Value 做交互。若不缓存：
- 每步都要对完整前缀重新做 Q/K/V 投影；
- 计算量随序列长度线性增长，导致生成速度越来越慢。

KV Cache 的洞察很简单：**历史 token 的 K、V 在上一步已经算过了，把它们存下来，新 token 只算自己的 K、V，再和缓存拼接即可。**

## 数学公式

标准缩放点积 Attention：

$$
Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

设当前已生成长度为 $t$，新 token 的 Key、Value 为 $k_{new}, v_{new}$，缓存中的历史 K、V 为 $K_{:t}, V_{:t}$：

$$
K_{:t+1} = concat(K_{:t}, k_{new}) \\
V_{:t+1} = concat(V_{:t}, v_{new}) \\
O = softmax\left(\frac{q_{new} K_{:t+1}^T}{\sqrt{d_k}}\right) V_{:t+1}
$$

其中 $q_{new}$ 是当前 token 的 Query。

## 时间/空间复杂度

- **时间复杂度（单步）**：$O(1)$ 增量计算 vs. 无缓存时的 $O(t)$
- **空间复杂度**：$O(B \cdot L \cdot H_{kv} \cdot T_{max} \cdot d_{head})$
  - $B$: batch size
  - $L$: 层数
  - $H_{kv}$: KV 头数（GQA/MQA 可大幅减少）
  - $T_{max}$: 最大序列长度
  - $d_{head}$: 头维度

显存占用是 LLM 推理中的主要瓶颈之一，因此业界衍生出 **MQA/GQA**、**KV Cache 量化**、**PageAttention** 等进一步压缩显存的技术。

## 面试高频考点

1. **问题：KV Cache 为什么能加速推理？代价是什么？**
   **答案**：加速原因是避免了历史 token K/V 的重复计算；代价是显存占用随序列长度线性增长，成为长上下文推理的主要瓶颈。

2. **问题：KV Cache 在多轮对话中如何管理？**
   **答案**：通常每轮对话结束后可选择保留（多轮复用前缀）或清空（避免上下文混淆）。工业界还会做 Prefix Cache，将系统提示等公共前缀跨请求共享。

3. **问题：GQA/MQA 与 KV Cache 的关系？**
   **答案**：MQA（Multi-Query Attention）让所有 Q 头共享一组 K/V 头，GQA（Grouped-Query Attention）将 Q 头分组共享。两者都能将 KV Cache 显存占用压缩 4~8 倍，是目前大模型的标配。

4. **问题：KV Cache 的显存占用如何估算？**
   **答案**：公式为 $2 \times B \times L \times H_{kv} \times T \times d_{head} \times bytes_{dtype}$。以 FP16、batch=1、层数=32、头数=8、长度=4096、head_dim=128 为例，约为 $2 \times 32 \times 8 \times 4096 \times 128 \times 2 \approx 512\ MB$。

## 代码解析

### KVCache 类

- **`__init__`**：预分配固定大小的零张量作为缓冲区，避免生成过程中动态扩容带来的显存碎片。
- **`update`**：将新 K/V 写入当前长度指针指向的位置，然后指针后移。返回完整缓存供 Attention 计算。
- **`reset`**：将长度指针归零并清零缓冲区，用于新对话或新 batch。

### CachedMultiHeadAttention 类

- **训练时**：`use_cache=False`，等价于标准 MHA，不启用缓存。
- **推理时**：`use_cache=True`，首次调用会根据 batch_size 和设备自动初始化 `KVCache`。
- **GQA 支持**：当 `num_kv_heads < num_heads` 时，通过 `repeat_interleave` 将 K/V 头复制到与 Q 头数量匹配，再计算 Attention。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原论文
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA 论文
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — GQA 论文
- [Hugging Face Blog: KV Cache](https://huggingface.co/blog/kv-cache) — KV Cache 科普
