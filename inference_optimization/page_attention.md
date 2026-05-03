# PageAttention (Simplified)

## 算法简介

PageAttention 是 vLLM 推理引擎的核心创新，它将操作系统中经典的 **虚拟内存 + 分页** 思想引入 LLM 的 KV Cache 管理。通过将逻辑 token 序列与物理显存块解耦，PageAttention 消除了传统静态/动态分配中的显存碎片，并天然支持请求间内存共享（如 beam search、并行采样）。

## 核心思想

传统 KV Cache 分配方式有两种缺陷：
1. **静态预留**：为每个请求按最大长度预分配连续显存，导致严重内部碎片（未生成长度被浪费）；
2. **动态扩容**：需要连续内存块，随着服务运行产生外部碎片，导致显存利用率低。

PageAttention 的洞察来自操作系统分页：
- 将物理显存划分为固定大小的 **块（block）**；
- 每个请求维护一张 **页表（page table）**，将逻辑块号映射到物理块号；
- 物理块按需分配，非连续存储，通过页表聚合访问；
- 引用计数实现 **Copy-on-Write** 共享。

## 数学公式

逻辑 token 索引 $t$ 到物理位置的映射：

$$
\text{logical_block} = \left\lfloor \frac{t}{B} \right\rfloor, \quad
\text{offset} = t \bmod B
$$

$$
\text{physical_block} = \text{page_table}[\text{logical_block}]
$$

其中 $B$ 为 block_size。Attention 计算时，kernel 根据上述映射从分散的物理块中 gather 数据：

$$
O = \text{Attention}\left(Q, \text{Gather}(K_{phys}), \text{Gather}(V_{phys})\right)
$$

## 时间/空间复杂度

- **块分配/回收**：$O(1)$ 均摊（基于空闲集合）
- **单 token 写入**：$O(1)$ 地址转换 + 显存拷贝
- **Attention 读取**：实际由 CUDA kernel 完成，通过页表索引直接读取物理块，避免显式 gather 开销
- **空间利用率**：接近 100%（仅最后一个块可能有内部碎片，大小为 $B-1$ token）

相比静态预留，显存容量可支撑 **2~4 倍** 并发请求。

## 面试高频考点

1. **问题：PageAttention 解决了传统 KV Cache 管理的哪些痛点？**
   **答案**：解决了内部碎片（静态预留）和外部碎片（动态连续分配）问题；同时通过引用计数实现内存共享，支持 beam search、并行采样等场景。

2. **问题：为什么叫 PageAttention？与操作系统分页的对应关系是什么？**
   **答案**：借鉴操作系统虚拟内存机制：逻辑地址（token 序列）通过页表映射到物理页帧（显存块）；块可以不连续，按需分配，支持 fork 时的写时复制。

3. **问题：PageAttention 如何支持 beam search？**
   **答案**：Beam search 展开多个候选序列时，子序列通过 `fork_sequence` 共享父序列的物理块引用；当某个候选生成新 token 时，若块被多个序列共享，则 copy-on-write 分配新块，避免污染其他候选。

4. **问题：vLLM 中 block_size 一般怎么选？**
   **答案**：常见为 16 或 32。太大会增加最后一个块的内部碎片；太小会增加页表长度和 kernel 索引开销。需在碎片与索引开销之间权衡。

## 代码解析

### BlockAllocator

- **`allocate`**：从空闲集合中弹出物理块 ID，O(1)。
- **`free`**：引用计数减一，仅当计数归零时才回收到空闲池，确保共享安全。
- **`incr_ref`**：fork 时增加父块引用计数。

### PagedKVCache

- **`add_sequence`**：根据初始长度计算所需逻辑块数，分配物理块并建立页表。
- **`fork_sequence`**：复制父页表并增加引用计数，实现 O(1) 的序列分叉。
- **`append`**：核心写入路径。计算 token 的逻辑块号和块内偏移；若逻辑块尚无物理映射，则动态申请新块。
- **`get_kv`**：按逻辑顺序从物理块 gather K/V。注意：真实 vLLM 中此步骤在 CUDA kernel 内完成，避免 host-device 同步。

### PagedAttentionLayer

演示如何将分页缓存接入 Attention 计算。为简化教学，这里先在 Python 侧 `get_kv`  gather 成连续张量，再计算 Attention。工业实现中，page table 会传入 custom CUDA kernel，直接按块索引读取显存。

## 参考资料

- [vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — 原论文
- [vLLM Paged Attention Docs](https://docs.vllm.ai/en/v0.10.1/design/paged_attention.html) — 官方设计文档
- [vLLM Block Table Implementation](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/block_table.py) — 官方代码
