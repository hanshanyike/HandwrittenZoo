# Switch Transformer

## 算法简介

Switch Transformer 是 Google 于 2021 年提出的 MoE 架构，其核心创新是**每个 token 只激活 1 个专家（Top-1）**，将稀疏性推向极致。相比传统 MoE（Top-2 或更多），Switch Transformer 在保持模型容量巨大优势的同时，显著降低了计算和通信开销，成功将模型规模扩展到万亿参数级别。

## 核心思想

1. **极致稀疏（K=1）**：每个 token 只路由到分数最高的 1 个专家，计算量最小。
2. **容量限制（Capacity Factor）**：每个专家有最大处理 token 数，防止单专家过载；超出的 token 被丢弃（dropped tokens）。
3. **可微负载均衡**：通过辅助损失（aux_loss）鼓励门控网络均匀分配 token。
4. **简化设计**：去掉传统 MoE 的噪声门控、重要性损失等复杂机制，用简单有效的方案实现大规模训练。

## 数学公式

### 1. Top-1 路由

$$
\text{expert}(x) = \arg\max_{i}(g_i(x))
$$

$$
y = g_{\text{expert}(x)}(x) \cdot \text{Expert}_{\text{expert}(x)}(x)
$$

其中 $g_i(x)$ 为门控网络对专家 $i$ 的 softmax 输出。

### 2. 容量限制

每个专家的最大容量为：

$$
\text{capacity} = \left\lceil \frac{T}{E} \cdot \text{capacity\_factor} \right\rceil
$$

其中 $T$ 为总 token 数，$E$ 为专家数。当路由到某专家的 token 超过 capacity 时，按门控分数排序，只保留前 capacity 个，其余被丢弃。

### 3. 负载均衡损失

与标准 MoE 相同：

$$
\mathcal{L}_{\text{aux}} = E \cdot \sum_{i=1}^{E} f_i \cdot P_i
$$

总损失：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \cdot \mathcal{L}_{\text{aux}}
$$

通常 $\alpha = 0.01$。

### 4. 专家并行（Expert Parallelism）

Switch Transformer 采用专家并行策略：
- 将 $E$ 个专家均匀分布在 $N$ 个设备上，每个设备负责 $E/N$ 个专家。
- 前向传播时，token 通过 all-to-all 通信发送到目标设备，计算后再 all-to-all 收集回来。
- K=1 时通信量最小，因为每个 token 只需发送到一个设备。

## 时间/空间复杂度

- **时间复杂度**：$O(B \cdot L \cdot d_{model} \cdot d_{ff})$
  - 每个 token 只激活 1 个专家，与标准 Transformer FFN 的计算量相同！
  - 但模型参数量扩展了 $E$ 倍。
- **空间复杂度**：$O(B \cdot L \cdot E)$（路由表） + $O(E \cdot d_{model} \cdot d_{ff})$（专家参数）
- **通信复杂度**：$O(B \cdot L \cdot d_{model})$（all-to-all 通信量）

## 面试高频考点

### Q1: Switch Transformer 与传统 MoE（如 GShard）的核心区别是什么？

**A**: 核心区别有三点：
1. **Top-1 vs Top-2**：Switch 只选 1 个专家，GShard 选 2 个。K=1 使计算量和通信量减半。
2. **容量限制**：Switch 显式引入 capacity factor 限制每个专家的负载，GShard 没有硬限制。
3. **简化设计**：Switch 去掉了噪声门控（Noisy Top-K Gating）和重要性损失，只用简单的 aux_loss，工程实现更简洁。

### Q2: Capacity Factor 取多大合适？太大或太小有什么问题？

**A**: 
- **太小（如 < 0.5）**：大量 token 被丢弃，模型无法充分利用所有信息，性能下降。
- **太大（如 > 2.0）**：虽然不会丢弃 token，但内存浪费严重，且负载均衡损失难以将负载压到均匀分布。
- **推荐值**：训练时通常取 1.0~1.25，推理时取 1.0 或更小（因为推理 batch 小，容量需求低）。

### Q3: Dropped Tokens 如何处理？直接丢弃会不会导致信息丢失？

**A**: Switch Transformer 对 dropped tokens 的处理是**直接走残差连接**：即该 token 不经过专家 FFN，只保留 Self-Attention 的输出。这确实会导致信息丢失，但通过以下方式缓解：
1. 负载均衡损失使丢弃率保持在较低水平（通常 < 1%）。
2. 深层网络中，后续层有机会补偿丢失的信息。
3. 训练时 capacity factor 稍大，丢弃率更低。

### Q4: Switch Transformer 的参数量和计算量关系如何？

**A**: 这是 Switch Transformer 最精妙的地方：
- **参数量**：随专家数 $E$ 线性增长。例如 8 个专家，参数量约为标准 Transformer 的 8 倍。
- **计算量（FLOPs）**：与标准 Transformer 几乎相同！因为每个 token 只激活 1 个专家。
- **实际速度**：由于 all-to-all 通信和负载不均，实际推理速度通常比同等计算量的稠密模型慢 10%~30%。

## 代码解析

### SwitchMoELayer 核心逻辑

```python
# 1) 路由：每个 token 选 1 个专家
expert_indices, gate_probs, expert_gate = self.router(x_flat)

# 2) 容量限制
capacity = int((num_tokens / self.num_experts) * self.capacity_factor)

# 3) 按专家处理，超出容量的 token 被丢弃
for expert_id in range(self.num_experts):
    token_mask = expert_indices == expert_id
    selected_indices = torch.where(token_mask)[0]
    if selected_indices.numel() > capacity:
        # 按分数排序，保留前 capacity 个
        selected_gate = expert_gate[selected_indices]
        _, sorted_idx = torch.sort(selected_gate, descending=True)
        keep_idx = selected_indices[sorted_idx[:capacity]]
        drop_idx = selected_indices[sorted_idx[capacity:]]
        dropped[drop_idx] = True
```

路由后按专家聚合 token，通过容量限制防止单专家过载。被丢弃的 token 直接复制输入，由外部残差连接处理。

### 辅助损失计算

```python
expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()
f = expert_mask.mean(dim=0)   # 实际分配比例
P = gate_probs.mean(dim=0)    # 平均门控概率
aux_loss = self.num_experts * torch.sum(f * P)
```

与标准 MoE 的负载均衡损失完全一致，确保各专家负载均匀。

## 参考资料

- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch Transformer 原始论文
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) — Google 的 MoE 并行框架
- [Tutel: Adaptive Mixture-of-Experts at Scale](https://arxiv.org/abs/2206.03382) — 微软的高效 MoE 系统实现
