# Mixture of Experts (MoE) Layer

## 算法简介

混合专家层（MoE Layer）是一种通过**稀疏激活**实现模型容量指数级扩展的神经网络结构。它将传统稠密 FFN 替换为多个并行的“专家”网络，并通过门控网络动态决定每个输入 token 应该由哪些专家处理。MoE 是现代大模型（如 Mixtral 8x7B、GShard、Switch Transformer）突破万亿参数规模的核心技术。

## 核心思想

1. **分而治之**：将复杂任务分解给多个“专家”子网络，每个专家专注于不同的数据分布或特征模式。
2. **稀疏激活**：每个输入只激活少数（Top-K）专家，而非全部，保证计算量不会随专家数线性爆炸。
3. **可学习路由**：门控网络根据输入内容动态选择最合适的专家，实现数据驱动的自适应计算。

## 数学公式

### 1. 门控网络（Gating Network）

给定输入 token 表示 $\mathbf{x} \in \mathbb{R}^{d_{model}}$，门控网络输出每个专家的分数：

$$
\mathbf{g}(\mathbf{x}) = \text{Softmax}(\mathbf{W}_g \mathbf{x}) \in \mathbb{R}^{E}
$$

其中 $E$ 为专家总数，$\mathbf{W}_g \in \mathbb{R}^{E \times d_{model}}$ 为门控权重矩阵。

### 2. Top-K 专家选择

对每个 token，只保留分数最高的 $K$ 个专家，并将它们的概率重新归一化：

$$
\mathcal{T} = \text{TopK}(\mathbf{g}(\mathbf{x}), K)
$$

$$
\tilde{g}_i(\mathbf{x}) = \frac{g_i(\mathbf{x})}{\sum_{j \in \mathcal{T}} g_j(\mathbf{x})}, \quad i \in \mathcal{T}
$$

### 3. MoE 输出

设第 $i$ 个专家的输出为 $\text{Expert}_i(\mathbf{x})$，则 MoE 层的最终输出为：

$$
\mathbf{y} = \sum_{i \in \mathcal{T}} \tilde{g}_i(\mathbf{x}) \cdot \text{Expert}_i(\mathbf{x})
$$

### 4. 与标准 FFN 的参数量对比

- 标准 FFN 参数量：$2 \cdot d_{model} \cdot d_{ff}$
- MoE 参数量：$E \cdot 2 \cdot d_{model} \cdot d_{ff} + E \cdot d_{model}$（门控网络）
- **激活参数量**：仅 $K \cdot 2 \cdot d_{model} \cdot d_{ff}$，远小于总参数量

## 时间/空间复杂度

- **时间复杂度**：$O(B \cdot L \cdot d_{model} \cdot K \cdot d_{ff})$
  - 仅 $K$ 个专家参与前向计算，与专家总数 $E$ 无关（理想情况下）。
- **空间复杂度**：$O(B \cdot L \cdot E)$（门控分数矩阵） + $O(E \cdot d_{model} \cdot d_{ff})$（专家参数）
- **与稠密模型对比**：在相同计算预算下，MoE 可将模型容量提升 $E/K$ 倍。

## 面试高频考点

### Q1: 为什么 MoE 能实现“大模型小计算”？

**A**: MoE 的核心是**稀疏性**。虽然模型总参数量随专家数 $E$ 线性增长，但每个输入 token 只激活 $K$ 个专家（$K \ll E$），因此前向传播的计算量仅与 $K$ 成正比。这使得模型可以在保持推理速度可控的前提下，大幅扩展参数量以提升表达能力。

### Q2: Top-K 选择是不可导的，MoE 如何训练？

**A**: Top-K 操作本身确实不可导，但 MoE 通过**直通估计器（Straight-Through Estimator）**或**软 Top-K（如 Gumbel-Softmax）**解决梯度回传问题。更常见的做法是：在反向传播时，Top-K 的 mask 被视为常数，梯度通过被选中的专家的门控权重回传；同时引入负载均衡损失（Load Balancing Loss）鼓励门控网络学习均匀路由，避免梯度消失。

### Q3: MoE 推理时有哪些工程挑战？

**A**: 主要挑战有三点：
1. **显存碎片**：不同 batch 中的 token 路由到的专家不同，导致显存访问不连续。
2. **通信开销**：专家并行（Expert Parallelism）下，需要将 token 发送到持有目标专家的 GPU，产生 all-to-all 通信。
3. **负载不均**：如果门控网络将所有 token 路由到少数热门专家，会导致这些专家排队，其他专家闲置（需要通过负载均衡损失缓解）。

### Q4: Top-K 的 K 一般取多少？为什么？

**A**: 实际系统中通常取 $K=1$（Switch Transformer）或 $K=2$（Mixtral 8x7B）。$K=1$ 时计算最省、通信最少，但表达能力稍弱；$K=2$ 在计算和效果之间取得较好平衡。$K$ 过大（如 $K \geq 4$）会导致计算量接近稠密模型，失去稀疏优势。

## 代码解析

### Expert 类

```python
class Expert(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
```

每个专家是一个标准的两层 FFN。所有专家结构相同但参数独立，相当于将一个大 FFN 拆分为多个小 FFN。

### 门控与 Top-K 选择

```python
gate_logits = self.gate(x_flat)          # (num_tokens, num_experts)
gate_probs = F.softmax(gate_logits, dim=-1)
topk_probs, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
```

门控网络将每个 token 映射为专家概率分布，然后 `torch.topk` 选出概率最高的 K 个专家索引。重新归一化确保每个 token 的权重和为 1。

### 按专家聚合计算

```python
for expert_id in range(self.num_experts):
    mask = topk_indices == expert_id
    token_mask = mask.any(dim=-1)
    expert_input = x_flat[token_mask]
    expert_output = self.experts[expert_id](expert_input)
    token_weights = topk_probs[token_mask][mask[token_mask]]
    output[token_mask] += token_weights.unsqueeze(-1) * expert_output
```

遍历每个专家，收集所有被路由到该专家的 token，批量计算后按门控权重写回输出缓冲区。这是 MoE 前向传播的核心逻辑。

## 参考资料

- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — MoE 的开山之作
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) — Google 的 MoE 并行训练框架
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8x7B 的 MoE 架构详解
