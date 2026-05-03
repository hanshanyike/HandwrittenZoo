# Prefix Tuning

## 算法简介

Prefix Tuning 是斯坦福大学于 2021 年提出的参数高效微调（PEFT）方法。与 Prompt Tuning 只在输入层添加可学习 token 不同，Prefix Tuning 在 **Transformer 每一层的 Key 和 Value 前面添加可学习的前缀向量**，通过影响注意力分布来引导模型行为，冻结全部预训练参数，只训练这些前缀。

## 核心思想

1. **深层干预**：不在输入层加 prompt，而是在每层的 K/V 中添加前缀，直接影响注意力机制。
2. **任务特定前缀**：不同任务学习不同的前缀，共享同一个冻结的预训练模型。
3. **重参数化稳定训练**：通过小 MLP 生成前缀向量，避免直接优化高维前缀的不稳定性。
4. **即插即用**：训练好的前缀可以像 LoRA 适配器一样保存和切换。

## 数学公式

### 1. 标准自注意力

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 $Q, K, V \in \mathbb{R}^{n \times d_k}$，$n$ 为序列长度。

### 2. 带前缀的自注意力

Prefix Tuning 在 $K$ 和 $V$ 前面拼接可学习的前缀向量 $P_k, P_v \in \mathbb{R}^{m \times d_k}$：

$$
K' = [P_k; K], \quad V' = [P_v; V]
$$

$$
\text{Attention}(Q, K', V') = \text{softmax}\left(\frac{Q[K']^T}{\sqrt{d_k}}\right)V'
$$

其中 $[;]$ 表示拼接操作，$m$ 为前缀长度。注意力矩阵的形状变为 $(n \times (m + n))$。

### 3. 每层独立前缀

对于 $L$ 层 Transformer，每层 $i$ 有独立的前缀参数：

$$
P_k^{(i)}, P_v^{(i)} \in \mathbb{R}^{m \times d_k}, \quad i = 1, 2, \dots, L
$$

总可训练参数量：

$$
2 \cdot L \cdot h \cdot m \cdot d_k
$$

其中 $h$ 为注意力头数（前缀在每个头上独立）。

### 4. MLP 重参数化

直接优化前缀向量 $P$ 训练不稳定，因此引入小 MLP：

$$
P = \text{MLP}(P_{\text{embed}})
$$

训练时优化 $P_{\text{embed}}$ 和 MLP 参数，推理时可缓存 MLP 输出，丢弃 MLP。

## 时间/空间复杂度

- **时间复杂度**：$O(B \cdot L_{seq} \cdot m \cdot d_{model})$ 每层
  - 额外开销来自序列长度增加 $m$（前缀长度），通常 $m \ll L_{seq}$。
- **空间复杂度**：$O(L \cdot m \cdot d_{model} \cdot 2)$
  - $L$ 层，每层 K 和 V 各一组前缀。
- **与 Prompt Tuning 对比**：Prompt Tuning 只在输入层加前缀，参数量 $O(m \cdot d_{model})$；Prefix Tuning 在每一层都加，参数量 $O(L \cdot m \cdot d_{model})$，但效果通常更好。

## 面试高频考点

### Q1: Prefix Tuning 和 Prompt Tuning 有什么区别？

**A**:

| 特性 | Prompt Tuning | Prefix Tuning |
|------|--------------|---------------|
| 插入位置 | 只在输入嵌入层 | 每层的 K/V |
| 参数量 | $m \cdot d$ | $2 \cdot L \cdot m \cdot d$ |
| 影响机制 | 改变输入表示 | 改变注意力分布 |
| 训练稳定性 | 较稳定 | 需 MLP 重参数化 |
| 效果 | 小模型上较弱 | 通常更强 |

Prompt Tuning 是 Prefix Tuning 的简化版，当模型规模足够大（>10B）时，Prompt Tuning 也能达到不错效果。

### Q2: 为什么 Prefix Tuning 要加在 K 和 V 上，而不是 Q 上？

**A**: 核心原因是**注意力机制的对称性**：
- 加在 K/V 上：每个 token 的 query 都会与前缀的 key 计算相似度，前缀的 value 会参与所有 token 的上下文聚合。这相当于给模型提供了一个**任务特定的全局上下文**。
- 加在 Q 上：只有前缀位置的 query 会参与计算，其他 token 不受影响，无法全局引导模型行为。

实验表明，只加在 K/V 上的效果远优于只加在 Q 上，或同时加在 Q/K/V 上。

### Q3: MLP 重参数化的作用是什么？训练完后可以去掉吗？

**A**: MLP 重参数化的作用是**提升训练稳定性**。直接优化高维前缀向量（如 $L \times m \times d$）容易陷入局部最优，且对初始化敏感。通过小 MLP（如 embed_dim -> 512 -> prefix_dim）生成前缀，相当于给优化过程增加了一个平滑的约束。

**训练完成后可以去掉 MLP**：将 MLP 的输出缓存为固定前缀向量，丢弃 MLP 参数，节省显存和计算。

### Q4: Prefix Tuning 与 LoRA 的对比？

**A**:

| 维度 | Prefix Tuning | LoRA |
|------|--------------|------|
| 修改对象 | 注意力输入（K/V） | 权重矩阵（W） |
| 参数量 | $O(L \cdot m \cdot d)$ | $O(r \cdot d)$ |
| 推理开销 | 序列长度增加 | 无（可合并） |
| 适用场景 | 生成任务、NLG | 通用，尤其 NLU |
| 多任务切换 | 切换前缀即可 | 切换 LoRA 权重即可 |

LoRA 更适合需要修改模型内部变换的场景，Prefix Tuning 更适合引导注意力分布的生成任务。

### Q5: Prefix Tuning 的前缀长度 $m$ 如何选择？

**A**: $m$ 的选择是效果与效率的权衡：
- **太小（如 m < 10）**：表达能力不足，难以覆盖复杂任务。
- **太大（如 m > 200）**：参数量增加，且可能引入噪声，效果边际递减。
- **推荐值**：原始论文推荐 50~200，对于简单任务 10~20 也可能足够。实际选择应通过验证集调优。

## 代码解析

### PrefixEmbedding 核心实现

```python
class PrefixEmbedding(nn.Module):
    def __init__(self, num_layers, num_heads, head_dim, prefix_len):
        self.prefix_tokens = nn.Parameter(
            torch.randn(num_layers, 2, num_heads, prefix_len, head_dim)
        )
```

前缀参数的形状为 `(num_layers, 2, num_heads, prefix_len, head_dim)`，其中 `2` 分别对应 key 和 value 的前缀。每个层、每个头都有独立的前缀，保证足够的表达能力。

### 带前缀的注意力计算

```python
# 在 K 和 V 前面拼接前缀
prefix_k = prefix_k.unsqueeze(0).expand(batch_size, -1, -1, -1)
K = torch.cat([prefix_k, K], dim=2)
V = torch.cat([prefix_v, V], dim=2)

scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
```

前缀向量被扩展到 batch 维度后，与原始 K/V 拼接。注意力计算时，每个 token 的 query 都会与前缀的 key 计算相似度，前缀的 value 参与所有位置的上下文聚合。

### MLP 重参数化

```python
class PrefixTuningMLP(nn.Module):
    def __init__(self, prefix_len, num_layers, num_heads, head_dim, mlp_dim=512):
        self.prefix_embed = nn.Parameter(torch.randn(prefix_len, mlp_dim))
        self.mlp = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.Tanh(),
            nn.Linear(mlp_dim, num_layers * 2 * num_heads * head_dim),
        )
```

通过一个小的 MLP 将低维嵌入映射到前缀向量。`Tanh` 激活提供非线性，`prefix_embed` 是可学习的输入。

## 参考资料

- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) — Prefix Tuning 原始论文
- [P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/abs/2110.07602) — Prefix Tuning 的扩展和优化
- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) — Prompt Tuning（Prefix Tuning 的简化版）
