# LoRA (Low-Rank Adaptation)

## 算法简介

LoRA（低秩适配）是微软于 2021 年提出的参数高效微调（PEFT）方法。它假设预训练模型在微调时的权重更新 $\Delta W$ 具有**低内在秩（low intrinsic rank）**，因此可以用两个远小于原矩阵的低秩矩阵来近似，从而将可训练参数量减少数个数量级。

## 核心思想

1. **冻结基座**：预训练权重 $W_0$ 完全冻结，保留预训练知识。
2. **低秩分解**：引入 $A \in \mathbb{R}^{d \times r}$ 和 $B \in \mathbb{R}^{r \times k}$，其中 $r \ll \min(d, k)$。
3. **残差更新**：微调后的前向传播为 $h = W_0 x + \frac{\alpha}{r} B A x$。
4. **即插即用**：LoRA 模块可注入到任意线性层，训练完成后可与基座权重合并，推理零开销。

## 数学公式

### 1. 低秩分解

对于某层的权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，全量微调的更新为 $\Delta W \in \mathbb{R}^{d \times k}$。LoRA 用低秩矩阵近似：

$$
\Delta W \approx B A, \quad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}
$$

### 2. 前向传播

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} B A x
$$

其中 $\alpha$ 为缩放超参数，$r$ 为秩。$\frac{\alpha}{r}$ 控制 LoRA 更新的幅度，通常 $\alpha$ 固定（如 16、32），通过调整 $r$ 来控制参数量。

### 3. 可训练参数量对比

| 方法 | 每层参数量 | d=4096, k=4096 时 |
|------|-----------|-------------------|
| 全量微调 | $d \times k$ | 16.8M |
| LoRA (r=8) | $r \times (d + k)$ | 65.5K |
| LoRA (r=64) | $r \times (d + k)$ | 524K |

**节省比例**：LoRA(r=8) 仅需要全量微调的 **0.39%** 参数。

### 4. 权重合并（Merge）

推理时可将 LoRA 合并回基座权重：

$$
W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A
$$

合并后推理速度与原始模型完全一致，无额外计算开销。

## 时间/空间复杂度

- **时间复杂度**：$O(B \cdot L \cdot d_{model} \cdot r)$ 每层
  - 相比全量微调的 $O(B \cdot L \cdot d_{model}^2)$，当 $r \ll d_{model}$ 时大幅节省。
- **空间复杂度**：$O(d_{model} \cdot r)$ 每层（仅存储 A 和 B）
- **与 Adapter 对比**：Adapter 在层间插入瓶颈层，会引入推理延迟；LoRA 可合并，推理零开销。

## 面试高频考点

### Q1: 为什么 LoRA 假设 $\Delta W$ 是低秩的？有理论依据吗？

**A**: 该假设基于**内在维度（Intrinsic Dimensionality）**研究。Aghajanyan et al. (2020) 发现，预训练模型在下游任务上的微调更新可以被投影到一个低维子空间中而不显著损失性能。LoRA 的作者进一步假设这个子空间的维度（即秩 $r$）非常小（如 $r \leq 64$），实验表明即使 $r=1$ 在某些任务上也能取得不错的效果。

### Q2: LoRA 的 $r$ 和 $\alpha$ 如何选择？它们的关系是什么？

**A**: 
- **$r$（秩）**：控制 LoRA 的表达能力。常见取值为 4、8、16、32、64。任务越复杂、数据量越大，$r$ 通常越大。
- **$\alpha$（缩放系数）**：控制 LoRA 更新的幅度。通常固定为 $2r$ 或 $4r$（如 $r=8$ 时 $\alpha=16$）。
- **关系**：实际缩放因子为 $\alpha / r$。当 $\alpha = 2r$ 时，缩放因子为 2；当 $\alpha = r$ 时，缩放因子为 1。增大 $\alpha$ 相当于增大 LoRA 的学习率。

### Q3: 为什么 lora_B 要初始化为零，而 lora_A 用随机初始化？

**A**: 这是 LoRA 训练稳定性的关键设计：
- **lora_B = 0**：确保训练开始时 LoRA 的输出为 0，模型行为与预训练时完全一致。这样可以从一个稳定的起点开始微调。
- **lora_A 随机初始化**：如果 A 也为 0，则梯度永远为 0，无法训练。A 的随机初始化保证梯度可以正常回传。

### Q4: LoRA 应该注入到哪些层？为什么通常只注入 Attention 的 Q 和 V？

**A**: 原始论文推荐注入到 Transformer 的 $W_q$ 和 $W_v$（查询和值投影）。原因：
1. **效果**：实验证明只注入 Q 和 V 就能达到接近注入所有层的效果。
2. **效率**：减少可训练参数量，加速训练。
3. **稳定性**：同时注入 Q、K、V、O 有时会导致训练不稳定。

但在实际应用中，对于复杂任务，也常注入到所有线性层（包括 FFN）以获得最佳效果。

### Q5: LoRA 与 Adapter、Prefix Tuning 的对比？

**A**:

| 方法 | 修改位置 | 推理开销 | 可合并 | 典型节省 |
|------|---------|---------|--------|---------|
| LoRA | 权重矩阵旁路 | 无（可合并） | 是 | 99%+ |
| Adapter | 层间插入瓶颈层 | 有（额外前向） | 否 | 90%+ |
| Prefix Tuning | 输入前加前缀 | 有（KV Cache 增加） | 否 | 90%+ |

LoRA 的最大优势是推理零开销和可合并性，非常适合生产部署。

## 代码解析

### LoRALayer 核心实现

```python
class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16.0, dropout=0.0):
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.scaling = alpha / rank

    def forward(self, x):
        x_d = self.dropout(x)
        h = torch.matmul(x_d, self.lora_A)
        h = torch.matmul(h, self.lora_B)
        return h * self.scaling
```

`lora_A` 用 Kaiming 初始化，`lora_B` 初始化为零，确保训练起点稳定。`scaling = alpha / rank` 控制更新幅度。

### LinearWithLoRA 包装层

```python
class LinearWithLoRA(nn.Module):
    def __init__(self, base_layer, rank, alpha, dropout):
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False  # 冻结基座
        self.lora = LoRALayer(...)

    def forward(self, x):
        return self.base_layer(x) + self.lora(x)
```

包装层将原始 `nn.Linear` 与 LoRA 增量相加，同时冻结基座权重。

### 权重合并

```python
def merge_lora_weights(model):
    delta_W = module.lora.scaling * (module.lora.lora_B @ module.lora.lora_A.t())
    module.base_layer.weight.data += delta_W.t()
```

合并后推理无额外开销，适合部署。

## 参考资料

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA 原始论文
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255) — 低秩假设的理论基础
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — LoRA 的量化扩展
