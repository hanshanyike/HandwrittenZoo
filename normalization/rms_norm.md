# RMSNorm (Root Mean Square Layer Normalization)

## 算法简介

RMSNorm（均方根层归一化）由 Zhang & Sennrich 于 2019 年提出，是对 LayerNorm 的精简改进。它去除了 LayerNorm 中的**均值中心化（mean centering）**步骤，仅保留均方根缩放，在保持模型性能的同时减少了计算开销。RMSNorm 已成为 LLaMA、GPT-NeoX、T5 等主流大语言模型的标准归一化层。

## 核心思想

LayerNorm 的公式包含两个部分：
1. **去均值**：$x - \mu$ —— 将数据分布中心移到 0
2. **重新缩放**：除以标准差并乘 $\gamma$ —— 控制分布尺度

RMSNorm 的作者通过实验发现：**LayerNorm 的成功主要归功于“重新缩放”而非“去均值”**。既然去均值对性能提升贡献有限，不如直接去掉，从而：
- 减少一次均值计算和一次减法操作；
- 简化网络结构，降低计算量；
- 在超大模型上保持与 LayerNorm 相当的性能。

## 数学公式

对输入向量 $x \in \mathbb{R}^{D}$，RMSNorm 计算如下：

$$
\text{RMS}(x) = \sqrt{ \frac{1}{D} \sum_{i=1}^{D} x_i^2 + \epsilon }
$$

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma
$$

其中：
- $\text{RMS}(x)$：均方根（Root Mean Square），相当于“无均值的 L2 范数除以维度”
- $\epsilon$：极小常数，防止除零
- $\gamma$（gamma）：可学习的缩放参数，初始化为 1
- **注意**：RMSNorm **没有** $\beta$（偏移参数），因为它不做均值中心化

与 LayerNorm 的对比：

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 去均值 | 有 ($x - \mu$) | 无 |
| 缩放 | 有 ($\gamma$) | 有 ($\gamma$) |
| 偏移 | 有 ($\beta$) | 无 |
| 计算量 | 略高 | 略低 |
| 大模型采用 | 早期 Transformer | LLaMA, GPT-NeoX, T5 |

## 时间/空间复杂度

- **时间复杂度**：$O(B \times T \times D)$ —— 与 LayerNorm 同级，但常数项更小（省去均值计算）
- **空间复杂度**：$O(D)$ —— 仅需存储 $\gamma$，比 LayerNorm 少一个 $\beta$
- **与 LayerNorm 对比**：
  - 前向传播减少约 10%~15% 的计算量（取决于硬件和实现）；
  - 在超大模型（数十亿到千亿参数）上，累积节省显著。

## 面试高频考点

1. **RMSNorm 与 LayerNorm 的核心区别是什么？**
   **答案**：RMSNorm 去除了 LayerNorm 中的均值中心化（$x - \mu$）和偏移参数 $\beta$，仅保留均方根缩放和可学习缩放参数 $\gamma$。作者认为 LayerNorm 的效果主要来自重新缩放而非去均值，因此 RMSNorm 在减少计算的同时保持了相近性能。

2. **为什么大模型（如 LLaMA）更倾向于使用 RMSNorm？**
   **答案**：① 计算效率更高，省去均值计算；② 参数量略少（没有 $\beta$）；③ 实验表明在超大模型上性能与 LayerNorm 相当甚至略优；④ 简化了实现，便于在不同硬件上做定制优化。

3. **RMSNorm 的输出均值一定为 0 吗？**
   **答案**：不一定。由于 RMSNorm 不做去均值操作，输出均值通常不为 0。而 LayerNorm 因为减去了均值，输出均值为 0（在应用 $\gamma$、$\beta$ 之前）。这是两者在数值特性上的根本差异。

4. **RMSNorm 的 eps 应该放在 RMS 内部还是外部？**
   **答案**：标准做法是在平方均值之后、开方之前加 eps：$\sqrt{\text{mean}(x^2) + \epsilon}$。这样可以防止当所有输入接近 0 时出现除零错误，同时保持数值稳定性。

5. **为什么 RMSNorm 通常没有 bias（beta）参数？**
   **答案**：因为 RMSNorm 的设计哲学是“只做缩放、不做平移”。去均值本身就是为了消除分布偏移，既然 RMSNorm 不去均值，也就不需要额外的 beta 来恢复偏移。保留 gamma 是为了让模型学习最优的缩放比例。

## 代码解析

### 核心归一化逻辑

```python
mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
rms = torch.sqrt(mean_square + self.eps)
x_norm = x_fp32 / rms
```

- 不计算均值，直接对输入平方后求平均；
- `keepdim=True` 保证广播时维度对齐；
- `eps` 在平方均值之后、开方之前加入，防止除零。

### 数据类型处理

```python
input_dtype = x.dtype
x_fp32 = x.float()
# ... 计算 ...
return x_norm.to(input_dtype)
```

- 大模型训练常用 `float16` 或 `bfloat16`，中间计算转为 `float32` 可避免低精度下的数值不稳定；
- 计算完成后转回原始类型，节省显存。

### 极简实现

```python
return self.weight * (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps))
```

- `torch.rsqrt` 计算 $1 / \sqrt{x}$，比先 `sqrt` 再除更高效；
- 此版本与 PyTorch 2.x 官方 `nn.RMSNorm` 逻辑一致。

## 参考资料

- [Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019](https://arxiv.org/abs/1910.07467)
- [PyTorch 官方文档 - torch.nn.RMSNorm](https://pytorch.org/docs/main/generated/torch.nn.RMSNorm.html)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) —— RMSNorm 在大模型中的实践
- [GPT-NeoX: An Open-Source Autoregressive Language Model](https://arxiv.org/abs/2204.06745)
