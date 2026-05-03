# Layer Normalization (LayerNorm)

## 算法简介

Layer Normalization（层归一化）是深度学习中一种**与 batch size 无关**的归一化技术，由 Ba 等人于 2016 年提出。它通过对**单个样本**在特征维度上计算统计量（均值、方差）进行归一化，广泛应用于 Transformer、RNN 及现代大语言模型（LLM）中。

## 核心思想

Batch Normalization 的痛点在于：
- 依赖 batch size，小批量时统计量不稳定；
- 对序列长度变化敏感（如 NLP 中不同句子长度），padding 会污染统计量；
- 训练和推理行为不一致（训练用 batch 统计量，推理用移动平均）。

LayerNorm 的关键洞察是：**把归一化的维度从“跨样本”转到“样本内部”**。每个样本独立计算均值和方差，无论 batch size 是多少、序列多长，归一化行为都稳定一致。

## 数学公式

对输入向量 $x \in \mathbb{R}^{D}$，LayerNorm 计算如下：

$$
\mu = \frac{1}{D} \sum_{i=1}^{D} x_i
$$

$$
\sigma^2 = \frac{1}{D} \sum_{i=1}^{D} (x_i - \mu)^2
$$

$$
\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
$$

其中：
- $\mu$：特征维度上的均值
- $\sigma^2$：特征维度上的**有偏方差**（与 PyTorch 官方实现一致，除以 $D$ 而非 $D-1$）
- $\epsilon$：极小常数（如 $10^{-5}$），防止除零
- $\gamma$（gamma）：可学习的缩放参数，初始化为 1
- $\beta$（beta）：可学习的偏移参数，初始化为 0
- $\odot$：逐元素乘法

### Pre-Norm vs Post-Norm

Transformer 中 LayerNorm 的放置位置有两种流派：

**Post-Norm（原始 Transformer）**：
$$
x_{out} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

**Pre-Norm（现代主流，GPT/LLaMA/T5 等）**：
$$
x_{out} = x + \text{Sublayer}(\text{LayerNorm}(x))
$$

| 特性 | Post-Norm | Pre-Norm |
|------|-----------|----------|
| 归一化位置 | 子层之后 | 子层之前 |
| 残差路径 | 经过归一化 | 保持纯净 |
| 训练稳定性 | 深层易梯度爆炸/消失 | 更稳定，易堆深层 |
| 收敛速度 | 需配合学习率预热 | 收敛更快 |
| 代表模型 | 原始 Transformer | GPT-2/3/4, LLaMA, BLOOM |

## 时间/空间复杂度

- **时间复杂度**：$O(B \times T \times D)$ —— 每个元素仅访问常数次
- **空间复杂度**：$O(D)$ —— 仅需存储 $\gamma$ 和 $\beta$ 两个可学习向量
- **与 BatchNorm 对比**：
  - 不依赖 batch size，batch_size=1 也能稳定工作；
  - 训练和推理逻辑完全一致，无需维护移动平均统计量。

## 面试高频考点

1. **LayerNorm 与 BatchNorm 的根本区别是什么？**
   **答案**：统计维度不同。BN 在 batch 维度（跨样本）统计，LN 在特征维度（样本内部）统计。因此 LN 与 batch size 无关，更适合变长序列和自回归模型。

2. **为什么 Transformer 使用 LayerNorm 而不是 BatchNorm？**
   **答案**：主要原因有三：① 序列长度可变，padding 会污染 BN 的 batch 统计量；② 自注意力机制中，同一位置不同样本的 token 语义无关，跨样本统计无意义；③ 小 batch 场景（如大模型分布式训练）BN 统计不稳定。

3. **Pre-Norm 和 Post-Norm 各有什么优缺点？**
   **答案**：Post-Norm 的残差路径经过归一化，深层网络梯度传播困难，训练不稳定；Pre-Norm 保持残差路径纯净，梯度可直接回传，训练更稳定，是现代大模型的主流选择，但可能略微削弱模型表达能力，需配合足够参数量弥补。

4. **LayerNorm 的方差计算为什么使用有偏估计（unbiased=False）？**
   **答案**：PyTorch 官方实现为了与前向 C++ 内核保持一致，使用有偏方差（除以 $N$）。面试中需指出这一点，若自己实现时误用 unbiased=True 会导致与官方输出存在微小差异。

5. **LayerNorm 中的 $\gamma$ 和 $\beta$ 有什么作用？可以去掉吗？**
   **答案**：$\gamma$ 和 $\beta$ 是可学习的仿射参数，允许模型在必要时“撤销”归一化效果，恢复原始表达能力。某些场景（如部分推理优化）会去掉它们（elementwise_affine=False），但通常保留以提升模型容量。

## 代码解析

### 核心归一化逻辑

```python
mean = x.mean(dim=-1, keepdim=True)          # 沿最后一维求均值
var = x.var(dim=-1, keepdim=True, unbiased=False)  # 有偏方差
x_norm = (x - mean) / torch.sqrt(var + self.eps)   # 标准化
```

- `keepdim=True` 保持维度为 `(B, T, 1)`，便于广播运算；
- `unbiased=False` 确保与 PyTorch 官方 `nn.LayerNorm` 数值等价。

### 可学习仿射变换

```python
if self.elementwise_affine:
    x_norm = x_norm * self.gamma
    if self.beta is not None:
        x_norm = x_norm + self.beta
```

- `gamma` 初始化为 1，`beta` 初始化为 0；
- 支持 `bias=False` 以兼容无偏移参数的变体。

### Pre-Norm / Post-Norm 结构示例

代码中提供了 `PreNormTransformerBlock` 和 `PostNormTransformerBlock` 两个辅助类，直观展示两种归一化位置的区别：
- **Pre-Norm**：`x + Sublayer(LayerNorm(x))` —— 残差路径纯净
- **Post-Norm**：`LayerNorm(x + Sublayer(x))` —— 原始 Transformer 风格

## 参考资料

- [Ba et al., "Layer Normalization", arXiv:1607.06450, 2016](https://arxiv.org/abs/1607.06450)
- [Vaswani et al., "Attention Is All You Need", NeurIPS 2017](https://arxiv.org/abs/1706.03762)
- [Xiong et al., "On Layer Normalization in the Transformer Architecture", ICML 2020](https://arxiv.org/abs/2002.04745) —— Pre-Norm 系统性分析
- [PyTorch 官方文档 - torch.nn.LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
- [Karpathy llm.c LayerNorm 教程](https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md)
