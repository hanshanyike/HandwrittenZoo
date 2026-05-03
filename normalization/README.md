# Normalization 归一化层

本目录包含深度学习中最核心的三种归一化层的从零实现与深度解析文档。

## 文件结构

```
normalization/
├── layer_norm.py      # LayerNorm 实现（含 Pre-Norm / Post-Norm 对比）
├── layer_norm.md      # LayerNorm 算法详解与面试考点
├── rms_norm.py        # RMSNorm 实现（大模型标配归一化）
├── rms_norm.md        # RMSNorm 算法详解与面试考点
├── batch_norm.py      # BatchNorm 实现（1D/2D，训练/推理双模式）
├── batch_norm.md      # BatchNorm 算法详解与面试考点
└── README.md          # 本文件：目录总览与面试焦点
```

## 三种归一化层速查对比

| 特性 | BatchNorm | LayerNorm | RMSNorm |
|------|-----------|-----------|---------|
| 提出时间 | 2015 (Ioffe & Szegedy) | 2016 (Ba et al.) | 2019 (Zhang & Sennrich) |
| 统计维度 | Batch (跨样本) | 特征维度 (样本内) | 特征维度 (样本内) |
| 去均值 | 有 | 有 | **无** |
| 偏移参数 $\beta$ | 有 | 有 | **无** |
| 依赖 batch size | **是** | 否 | 否 |
| 训练/推理一致性 | 不一致（需移动平均） | **一致** | **一致** |
| 适用序列模型 | 差 | 好 | 好 |
| 代表模型 | ResNet, VGG, Inception | 原始 Transformer, BERT | LLaMA, GPT-NeoX, T5 |
| 面试热度 | 高 | **极高** | 高 |

## 面试焦点速览

### 1. 为什么 Transformer 用 LayerNorm 而不是 BatchNorm？

这是面试中出现频率最高的问题之一，需从四个维度回答：

- **序列长度可变**：NLP 中句子长度不同，padding 会污染 BN 的 batch 统计量；LN 在样本内部统计，不受序列长度影响。
- **自注意力机制**：同一 batch 中同一位置的 token 可能来自语义完全不同的句子，跨样本统计无意义。
- **batch size 敏感**：大模型分布式训练时单卡 batch size 很小，BN 统计不稳定；LN 与 batch size 无关。
- **训练/推理一致性**：BN 推理需依赖移动平均，LN 训练和推理逻辑完全一致，实现更简单。

### 2. Pre-Norm vs Post-Norm 怎么选？

| 特性 | Post-Norm | Pre-Norm |
|------|-----------|----------|
| 结构 | `LN(x + Sublayer(x))` | `x + Sublayer(LN(x))` |
| 残差路径 | 经过归一化 | **保持纯净** |
| 深层训练 | 易梯度爆炸/消失 | **更稳定** |
| 主流采用 | 原始 Transformer | **GPT/LLaMA/现代 LLM** |

**结论**：现代大模型（GPT 系列、LLaMA、BLOOM 等）**全部使用 Pre-Norm**，因为它让深层网络的梯度传播更稳定。

### 3. RMSNorm 相比 LayerNorm 的优势是什么？

- **计算效率**：去掉均值计算，前向传播减少约 10%~15% 计算量；
- **参数量更少**：没有 $\beta$ 偏移参数；
- **性能相当**：在超大模型上实验表明与 LayerNorm 性能持平甚至略优；
- **实现简洁**：更易在自定义硬件/内核上做深度优化。

### 4. BatchNorm 的 momentum 是怎么工作的？

PyTorch 中的 `momentum` 定义：

```
running_mean = (1 - momentum) * running_mean + momentum * batch_mean
```

默认 `momentum=0.1` 表示新 batch 统计量占 10%，历史累积占 90%。**注意**：这与 SGD 优化器中的 momentum 含义不同，不要混淆。

### 5. 归一化层中的 eps 有什么作用？应该放在哪里？

`eps`（如 1e-5）是为了防止分母为 0 的数值稳定项。标准位置是在**方差/平方均值之后、开方之前**：

```
sqrt(var + eps)   # LayerNorm / BatchNorm
sqrt(mean(x^2) + eps)  # RMSNorm
```

这样即使输入全为 0，分母也不会出现 0。

## 学习路径建议

1. **入门**：先阅读 `batch_norm.md`，理解归一化的基本动机和“训练/推理不一致”这一关键设计；
2. **进阶**：阅读 `layer_norm.md`，掌握 Pre-Norm/Post-Norm 的区别，这是 Transformer 面试的核心考点；
3. **前沿**：阅读 `rms_norm.md`，了解大模型时代的归一化趋势，理解“去均值”这一简化背后的理论依据。

## 参考资料

- [Ioffe & Szegedy, "Batch Normalization", ICML 2015](https://arxiv.org/abs/1502.03167)
- [Ba et al., "Layer Normalization", 2016](https://arxiv.org/abs/1607.06450)
- [Vaswani et al., "Attention Is All You Need", NeurIPS 2017](https://arxiv.org/abs/1706.03762)
- [Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019](https://arxiv.org/abs/1910.07467)
- [Xiong et al., "On Layer Normalization in the Transformer Architecture", ICML 2020](https://arxiv.org/abs/2002.04745)
