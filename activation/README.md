# Activation & Feed-Forward Networks

本目录涵盖深度学习与大模型中最核心的激活函数及前馈网络实现，从基础 ReLU 到现代 SwiGLU，完整覆盖面试与工程实践中的高频考点。

## 文件结构

| 文件 | 内容 |
|------|------|
| [activations.py](activations.py) + [activations.md](activations.md) | ReLU、GELU、Swish(SiLU) 激活函数从零实现 |
| [swiglu.py](swiglu.py) + [swiglu.md](swiglu.md) | SwiGLU 门控激活及参数计数陷阱详解 |
| [feed_forward.py](feed_forward.py) + [feed_forward.md](feed_forward.md) | 标准 FFN vs SwiGLU FFN 对比实现 |

## 面试聚焦点

### 1. 激活函数演进 (activations)
- **ReLU 的缺陷**: 负区间神经元死亡，如何解决？
- **GELU 的优势**: 为什么 Transformer 用 GELU 而非 ReLU？
- **Swish vs GELU**: 两者都是平滑非单调，设计动机有何不同？

### 2. SwiGLU 参数计数陷阱 (swiglu) — 极高频
- **核心陷阱**: SwiGLU 有三个矩阵，标准 FFN 只有两个。若 $d_{ff}$ 相同，参数量多 50%。
- **正确做法**: 令 $d_{ff}^{\text{SwiGLU}} = \frac{2}{3} d_{ff}^{\text{std}}$ 以保持总参数量一致。
- **实战验证**: LLaMA hidden_size=4096，标准 $d_{ff}=16384$，SwiGLU 实际使用 $11008 \approx \frac{2}{3} \times 16384$。
- **面试话术**: "SwiGLU 的参数量是 $3 \cdot d_{model} \cdot d_{ff}$，要与标准 FFN 的 $2 \cdot d_{model} \cdot d_{ff}$ 对齐，中间维度需压缩为 $2/3$。"

### 3. FFN 架构选型 (feed_forward)
- **标准 FFN**: 两矩阵 + ReLU/GELU，BERT/GPT-2 使用。
- **SwiGLU FFN**: 三矩阵 + 门控，LLaMA/PaLM/Mistral 使用，表达力更强。
- **bias=False**: 现代大模型普遍去掉偏置，减少参数量并简化计算图。
- **分工理解**: Attention 负责全局交互，FFN 负责局部非线性变换。

## 快速验证

```bash
cd d:\Code\Python_Workplace\HandwrittenZoo\activation
python activations.py
python swiglu.py
python feed_forward.py
```

## 关键公式速查

| 组件 | 公式 | 参数量 |
|------|------|--------|
| ReLU | $\max(0, x)$ | — |
| GELU | $0.5x(1 + \text{erf}(x/\sqrt{2}))$ | — |
| Swish | $x \cdot \sigma(x)$ | — |
| ReLU FFN | $\max(0, xW_1)W_2$ | $2 \cdot d_{m} \cdot d_{ff}$ |
| SwiGLU FFN | $(\text{SiLU}(xW_g) \odot xW_v) W_o$ | $3 \cdot d_{m} \cdot d_{ff}$ |

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) — Hendrycks & Gimpel, 2016
- [Searching for Activation Functions](https://arxiv.org/abs/1710.05941) — Ramachandran et al., 2017
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer, 2020
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) — Chowdhery et al., 2022
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
