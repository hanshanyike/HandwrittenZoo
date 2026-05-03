# SwiGLU (Swish Gated Linear Unit)

## 算法简介

SwiGLU 是 GLU (Gated Linear Unit) 的 Swish 变体，由 Google 在 PaLM 论文中引入，现已成为 LLaMA、Mistral、Qwen 等主流大模型的标准 FFN 激活单元。它通过门控机制选择性地放大或抑制特征，显著提升了 Transformer 的表达能力。

## 核心思想

GLU 系列的核心是"门控"：将输入通过两个并行的线性投影，一个作为"门"(gate)控制另一个"值"(value)的通过比例。SwiGLU 选用 Swish(SiLU) 作为门控激活函数，公式为：

$$
\text{SwiGLU}(x, W_g, W_v) = \text{Swish}(x W_g) \odot (x W_v)
$$

其中 $\odot$ 为逐元素相乘。门控分支决定哪些信息应该被保留，值分支提供原始信息，两者相乘实现自适应特征选择。

## 数学公式

### SwiGLU 门控激活
$$
\text{SwiGLU}(x) = \text{SiLU}(x W_g + b_g) \odot (x W_v + b_v)
$$

### 完整 SwiGLU FFN (含输出投影)
$$
\text{FFN}_{\text{SwiGLU}}(x) = \big(\text{SiLU}(x W_g) \odot (x W_v)\big) W_o
$$

### 参数计数对比
- 标准 ReLU FFN 参数量: $2 \cdot d_{\text{model}} \cdot d_{ff}$
- SwiGLU FFN 参数量: $3 \cdot d_{\text{model}} \cdot d_{ff}$ (多一个门控投影)

为了保持总参数量一致，SwiGLU 的中间维度应设为：
$$
d_{ff}^{\text{SwiGLU}} = \frac{2}{3} \cdot d_{ff}^{\text{ReLU}}
$$

## 时间/空间复杂度

- **时间复杂度**: $O(n)$ — 门控激活为逐元素操作
- **空间复杂度**: $O(n)$ — 需存储 gate/value 中间张量
- **与 ReLU FFN 对比**: 多一次矩阵乘法（gate 投影），前向计算量约为 1.5 倍；若调整 $d_{ff}$ 保持参数量一致，则计算量大致相同

## 面试高频考点

1. **问题**: SwiGLU 为什么需要三个矩阵？与标准 FFN 的参数量差异是多少？
   **答案**: SwiGLU 包含 gate 投影 $W_g$、value 投影 $W_v$ 和输出投影 $W_o$ 三个矩阵。标准 ReLU FFN 只有 $W_1$ 和 $W_2$ 两个矩阵。若中间维度相同，SwiGLU 参数量是 ReLU FFN 的 1.5 倍（$3/2$）。

2. **问题**: LLaMA 的 SwiGLU 中间维度为什么是 11008，而不是 16384？
   **答案**: 这是经典的参数计数陷阱。LLaMA 的 hidden_size 为 4096，传统 FFN 的 $d_{ff}$ 通常为 $4 \times 4096 = 16384$。但 SwiGLU 有三个矩阵，为了保持与标准两矩阵 FFN 的参数量一致，需令 $d_{ff}^{\text{SwiGLU}} = \frac{2}{3} \times 16384 \approx 10922.67$，实际取整为 11008。验证：$3 \times 4096 \times 11008 \approx 2 \times 4096 \times 16384$。

3. **问题**: 为什么 SwiGLU 比 ReLU/GELU 效果更好？
   **答案**: 三个原因：(1) 门控机制实现自适应特征选择，表达能力更强；(2) Swish 激活平滑且非单调，负区间保留梯度；(3) 双分支结构增加了网络的路径多样性，类似隐式集成效果。PaLM 论文实验显示 SwiGLU 显著优于 ReLU、GELU 和纯 Swish。

4. **问题**: SwiGLU 中的 bias 为什么通常设为 False？
   **答案**: LLaMA 等模型在 Linear 层中普遍使用 `bias=False`，原因包括：(1) 减少参数量（大模型中累积效果显著）；(2) LayerNorm 后的数据均值为零，偏置项作用被削弱；(3) 简化计算图，提升训练效率。

5. **问题**: 如果面试官说"SwiGLU 有 4 个矩阵"，你怎么反驳？
   **答案**: 需区分"纯 SwiGLU 激活"和"完整 FFN"。纯 SwiGLU 门控确实只有两个矩阵（$W_g$ 和 $W_v$），但完整的 FFN 块还需要一个输出投影 $W_o$ 将维度映射回 $d_{model}$，因此共三个矩阵。不存在第四个矩阵，除非将输入的 Q/K/V 投影误算进来。

## 代码解析

### SwiGLU 门控激活
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate = torch.nn.functional.silu(self.gate_proj(x))
    value = self.value_proj(x)
    return gate * value
```
`gate_proj` 和 `value_proj` 是两个独立的线性层。gate 分支经过 SiLU 激活后与 value 分支逐元素相乘，形成门控输出。

### 参数计数辅助函数
```python
def compute_swiglu_ffn_params(d_model: int, d_ff: int, bias: bool = False) -> int:
    mat_params = 3 * d_model * d_ff
    bias_params = 3 * d_ff if bias else 0
    return mat_params + bias_params
```
显式计算三个矩阵的参数量，用于面试中快速验证。若要与 ReLU FFN 参数量对齐，令 $d_{ff} = \frac{2}{3} d_{ff}^{\text{orig}}$。

## 参考资料

- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) — Chowdhery et al., 2022 (首次在大模型中引入 SwiGLU)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer, 2020 (GLU 系列的理论基础)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
- [The llama3.py Code](https://github.com/meta-llama/llama3/blob/main/llama/model.py) — Meta LLaMA3 官方实现
