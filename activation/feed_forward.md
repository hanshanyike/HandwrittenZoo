# Feed-Forward Network (FFN) 对比实现

## 算法简介

FFN (Feed-Forward Network) 是 Transformer 编码器/解码器层中除注意力外的另一核心组件。它对每个位置独立地进行非线性变换，起到"特征提炼"的作用。当前主流实现分为两代：标准两矩阵 FFN (ReLU/GELU) 与门控三矩阵 FFN (SwiGLU)。

## 核心思想

- **标准 FFN**: 遵循 Transformer 原始论文设计，先升维再降维，中间夹一个激活函数。结构简单、参数量可控，是 BERT、GPT-2 等模型的选择。
- **SwiGLU FFN**: 引入门控机制，用两个并行投影替代单一投影，通过 Swish 门控实现自适应特征选择。表达能力更强，但参数量增加 50%，需压缩中间维度以保持总参数量一致。是 LLaMA、PaLM、Mistral 等现代大模型的标准配置。

## 数学公式

### 标准 ReLU FFN
$$
\text{FFN}_{\text{ReLU}}(x) = \max(0, x W_1 + b_1) W_2 + b_2
$$

### 标准 GELU FFN
$$
\text{FFN}_{\text{GELU}}(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2
$$

### SwiGLU FFN
$$
\text{FFN}_{\text{SwiGLU}}(x) = \big(\text{SiLU}(x W_g + b_g) \odot (x W_v + b_v)\big) W_o + b_o
$$

### 参数量对比 (忽略偏置)
| 类型 | 参数量 | 保持一致的 $d_{ff}$ |
|------|--------|-------------------|
| ReLU/GELU FFN | $2 \cdot d_{\text{model}} \cdot d_{ff}$ | $d_{ff}$ |
| SwiGLU FFN | $3 \cdot d_{\text{model}} \cdot d_{ff}$ | $\frac{2}{3} \cdot d_{ff}$ |

## 时间/空间复杂度

- **标准 FFN**
  - 时间: $O(B \cdot L \cdot d_{\text{model}} \cdot d_{ff})$，其中 $B$ 为 batch size，$L$ 为序列长度
  - 空间: $O(B \cdot L \cdot d_{ff})$ — 中间激活张量

- **SwiGLU FFN**
  - 时间: $O(B \cdot L \cdot d_{\text{model}} \cdot d_{ff})$
  - 空间: $O(B \cdot L \cdot d_{ff})$ (gate + value 两个中间张量，但调整 $d_{ff}$ 后总空间相近)

## 面试高频考点

1. **问题**: Transformer 的 FFN 为什么先升维再降维？
   **答案**: 升维 ($d_{\text{model}} \to d_{ff}$) 将特征映射到高维空间，增加非线性表达能力；降维 ($d_{ff} \to d_{\text{model}}$) 将提炼后的特征映射回原始维度，与残差连接兼容。$d_{ff}$ 通常为 $4 \times d_{\text{model}}$，是经验上的效率与效果平衡点。

2. **问题**: FFN 与 Attention 的分工是什么？
   **答案**: Attention 负责"全局信息整合"（不同位置之间的交互），FFN 负责"局部特征变换"（每个位置独立进行非线性映射）。两者互补：Attention 提取上下文关系，FFN 增强特征表达能力。

3. **问题**: 为什么 SwiGLU FFN 要设置 $d_{ff} = \frac{2}{3} \times 4d_{\text{model}}$？
   **答案**: 标准 FFN 有 2 个矩阵，SwiGLU 有 3 个矩阵。若 $d_{ff}$ 相同，SwiGLU 参数量多 50%。为保持与标准 FFN 参数量一致（公平对比），需满足 $3 \cdot d_{\text{model}} \cdot d_{ff}^{\text{SwiGLU}} = 2 \cdot d_{\text{model}} \cdot d_{ff}^{\text{std}}$，解得 $d_{ff}^{\text{SwiGLU}} = \frac{2}{3} d_{ff}^{\text{std}}$。

4. **问题**: 现代大模型为什么普遍使用 bias=False？
   **答案**: (1) 节省参数量：每个 Linear 层省去 $d_{out}$ 个参数，在百亿/千亿参数模型中累积显著；(2) LayerNorm 后的数据均值为零，偏置项的平移作用被削弱；(3) 简化计算图，有利于硬件加速和量化部署。

5. **问题**: 如果让你设计一个 7B 参数的模型，FFN 选 ReLU 还是 SwiGLU？
   **答案**: 选 SwiGLU。PaLM、LLaMA 系列的大量实验已证明，在相同参数量下 SwiGLU 的下游任务表现显著优于 ReLU/GELU。虽然实现略复杂，但收益明确。唯一需注意是正确设置 $d_{ff}$ 以保持总参数量预算。

## 代码解析

### ReLU FFN
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    hidden = F.relu(self.fc1(x))
    hidden = self.dropout(hidden)
    return self.fc2(hidden)
```
两个线性层夹 ReLU，最原始的 Transformer 设计。`fc1` 升维，`fc2` 降维。

### SwiGLU FFN
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    gate = F.silu(self.gate_proj(x))
    value = self.value_proj(x)
    hidden = gate * value
    hidden = self.dropout(hidden)
    return self.out_proj(hidden)
```
三个线性层：`gate_proj` 和 `value_proj` 并行，输出逐元素相乘后经 `out_proj` 降维。`bias=False` 是现代大模型的默认选择。

### 参数量对齐工具
```python
def create_matched_swiglu(d_model: int, d_ff_standard: int, **kwargs) -> FeedForwardSwiGLU:
    d_ff_swiglu = int(2 * d_ff_standard / 3)
    return FeedForwardSwiGLU(d_model=d_model, d_ff=d_ff_swiglu, **kwargs)
```
面试中常考的参数计数陷阱。此函数自动计算 SwiGLU 应使用的中间维度，确保与标准 FFN 参数量一致。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017 (Transformer 原始论文，标准 FFN)
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer, 2020 (GLU 系列理论)
- [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) — Chowdhery et al., 2022 (SwiGLU 在大模型中的首次应用)
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
