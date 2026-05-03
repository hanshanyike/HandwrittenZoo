# Attention with Linear Biases (ALiBi, 线性偏置注意力)

## 算法简介

ALiBi 是 2021 年提出的相对位置编码方案，被 MPT、BLOOM、Baichuan 等模型采用。与正弦编码、RoPE 不同，ALiBi **不修改词嵌入或 Q/K 向量**，而是直接在 attention score（即 $QK^T / \sqrt{d_k}$）上添加一个与 query-key 距离成线性关系的负偏置。距离越远的 key 受到的惩罚越大，softmax 后的注意力权重自然衰减，从而隐式编码位置信息。

## 核心思想

传统位置编码的核心假设是：位置信息必须先注入到输入表示中，才能在注意力中发挥作用。ALiBi 挑战了这一假设：

> 位置信息只需要影响 attention 的分布，因此可以直接在 attention score 上操作，无需修改输入嵌入。

具体设计：
1. **线性偏置**：对于 query 位置 $i$ 和 key 位置 $j$，添加偏置 $m \cdot (-|i-j|)$。距离 $|i-j|$ 越大，偏置越负。
2. **多头斜率**：每个 attention head 使用不同的固定斜率 $m_h$，使不同 head 关注不同尺度的位置模式。
3. **非学习参数**：斜率是固定的数学函数，不增加可训练参数，减少过拟合。
4. **长度外推**：由于偏置是距离的连续函数，对训练时未见过的更长序列具有优异的外推能力。

## 数学公式

### 1. Attention Score with ALiBi

标准自注意力：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

ALiBi 自注意力：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right)V
$$

其中偏置矩阵 $B$ 的元素为：

$$
B_{h,i,j} = m_h \cdot (-|i - j|)
$$

- $h$：attention head 索引
- $i$：query 位置
- $j$：key 位置
- $m_h$：第 $h$ 个 head 的斜率

### 2. 斜率计算

对于 $n$ 个 head，斜率构成等比数列：

$$
m_h = 2^{-\frac{8h}{n}}, \quad h = 1, 2, \dots, n
$$

当 $n$ 不是 2 的幂时，先计算最接近的较小 2 的幂的斜率，再补充剩余 head。

### 3. 因果注意力中的 ALiBi

在自回归模型中，结合因果 mask（上三角为 $-\infty$）：

$$
B_{h,i,j} =
\begin{cases}
m_h \cdot (-|i - j|), & i \ge j \\
-\infty, & i < j
\end{cases}
$$

## 时间/空间复杂度

- **时间复杂度**：生成偏置矩阵 $O(n_{\text{heads}} \cdot \text{seq_len}^2)$；与标准 attention 同阶
- **空间复杂度**：$O(n_{\text{heads}} \cdot \text{max_seq_len}^2)$ 缓存，或动态生成
- **与替代方案对比**：
  - 对比正弦编码：无需修改输入嵌入，外推能力显著更强；
  - 对比 RoPE：ALiBi 修改 attention score，RoPE 修改 Q/K 表示；两者正交，可结合使用（如 Baichuan 的 Dynamic NTK-ALiBi）。

## 面试高频考点

1. **问题**：ALiBi 与 RoPE 的本质区别是什么？
   **答案**：ALiBi 直接在 attention score 上添加距离相关的负偏置，不修改 Q/K/V 的表示；RoPE 通过旋转矩阵修改 Q 和 K 的表示，使内积体现相对位置。ALiBi 更简单、外推性更强；RoPE 更灵活，可与 NTK 缩放结合。

2. **问题**：ALiBi 的斜率为什么使用等比数列 $2^{-8h/n}$？
   **答案**：等比数列确保不同 head 的斜率覆盖多个数量级，使不同 head 关注不同尺度的位置模式（有的 head 对远距离敏感，有的只对近距离敏感）。底数 2 和指数 -8 是论文中的经验值，确保斜率范围合理（如 8 head 时从 1/2 到 1/256）。

3. **问题**：为什么 ALiBi 的外推能力比正弦编码强？
   **答案**：正弦编码的每个位置编码是独立的向量，超出训练长度后模型没见过对应的位置向量，分布偏移大。ALiBi 的偏置是距离的连续函数，训练时已经见过各种距离（只是最大距离受限于训练长度），因此推理时更长的距离只是函数的连续延伸，分布偏移小。

4. **问题**：ALiBi 的偏置是在 softmax 前还是 softmax 后添加？
   **答案**：必须在 softmax **前**添加。因为 softmax 是单调函数，偏置直接影响概率分布。如果在 softmax 后添加，会破坏概率归一性，且无法通过梯度有效学习（虽然 ALiBi 的斜率是固定的）。

5. **问题**：ALiBi 是否可以与 RoPE 同时使用？
   **答案**：理论上可以，但通常不这样做，因为两者都是相对位置编码，功能重叠。Baichuan 等模型使用 ALiBi 而不使用 RoPE。不过有研究尝试结合两者优势，如 Dynamic NTK-RoPE + ALiBi，但这不是主流做法。

## 代码解析

### 1. 斜率计算

```python
def get_slopes_power_of_2(n: int) -> list[float]:
    start = 2 ** (-(2 ** -(math.log2(n) - 3)))
    ratio = start
    return [start * (ratio ** i) for i in range(n)]
```

- 公式推导：$\text{start} = 2^{-(2^{- (\log_2 n - 3)})} = 2^{-8/n}$
- 等比数列确保不同 head 的斜率覆盖多个数量级。
- 非 2 的幂时，通过补充偶数索引斜率解决。

### 2. 偏置矩阵生成

```python
distance_matrix = -torch.abs(pos.unsqueeze(1) - pos.unsqueeze(0))
alibi_bias = slopes.view(n_heads, 1, 1) * distance_matrix.unsqueeze(0)
```

- `pos.unsqueeze(1) - pos.unsqueeze(0)` 生成所有位置对的距离矩阵。
- 取负绝对值，确保距离越大值越负。
- 广播乘法：每个 head 用自己的斜率缩放距离矩阵。

### 3. 与 Attention 集成

```python
scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
alibi_bias = self.alibi(seq_len)
scores = scores + alibi_bias.unsqueeze(0)
attn_weights = torch.softmax(scores, dim=-1)
```

- 标准 QK^T 计算后，添加 ALiBi 偏置。
- 再通过 softmax，距离远的 key 自然获得更小权重。
- 无需修改 Q/K/V 的投影逻辑，集成简单。

## 参考资料

- [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (Press et al., 2021)](https://arxiv.org/abs/2108.12409)
- [MPT-7B: MosaicML's Foundation Model](https://www.mosaicml.com/blog/mpt-7b)
- [Hugging Face MPT Modeling](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mpt/modeling_mpt.py)
- [LabML ALiBi Implementation](https://nn.labml.ai/transformers/alibi/index.html)
