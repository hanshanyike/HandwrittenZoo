# Rotary Position Embedding (RoPE, 旋转位置编码)

## 算法简介

RoPE 是苏剑林于 2021 年提出的相对位置编码方案，被 LLaMA、Mistral、Qwen、Baichuan 等主流大模型广泛采用。它通过**旋转矩阵**对 Query 和 Key 向量进行位置相关的旋转变换，使得旋转后的 Q 与 K 的内积仅依赖于两者的相对距离，从而将绝对位置信息编码进内积运算，同时自然显式表达相对位置依赖。

## 核心思想

传统绝对位置编码（如正弦编码）将位置向量加到词嵌入上，但自注意力的内积运算 $QK^T$ 无法直接体现两个 token 的相对距离。RoPE 的核心洞察来自数学观察：

> 如果我们将二维向量 $(x, y)$ 旋转角度 $\theta$，旋转后的向量与另一旋转向量的内积，仅取决于两者旋转角度的差值。

推广到高维：将 $d$ 维向量分成 $d/2$ 个二维子空间，每个子空间独立旋转不同频率的角度。这样，位置 $m$ 的 Query 与位置 $n$ 的 Key 的内积，严格只与相对距离 $(m-n)$ 有关，与绝对位置 $m, n$ 无关。

这一性质称为**相对位置编码的内积不变性**，是 RoPE 成为现代 LLM 标配的根本原因。

## 数学公式

### 1. 旋转角度预计算

对于维度组 $i = 0, 1, \dots, d/2 - 1$，基础频率为：

$$
\theta_i = \text{base}^{-2i/d}
$$

位置 $m$ 的旋转角度为：

$$
\Theta_m = [m \cdot \theta_0, m \cdot \theta_1, \dots, m \cdot \theta_{d/2-1}]
$$

### 2. 旋转变换

将向量 $x \in \mathbb{R}^d$ 分成 $d/2$ 个二维子向量 $(x_{2i}, x_{2i+1})$，每个子向量旋转角度 $m \cdot \theta_i$：

$$
\begin{pmatrix} x_{2i}^{(m)} \\ x_{2i+1}^{(m)} \end{pmatrix}
=
\begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}
\begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}
$$

### 3. 复数表示（工程实现常用）

将二维子向量视为复数 $z_i = x_{2i} + i \cdot x_{2i+1}$，旋转等价于乘以复指数：

$$
z_i^{(m)} = z_i \cdot e^{i \cdot m\theta_i} = z_i \cdot (\cos(m\theta_i) + i\sin(m\theta_i))
$$

### 4. 相对位置不变性

对于位置 $m$ 的 Query $q^{(m)}$ 和位置 $n$ 的 Key $k^{(n)}$，有：

$$
\langle R(m)q, R(n)k \rangle = \langle R(m-n)q, k \rangle = g(q, k, m-n)
$$

即内积严格只与相对距离 $(m-n)$ 有关。

## 时间/空间复杂度

- **时间复杂度**：预计算 $O(\text{max_seq_len} \cdot \text{head_dim})$；前向传播 $O(\text{batch} \cdot \text{seq_len} \cdot \text{num_heads} \cdot \text{head_dim})$
- **空间复杂度**：$O(\text{max_seq_len} \cdot \text{head_dim})$ 缓存 $\cos/\sin$ 值
- **与替代方案对比**：
  - 对比正弦编码：RoPE 显式建模相对位置，长文本外推能力更强；
  - 对比 ALiBi：RoPE 修改的是 Q/K 表示，ALiBi 修改的是 attention score；两者可结合使用（如 Dynamic NTK-RoPE）。

## 面试高频考点

1. **问题**：RoPE 与正弦位置编码的本质区别是什么？
   **答案**：正弦编码是绝对位置编码，将位置向量加到词嵌入上；RoPE 是相对位置编码，通过旋转矩阵修改 Q/K 向量，使得 $Q_m$ 与 $K_n$ 的内积仅依赖于 $(m-n)$ 的相对距离。RoPE 不修改 Value，也不直接修改 attention score。

2. **问题**：为什么 RoPE 使用复数乘法实现旋转？
   **答案**：将二维子向量 $(x, y)$ 视为复数 $x+iy$，乘以 $e^{i\theta} = \cos\theta + i\sin\theta$ 等价于二维旋转矩阵作用。复数乘法在工程上可以通过 PyTorch 的 `view_as_complex` 高效实现，避免显式构造旋转矩阵。

3. **问题**：RoPE 的 base（默认 10000）有什么作用？如果增大 base 会怎样？
   **答案**：base 控制旋转角度的频率范围。base 越大，高频分量越少，旋转角度变化越缓慢。增大 base 可以增强长文本外推能力（如 CodeLLaMA 使用 base=1000000），因为远距离 token 的旋转角度差异更小，注意力分布更平滑。

4. **问题**：什么是 NTK-aware 缩放 / Dynamic NTK？与 RoPE 的关系？
   **答案**：当推理序列长度超过训练长度时，直接外推 RoPE 会导致注意力分布崩溃（角度超出训练分布）。NTK-aware 缩放通过动态调整 base 值，将长序列的位置信息"压缩"到训练时见过的角度范围内，从而无需微调即可处理更长上下文。

5. **问题**：RoPE 为什么只应用于 Q 和 K，而不应用于 V？
   **答案**：自注意力的输出是 $\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$。位置信息只需要在 $QK^T$ 的内积中体现，以决定 attention weight 的分布。Value 向量只负责提供内容信息，其加权平均不需要位置信息。对 V 应用 RoPE 不会带来额外收益，反而增加计算量。

## 代码解析

### 1. 预计算 freqs_cis

```python
freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)] / dim))
t = torch.arange(end, dtype=torch.float32)
freqs = torch.outer(t, freqs)
freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
```

- `torch.arange(0, dim, 2)` 生成维度组索引，共 `dim//2` 组。
- `torch.outer(t, freqs)` 计算每个位置、每个维度组的旋转角度。
- `torch.polar` 将角度转换为复数 $e^{i\theta}$，便于后续复数乘法旋转。

### 2. apply_rotary_emb

```python
xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
freqs_cis = reshape_for_broadcast(freqs_cis, xq)
xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
```

- `reshape(..., -1, 2)` 将最后一维拆成 `[实部, 虚部]` 对。
- `view_as_complex` 将实数张量视为复数张量。
- 复数乘法 `xq_ * freqs_cis` 实现旋转。
- `view_as_real` 转回实数，`flatten` 恢复原始维度。

### 3. 广播机制

```python
shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
shape[-1] = x.shape[-1] // 2
return freqs_cis.view(*shape)
```

- 将 `(seq_len, head_dim//2)` 的 freqs_cis 扩展为 `(1, seq_len, 1, ..., head_dim//2)`。
- 使其能与 `(batch, seq_len, num_heads, head_dim//2)` 的复数 Q/K 进行广播乘法。

## 参考资料

- [RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al., 2021)](https://arxiv.org/abs/2104.09864)
- [LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971)
- [PyTorch torchtune RotaryPositionalEmbeddings](https://docs.pytorch.org/torchtune/main/generated/torchtune.modules.RotaryPositionalEmbeddings.html)
- [NTK-Aware Scaled RoPE (bloc97, 2023)](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
