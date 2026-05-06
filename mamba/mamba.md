# Mamba: 选择性状态空间模型

## 算法简介

Mamba 是一种新型的线性时间序列建模架构，于 2023 年提出，旨在挑战 Transformer 在长序列建模中的统治地位。它通过引入**选择性机制（Selection Mechanism）**，使状态空间模型（SSM）能够像注意力机制一样，根据输入内容动态决定关注哪些信息，同时保持 $O(n)$ 的线性时间复杂度。

## 核心思想

### 1. 从 S4 到 Mamba 的演进

传统 S4（Structured State Space Sequence Model）的核心问题是：**状态矩阵 A、B、C 是静态的，与输入无关**。这导致模型无法灵活地选择性记住或遗忘信息——对于不同类型的 token（如关键词与停用词），模型只能使用相同的动态。

Mamba 的核心洞察是：**让 SSM 的参数成为输入的函数**，具体通过以下步骤实现：

1. **输入依赖的投影**：用线性变换从输入 $x_t$ 生成 $\Delta_t$（时间步）、$B_t$（输入映射）、$C_t$（输出映射）
2. **选择性扫描**：根据输入动态调整 SSM 的行为，实现内容感知的计算

### 2. 状态空间离散化

原始连续 SSM：
$$x'(t) = A x(t) + B u(t), \quad y(t) = C x(t)$$

通过 Zero-Order Hold 离散化（步长 $\Delta_t$）：
$$x_t = \bar{A}_t x_{t-1} + \bar{B}_t u_t$$
$$y_t = \bar{C}_t x_{t-1}$$

其中：
- $\bar{A}_t = e^{\Delta_t A}$
- $\bar{B}_t = (\Delta_t A)^{-1} (e^{\Delta_t A} - I) \Delta_t B_t$

### 3. 硬件感知的并行扫描

虽然递归形式是 $O(n)$ 的，但串行计算效率低。Mamba 通过 **Parallel Prefix Scan** 实现 GPU 友好的并行计算：

```
输入序列: [x_0, x_1, x_2, x_3, ...]
状态递推: [h_0, h_1, h_2, h_3, ...]
```

利用 GPU 的并行能力，在硬件友好性和计算效率之间取得平衡。

## 数学公式

### 选择性 SSM 前向传播

给定输入序列 $x_1, x_2, ..., x_n$，Mamba 的前向计算为：

$$
\begin{aligned}
\bar{x}_t &= \text{SiLU}(\text{Conv1D}(x_t)) \\
[\Delta_t, B_t, C_t] &= \text{x\_proj}(\bar{x}_t) \\
\bar{A}_t &= e^{\Delta_t A} \\
\bar{B}_t &= \Delta_t B_t \\
h_t &= \bar{A}_t h_{t-1} + \bar{B}_t \bar{x}_t \quad \text{（选择性扫描）} \\
y_t &= C_t h_t + D \cdot \bar{x}_t \quad \text{（跳跃连接）}
\end{aligned}
$$

### 复杂度对比

| 架构 | 时间复杂度 | 空间复杂度 | 序列长度外推 |
|------|-----------|------------|-------------|
| Transformer | $O(n^2)$ | $O(n^2)$ | 困难 |
| Mamba | $O(n)$ | $O(n \cdot d_{state})$ | 优秀 |
| Linear Attention | $O(n)$ | $O(n \cdot d)$ | 中等 |

## 时间/空间复杂度

- **时间复杂度**：$O(n \cdot d_{model} \cdot d_{state})$，其中 $n$ 是序列长度，$d_{model}$ 是模型维度，$d_{state}$ 是 SSM 状态维度。
- **空间复杂度**：$O(n \cdot d_{state})$，需存储隐状态。
- **推理优势**：递归形式天然支持 KV Cache 的增量计算，无需存储完整的注意力矩阵。

## 面试高频考点

1. **问题**：Mamba 相比 Transformer 的核心优势是什么？
   **答案**：线性时间复杂度（$O(n)$ vs $O(n^2)$），推理时显存占用更小，序列长度外推能力更强。

2. **问题**：Mamba 的"选择性"体现在哪里？
   **答案**：体现在 $\Delta_t$、$B_t$、$C_t$ 三个参数由输入动态生成，而非静态。这使得模型可以像注意力机制一样，根据内容选择性关注/忽略信息。

3. **问题**：为什么 Mamba 需要局部卷积（d_conv）？
   **答案**：SSM 的递归特性使其难以捕捉局部模式。Mamba 在 SSM 之前添加因果卷积，帮助模型同时捕获局部信息和长程依赖。

4. **问题**：Mamba 与 Linear Attention 的区别？
   **答案**：Linear Attention 通过核函数将 softmax 解耦；Mamba 通过选择性机制和硬件感知的并行扫描，在保持表达能力的同时实现线性复杂度。两者都声称 $O(n)$，但 Mamba 的实际建模能力更接近 Transformer。

5. **问题**：Mamba 的残差连接和 RMSNorm 作用？
   **答案**：残差连接帮助深层模型的梯度流动；RMSNorm 在每个 MambaBlock 的输出处进行归一化，稳定训练。

## 代码解析

### 选择性扫描实现

```python
def _selective_scan(self, x, dt, A, B, C, D):
    delta = F.softplus(dt)  # 确保 dt > 0

    # 离散化状态矩阵
    dA = torch.exp(delta.unsqueeze(-1) * A)
    dB_u = delta.unsqueeze(-1) * x.unsqueeze(-1) * B.unsqueeze(2)

    # 递归计算隐状态
    h = torch.zeros(batch, d_inner, d_state, device=x.device)
    ys = []
    for i in range(seq_len):
        h = dA[:, i] * h + dB_u[:, i]  # 状态更新
        y = torch.einsum("bdn,bn->bd", h, C[:, i])  # 投影到输出
        ys.append(y)

    y = torch.stack(ys, dim=1)
    y = y + x * D  # 跳跃连接
    return y
```

**关键点**：
- `softplus` 确保时间步 $\Delta_t > 0$
- 状态更新公式来自连续 SSM 的离散化
- 跳跃连接 $D$ 是一个静态参数，类似残差分支

### 整体前向传播

```python
def forward(self, x):
    # 1. 输入投影 + 门控
    xz = self.in_proj(x)
    x_inner, z = xz.chunk(2, dim=-1)

    # 2. 局部卷积（捕获 n-gram 模式）
    x_conv = self.conv1d(x_inner.transpose(1, 2))
    x_conv = x_conv.transpose(1, 2)[:, :seq_len, :]
    x_ssm = self.act(x_conv)

    # 3. 选择性 SSM
    x_dbl = self.x_proj(x_ssm)
    dt, B, C = x_dbl.split([self.dt_rank, d_state, d_state], dim=-1)
    dt = self.dt_proj(dt)
    y = self._selective_scan(x_ssm, dt, A, B, C, D)

    # 4. 门控 + 输出投影
    y = y * F.silu(z)
    return self.out_proj(y)
```

## 参考资料

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Mamba-minimal](https://github.com/state-spaces/mamba) — 官方简化实现
- [Hungry Hungry Hippos](https://arxiv.org/abs/2310.01889) — 相关的线性注意力工作
