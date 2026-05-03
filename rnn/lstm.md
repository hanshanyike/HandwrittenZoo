# LSTM (Long Short-Term Memory)

## 算法简介
LSTM（长短期记忆网络）是由 Hochreiter 和 Schmidhuber 于 1997 年提出的循环神经网络变体。它通过引入**细胞状态（Cell State）**和**三门控机制**（遗忘门、输入门、输出门），有效解决了传统 RNN 的梯度消失/爆炸问题，成为序列建模领域最具影响力的架构之一，广泛应用于机器翻译、语音识别、文本生成等任务。

## 核心思想
传统 RNN 的隐藏状态在每个时间步被完全覆盖更新，导致长距离信息难以保留。LSTM 的核心设计是：
1. **细胞状态（Cell State）**：一条贯穿时间序列的"信息高速公路"，仅通过线性操作（加法和乘法）进行更新，梯度可以相对完整地反向传播。
2. **遗忘门（Forget Gate）**：决定从细胞状态中丢弃哪些旧信息。
3. **输入门（Input Gate）**：决定哪些新候选信息将被存储到细胞状态。
4. **输出门（Output Gate）**：决定细胞状态的哪些部分将输出到隐藏状态。

这种"选择性记忆"机制使得 LSTM 能够学习在数千个时间步的距离上保持和访问信息。

## 数学公式

### 门控计算
给定当前输入 $\mathbf{x}_t$ 和上一时刻隐藏状态 $\mathbf{h}_{t-1}$：

$$\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(输入门)}$$
$$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(遗忘门)}$$
$$\mathbf{g}_t = \tanh(\mathbf{W}_g \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_g) \quad \text{(候选状态)}$$
$$\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(输出门)}$$

### 状态更新
$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t \quad \text{(细胞状态)}$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t) \quad \text{(隐藏状态)}$$

其中 $\odot$ 表示逐元素乘法（Hadamard 积），$\sigma$ 为 sigmoid 函数。

## 时间/空间复杂度
- **时间复杂度**: $O(T \cdot (4 \cdot H \cdot (H + I)))$，其中 $T$ 为序列长度，$H$ 为隐藏维度，$I$ 为输入维度
- **空间复杂度**: $O(T \cdot H)$，存储每个时间步的隐藏状态和细胞状态
- **参数量**: $4 \cdot (I + H) \cdot H + 4 \cdot H$（四个门的权重和偏置）
- **与替代方案对比**:
  - 传统 RNN: 参数量 $O((I+H) \cdot H)$，但梯度消失问题严重
  - GRU: 参数量约 LSTM 的 75%（只有重置门和更新门），性能相近但计算更快
  - Transformer: $O(T^2)$ 时间复杂度，可并行，但长序列内存开销大

## 面试高频考点

1. **LSTM 如何解决传统 RNN 的梯度消失问题？**
   **答案**: LSTM 通过**细胞状态**（Cell State）引入了一条梯度传播的"高速公路"。细胞状态的更新是线性操作（加法和乘法），没有非线性激活函数的连乘，因此梯度在反向传播时不会因为连乘而指数级衰减。遗忘门控制梯度衰减程度，当遗忘门接近 1 时，梯度可以几乎无损地回传。

2. **LSTM 的三个门分别起什么作用？**
   **答案**:
   - **遗忘门** $f_t$：决定从旧细胞状态 $C_{t-1}$ 中保留多少信息（0 = 全部遗忘，1 = 全部保留）
   - **输入门** $i_t$：决定新候选信息 $g_t$ 有多少被写入细胞状态
   - **输出门** $o_t$：决定细胞状态 $C_t$ 的哪些部分被输出到隐藏状态 $h_t$
   三者协同工作，实现了对信息的"选择性遗忘、选择性写入、选择性输出"。

3. **为什么 LSTM 使用 sigmoid 和 tanh 两种激活函数？**
   **答案**: **sigmoid** 输出范围 (0, 1)，适合作为"门控"——控制信息通过的比例。**tanh** 输出范围 (-1, 1)，适合作为"候选内容"——表示信息的实际内容（可正可负）。细胞状态更新公式中，$f_t$ 和 $i_t$ 用 sigmoid 控制比例，$g_t$ 用 tanh 生成内容，$C_t$ 的输出用 tanh 进行非线性压缩。

4. **LSTM 的参数量为什么是传统 RNN 的 4 倍？**
   **答案**: LSTM 有四个独立的门/状态（输入门、遗忘门、候选状态、输出门），每个都有自己的权重矩阵和偏置。传统 RNN 只有一个隐藏状态更新。因此 LSTM 的参数量约为 $4 \times ((I+H) \cdot H + H)$，即传统 RNN 的约 4 倍。

5. **LSTM 中遗忘门偏置初始化为 1 的作用？**
   **答案**: 这是 Jozefowicz 等人（2015）提出的训练技巧。遗忘门偏置初始化为 1 意味着初始阶段遗忘门输出接近 1（即"几乎全部保留"），防止模型在训练初期就遗忘掉所有历史信息，帮助梯度更稳定地传播，加速收敛。

6. **LSTM 和 GRU 的主要区别？**
   **答案**:
   - **门数量**：LSTM 有 3 个门（遗忘、输入、输出），GRU 有 2 个门（重置、更新）
   - **状态**：LSTM 分离了细胞状态和隐藏状态；GRU 将两者合并为单一隐藏状态
   - **参数量**：GRU 参数量约为 LSTM 的 75%
   - **性能**：在大多数任务上两者性能相近，GRU 训练更快，LSTM 在需要精细控制的长序列任务上可能更优

## 代码解析
- `LSTMCell`: 单个 LSTM 单元的从零实现，包含四个门的计算和状态更新。
- `weight_ih` / `weight_hh`: 输入到隐藏、隐藏到隐藏的权重，通过 `torch.cat` 合并后一次性计算四个门。
- `chunk(4, dim=-1)`: 将拼接后的门控预激活值切分为输入门、遗忘门、候选状态、输出门。
- `bias[hidden_size:2*hidden_size].fill_(1.0)`: 遗忘门偏置初始化为 1，防止初期遗忘过多。
- `LSTM`: 完整多层 LSTM 实现，支持层间 Dropout 和可选的输出投影。
- `nn.LSTM` 对比测试：验证自定义实现与 PyTorch 官方实现在相同参数下的数值一致性。

## 参考资料
- [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) (Hochreiter & Schmidhuber, Neural Computation 1997)
- [An Empirical Exploration of Recurrent Network Architectures](https://arxiv.org/abs/1503.04069) (Jozefowicz et al., ICML 2015) —— 遗忘门偏置初始化技巧
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (Christopher Olah, 2015)
- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
