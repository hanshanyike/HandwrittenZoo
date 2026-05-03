# GRU (Gated Recurrent Unit)

## 算法简介
GRU（门控循环单元）是由 Cho 等人于 2014 年提出的循环神经网络变体。它将 LSTM 的三门控机制简化为两门控（重置门、更新门），并合并细胞状态与隐藏状态，在保持长程依赖建模能力的同时减少了参数量和计算开销，是 LSTM 最主流的轻量级替代方案。

## 核心思想
LSTM 通过细胞状态和三个门控实现了对长程依赖的有效建模，但参数量较大（约为传统 RNN 的 4 倍）。GRU 的核心洞察是：
1. **状态合并**：将 LSTM 中分离的细胞状态和隐藏状态合并为单一的隐藏状态，简化架构。
2. **两门控设计**：
   - **重置门（Reset Gate）**：控制前一时刻隐藏状态有多少参与当前候选状态的计算。
   - **更新门（Update Gate）**：同时承担 LSTM 中遗忘门和输入门的角色，控制前一时刻状态有多少被保留。
3. **效率与效果平衡**：GRU 参数量约为 LSTM 的 75%，在多数 NLP 任务上性能与 LSTM 相当甚至更优。

## 数学公式

### 门控计算
$$\mathbf{r}_t = \sigma(\mathbf{W}_{ir} \mathbf{x}_t + \mathbf{W}_{hr} \mathbf{h}_{t-1} + \mathbf{b}_r) \quad \text{(重置门)}$$
$$\mathbf{z}_t = \sigma(\mathbf{W}_{iz} \mathbf{x}_t + \mathbf{W}_{hz} \mathbf{h}_{t-1} + \mathbf{b}_z) \quad \text{(更新门)}$$

### 候选状态与隐藏状态
$$\mathbf{g}_t = \tanh(\mathbf{W}_{ig} \mathbf{x}_t + \mathbf{r}_t \odot (\mathbf{W}_{hg} \mathbf{h}_{t-1}) + \mathbf{b}_g) \quad \text{(候选状态)}$$
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{g}_t + \mathbf{z}_t \odot \mathbf{h}_{t-1} \quad \text{(隐藏状态)}$$

其中 $\odot$ 表示逐元素乘法，$\sigma$ 为 sigmoid 函数。

### 与 LSTM 的对应关系
| LSTM | GRU |
|------|-----|
| 遗忘门 $f_t$ + 输入门 $i_t$ | 更新门 $z_t$ |
| 输出门 $o_t$ | 无（隐藏状态直接输出） |
| 细胞状态 $C_t$ | 合并到隐藏状态 $h_t$ |
| 4 套权重 | 3 套权重 |

## 时间/空间复杂度
- **时间复杂度**: $O(T \cdot (3 \cdot H \cdot (H + I)))$，其中 $T$ 为序列长度，$H$ 为隐藏维度，$I$ 为输入维度
- **空间复杂度**: $O(T \cdot H)$，存储每个时间步的隐藏状态
- **参数量**: $3 \cdot (I + H) \cdot H + 3 \cdot H$，约为 LSTM 的 75%
- **与替代方案对比**:
  - LSTM: 参数量多 33%，三门控分离设计，某些长序列任务上更稳定
  - 传统 RNN: 参数量少，但梯度消失问题严重
  - Transformer: 可并行，但 $O(T^2)$ 复杂度，短序列上 RNN 类可能更快

## 面试高频考点

1. **GRU 与 LSTM 的核心区别是什么？**
   **答案**:
   - **门数量**：LSTM 有 3 个门（遗忘、输入、输出），GRU 有 2 个门（重置、更新）
   - **状态设计**：LSTM 分离细胞状态 $C_t$ 和隐藏状态 $h_t$；GRU 将两者合并为单一隐藏状态
   - **参数量**：GRU 参数量约为 LSTM 的 75%
   - **更新门的作用**：GRU 的更新门 $z_t$ 同时承担了 LSTM 中遗忘门（控制保留多少旧信息）和输入门（控制接受多少新信息）的功能
   - **性能**：多数任务上两者性能相近，GRU 训练更快，LSTM 在极长序列上可能更稳定

2. **GRU 的重置门和更新门分别起什么作用？**
   **答案**:
   - **重置门** $r_t$：控制前一时刻隐藏状态 $h_{t-1}$ 有多少被用于计算当前候选状态 $g_t$。当 $r_t \to 0$ 时，候选状态几乎完全由当前输入决定，相当于"忘记"历史。
   - **更新门** $z_t$：控制前一时刻隐藏状态有多少被保留到当前时刻。当 $z_t \to 1$ 时，$h_t \approx h_{t-1}$，几乎全部保留历史；当 $z_t \to 0$ 时，$h_t \approx g_t$，几乎全部接受新信息。

3. **为什么 GRU 的参数量比 LSTM 少？**
   **答案**: LSTM 有 4 个独立的门/状态（输入门、遗忘门、候选状态、输出门），每个都有自己的权重矩阵。GRU 只有 3 个（重置门、更新门、候选状态），且没有独立的细胞状态，因此参数量为 $3 \times ((I+H) \cdot H + H)$，约为 LSTM 的 75%。

4. **GRU 的更新门如何同时替代 LSTM 的遗忘门和输入门？**
   **答案**: LSTM 的细胞状态更新为 $C_t = f_t \odot C_{t-1} + i_t \odot g_t$，其中 $f_t$ 控制遗忘比例，$i_t$ 控制输入比例。GRU 的隐藏状态更新为 $h_t = (1-z_t) \odot g_t + z_t \odot h_{t-1}$。对比可见：$z_t$ 对应 $f_t$（保留旧信息的比例），$(1-z_t)$ 对应 $i_t$（接受新信息的比例）。GRU 通过 $z_t + (1-z_t) = 1$ 的约束，将两个门合并为一个。

5. **什么场景下选择 GRU 而不是 LSTM？**
   **答案**:
   - **计算资源受限**：GRU 参数量少 25%，推理速度更快
   - **中等长度序列**：GRU 在大多数 NLP 任务（翻译、摘要、分类）上与 LSTM 性能相当
   - **快速原型**：GRU 训练收敛通常更快
   - **选择 LSTM 的场景**：极长序列（>1000 步）、需要精细控制信息流的任务（如某些音频/视频建模）

6. **GRU 如何解决梯度消失问题？**
   **答案**: 与 LSTM 类似，GRU 通过更新门 $z_t$ 控制历史信息的传递。当 $z_t \to 1$ 时，$h_t \approx h_{t-1}$，梯度可以直接沿 $h$ 传播而不经过非线性变换的连乘，形成类似 LSTM 细胞状态的"梯度高速公路"。虽然 GRU 没有独立的细胞状态，但更新门的线性插值机制同样保护了梯度传播。

## 代码解析
- `GRUCell`: 单个 GRU 单元的从零实现，包含重置门、更新门、候选状态计算和状态更新。
- `weight_ir` / `weight_hr`: 重置门和更新门共享的输入/隐藏权重，通过 `chunk(2, dim=-1)` 切分。
- `weight_ig` / `weight_hg`: 候选状态的独立权重，重置门先对历史状态进行筛选 (`r * h_prev`) 再参与计算。
- `h_t = (1 - z) * g + z * h_prev`: 更新门的线性插值，同时控制新旧信息的混合比例。
- `GRU`: 完整多层 GRU 实现，支持层间 Dropout 和可选的输出投影。
- 与 `nn.GRU` 对比测试：验证自定义实现与 PyTorch 官方实现在相同参数下的数值一致性（注意参数顺序的重排）。

## 参考资料
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) (Cho et al., EMNLP 2014)
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555) (Chung et al., 2014) —— GRU vs LSTM 系统对比
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) (Christopher Olah, 2015) —— 同时涵盖 GRU
- [PyTorch GRU Documentation](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
