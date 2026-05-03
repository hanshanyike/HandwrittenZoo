# BiLSTM (Bidirectional LSTM)

## 算法简介
BiLSTM（双向长短期记忆网络）是 LSTM 的扩展形式，由 Schuster 和 Paliwal 于 1997 年提出。它同时训练两个独立的 LSTM：一个按正序处理序列（前向 LSTM），一个按逆序处理序列（后向 LSTM），将两者的隐藏状态融合，使每个时间步的表示同时包含过去和未来的上下文信息，是 NLP 领域最基础且广泛使用的序列编码器之一。

## 核心思想
单向 LSTM 只能利用当前时间步之前的信息（左上下文），无法看到未来的信息（右上下文）。然而在许多任务中，当前词的理解强烈依赖于后续内容：
- **命名实体识别（NER）**："Apple" 在 "Apple is delicious"（水果）和 "Apple launched iPhone"（公司）中的类别不同。
- **情感分析**："这部电影并不差"中，"不" 的否定含义需要看到后面的 "差" 才能确定。

BiLSTM 的核心设计：
1. **前向 LSTM**：按 $t = 0, 1, ..., T-1$ 顺序处理，编码左上下文。
2. **后向 LSTM**：按 $t = T-1, ..., 1, 0$ 逆序处理，编码右上下文。
3. **输出融合**：将两个方向在每个时间步的隐藏状态拼接（concat）或求和（sum）。
4. **参数独立**：两个方向的 LSTM 参数不共享，各自独立学习不同方向的特征。

## 数学公式

### 前向 LSTM
$$\overrightarrow{\mathbf{h}}_t = \text{LSTM}(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1})$$

### 后向 LSTM
$$\overleftarrow{\mathbf{h}}_t = \text{LSTM}(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t+1})$$

### 输出融合
**拼接模式（concat）**：
$$\mathbf{h}_t^{\text{bi}} = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{2H}$$

**求和模式（sum）**：
$$\mathbf{h}_t^{\text{bi}} = \overrightarrow{\mathbf{h}}_t + \overleftarrow{\mathbf{h}}_t \in \mathbb{R}^{H}$$

其中 $[;]$ 表示向量拼接，$H$ 为每个方向的隐藏维度。

## 时间/空间复杂度
- **时间复杂度**: $O(T \cdot (8 \cdot H \cdot (H + I)))$，约为单向 LSTM 的 2 倍
- **空间复杂度**: $O(T \cdot 2 \cdot H)$，存储两个方向的隐藏状态和细胞状态
- **参数量**: $2 \times [4 \cdot (I + H) \cdot H + 4 \cdot H]$，两个独立 LSTM 的参数之和
- **与替代方案对比**:
  - 单向 LSTM: 只能编码左上下文，参数量减半
  - Transformer Encoder: 双向 by design，$O(T^2)$ 复杂度，可并行，但长序列内存开销大
  - CNN: 通过膨胀卷积实现上下文编码，但长距离依赖需要多层堆叠

## 面试高频考点

1. **BiLSTM 与单向 LSTM 的核心区别？**
   **答案**: 单向 LSTM 只能利用当前时间步之前的信息（因果/历史信息），而 BiLSTM 同时利用前后文信息。BiLSTM 包含两个独立的 LSTM：前向 LSTM 按正序处理序列编码左上下文，后向 LSTM 按逆序处理序列编码右上下文。两者的输出在每个时间步进行融合（拼接或求和）。

2. **BiLSTM 的两个方向参数是否共享？为什么？**
   **答案**: **不共享**。前向和后向 LSTM 各自有独立的参数。原因是：两个方向处理的信息流不同（左上下文 vs 右上下文），共享参数会强制两个方向学习相同的特征提取模式，限制了模型能力。独立参数使每个方向可以专门学习该方向特有的语言模式。

3. **BiLSTM 的 concat 和 sum 融合方式各有什么优缺点？**
   **答案**:
   - **concat**：保留两个方向的完整信息，表达能力更强，但输出维度翻倍（$2H$），后续层参数量增加。
   - **sum**：输出维度不变（$H$），计算更高效，但可能损失部分方向性信息（两个方向的信号可能相互抵消）。
   - 实践中 **concat 更常用**，尤其在需要丰富表示的编码器任务中。

4. **BiLSTM 能否用于语言模型（LM）或文本生成任务？**
   **答案**: **不能用于自回归生成**。语言模型需要按顺序生成 token，生成 $t$ 时刻的 token 时只能使用 $t$ 之前的信息。BiLSTM 的后向 LSTM 会"偷看"未来的 token，违反因果性约束。BiLSTM 主要用于**编码任务**（如分类、NER、句法分析），而非**生成任务**。

5. **BiLSTM 如何解决梯度消失问题？**
   **答案**: BiLSTM 继承了 LSTM 的细胞状态机制，每个方向的梯度都可以通过细胞状态的线性路径传播。此外，双向结构提供了两条独立的梯度传播路径（前向和后向），即使某一方向在某些时间步出现梯度衰减，另一方向仍可能保持稳定的梯度流。

6. **BiLSTM 在实际应用中的常见变体有哪些？**
   **答案**:
   - **BiLSTM + CRF**：在序列标注任务（NER、POS）中，BiLSTM 提取特征，CRF 建模标签转移约束。
   - **BiLSTM + Attention**：在文本分类中，对 BiLSTM 输出应用自注意力，提取关键片段。
   - **多层 BiLSTM**：堆叠多层 BiLSTM 提取层次化特征（底层词法、高层语义）。
   - **BiGRU**：用 GRU 替代 LSTM，参数量减少约 25%，速度更快。

7. **为什么后向 LSTM 需要将输入翻转？**
   **答案**: LSTM 的实现天然按时间步顺序处理（$t=0 \to T-1$）。为了让第二个 LSTM 从序列末尾开始处理，我们将输入张量在第 1 维（时间维）翻转 `torch.flip(x, dims=[1])`，使其按 $x_T, x_{T-1}, ..., x_1$ 的顺序输入。前向传播完成后，再将输出翻转回正序，以便与前向 LSTM 的输出按时间步对齐融合。

## 代码解析
- `forward_lstm` / `backward_lstm`: 两个独立的单向 LSTM，参数不共享。
- `torch.flip(x, dims=[1])`: 在时间维度翻转输入，使后向 LSTM 从序列末尾开始处理。
- `merge_mode`: 支持 `"concat"`（拼接，维度翻倍）和 `"sum"`（求和，维度不变）两种融合方式。
- `BiLSTMClassifier`: 基于 BiLSTM 的文本分类器示例，包含 Embedding -> BiLSTM -> 全局池化 -> 全连接。
- 与 `nn.LSTM(bidirectional=True)` 对比测试：验证自定义实现与 PyTorch 官方双向 LSTM 的数值一致性。

## 参考资料
- [Bidirectional Recurrent Neural Networks](https://ieeexplore.ieee.org/document/650093) (Schuster & Paliwal, IEEE 1997)
- [Speech Recognition with Deep Recurrent Neural Networks](https://arxiv.org/abs/1303.5778) (Graves et al., ICASSP 2013)
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) (Lample et al., NAACL 2016) —— BiLSTM-CRF 经典应用
- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
