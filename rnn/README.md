# RNN 经典架构 (Classic RNN Architectures)

本目录包含序列建模领域最具影响力的循环神经网络及其变体的从零实现。

## 目录结构

| 文件 | 说明 | 面试频率 |
|------|------|----------|
| [lstm.py](lstm.py) + [lstm.md](lstm.md) | LSTM 长短期记忆网络（三门控机制） | 极高 |
| [gru.py](gru.py) + [gru.md](gru.md) | GRU 门控循环单元（两门控简化版） | 高 |
| [bilstm.py](bilstm.py) + [bilstm.md](bilstm.md) | BiLSTM 双向 LSTM（前后文编码） | 极高 |

## 架构演进脉络

```
RNN (1986) → LSTM (1997) → GRU (2014) → BiLSTM (1997/2013) → Transformer (2017)
              ↑              ↑              ↑
         细胞状态+三门控   两门控简化      双向上下文编码
         解决梯度消失      参数量减少25%   编码器任务标配
```

### LSTM (1997)
- **核心贡献**：通过细胞状态和三门控（遗忘门、输入门、输出门）解决 RNN 梯度消失问题
- **关键设计**：细胞状态作为"信息高速公路"，线性更新使梯度可长距离传播
- **主要缺点**：参数量大（约为传统 RNN 的 4 倍），计算开销高

### GRU (2014)
- **核心贡献**：将 LSTM 的三门控简化为两门控，合并细胞状态与隐藏状态
- **关键设计**：重置门控制历史参与度，更新门同时承担遗忘/输入功能
- **主要优势**：参数量约为 LSTM 的 75%，训练更快，多数任务性能相当

### BiLSTM (1997/2013)
- **核心贡献**：同时编码左上下文和右上下文，获得完整的双向语义表示
- **关键设计**：前向 LSTM + 后向 LSTM，输出拼接或求和融合
- **主要应用**：NER、POS、文本分类等编码任务的标准基线

## 快速开始

```python
from rnn.lstm import LSTM
from rnn.gru import GRU
from rnn.bilstm import BiLSTM, BiLSTMClassifier

# LSTM
lstm = LSTM(input_size=128, hidden_size=256, num_layers=2, output_size=10)
out, (h_n, c_n) = lstm(torch.randn(2, 20, 128))

# GRU
gru = GRU(input_size=128, hidden_size=256, num_layers=2, output_size=10)
out, h_n = gru(torch.randn(2, 20, 128))

# BiLSTM
bilstm = BiLSTM(input_size=128, hidden_size=256, num_layers=2, merge_mode="concat")
out, _ = bilstm(torch.randn(2, 20, 128))  # out.shape: (2, 20, 512)

# BiLSTM 文本分类器
classifier = BiLSTMClassifier(vocab_size=10000, embed_dim=128, hidden_size=256, num_classes=5)
logits = classifier(torch.randint(0, 10000, (2, 30)))
```

## 面试重点
- **梯度消失**：LSTM/GRU 如何通过门控和线性路径解决？与传统 RNN 的对比？
- **三门控 vs 两门控**：LSTM 的三个门各负责什么？GRU 如何用两个门实现类似功能？
- **双向机制**：BiLSTM 的两个方向参数是否共享？为什么后向需要翻转输入？
- **应用场景**：BiLSTM 为什么不能用于语言模型？什么任务适合用 BiLSTM？
- **参数量对比**：RNN vs LSTM vs GRU 的参数量和计算量差异
