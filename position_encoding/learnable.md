# Learnable Positional Embedding (可学习位置嵌入)

## 算法简介

可学习位置嵌入是 BERT、GPT-2、Vision Transformer (ViT) 等模型采用的位置编码方案。它将每个位置的位置向量作为**可训练参数**，与词嵌入相加，让模型通过反向传播和大量数据自主学习最优的位置表示。相比手工设计的正弦编码，可学习嵌入更灵活，但增加了参数量且缺乏长度外推能力。

## 核心思想

正弦编码是确定性的数学函数，虽然具有优雅的相对位置线性性质，但可能不是针对特定任务的最优表示。可学习位置嵌入的核心假设是：

> 如果给模型足够的参数和数据，它可以通过端到端训练自动发现比手工设计更好的位置表示。

设计要点：
1. **参数化**：每个位置 $pos \in [0, \text{max_len}-1]$ 对应一个独立的 $d_{\text{model}}$ 维向量 $\mathbf{p}_{pos}$。
2. **查表机制**：前向传播时，根据位置索引从嵌入矩阵中查找对应向量，与词嵌入相加。
3. **端到端学习**：位置嵌入与模型其他参数一起通过反向传播更新。
4. **初始化敏感**：通常使用较小的随机值初始化，避免初始阶段位置编码淹没词嵌入信号。

## 数学公式

### 1. 位置嵌入矩阵

定义可学习的位置嵌入矩阵 $\mathbf{P} \in \mathbb{R}^{\text{max_len} \times d_{\text{model}}}$，其中第 $pos$ 行是位置 $pos$ 的嵌入向量：

$$
\mathbf{P} = \begin{bmatrix}
\mathbf{p}_0 \\
\mathbf{p}_1 \\
\vdots \\
\mathbf{p}_{\text{max_len}-1}
\end{bmatrix}
$$

### 2. 前向传播

对于输入序列 $X_{\text{embed}} \in \mathbb{R}^{\text{batch} \times \text{seq_len} \times d_{\text{model}}}$：

$$
X_{\text{final}} = X_{\text{embed}} + \mathbf{P}_{[0:\text{seq_len}]}
$$

其中 $\mathbf{P}_{[0:\text{seq_len}]}$ 表示取前 $\text{seq_len}$ 行位置嵌入，通过广播机制加到每个样本上。

### 3. 初始化

常用初始化方式：

$$
\mathbf{p}_{pos}^{(i)} \sim \mathcal{N}(0, \sigma^2), \quad \sigma = 0.02
$$

或使用正弦编码的值进行初始化，兼顾手工设计的先验与学习的灵活性。

## 时间/空间复杂度

- **时间复杂度**：$O(\text{seq_len} \cdot d_{\text{model}})$，前向传播仅为查表和向量相加
- **空间复杂度**：$O(\text{max_len} \cdot d_{\text{model}})$ 可训练参数
- **参数量**：$\text{max_len} \cdot d_{\text{model}}$
- **与替代方案对比**：
  - 对比正弦编码：更灵活，但增加参数量；无确定性相对位置归纳偏置；外推性差。
  - 对比 RoPE/ALiBi：无法显式建模相对位置；长文本外推能力最弱。

## 面试高频考点

1. **问题**：可学习位置嵌入与正弦编码相比，优缺点分别是什么？
   **答案**：
   - 优点：更灵活，模型可以针对特定任务学习最优位置表示；实现简单直观。
   - 缺点：增加 $\text{max_len} \cdot d_{\text{model}}$ 参数量；对更长序列无法外推（超出 max_len 的位置没有对应参数）；缺乏确定性的相对位置归纳偏置。

2. **问题**：为什么可学习位置嵌入无法外推到更长的序列？
   **答案**：因为每个位置都有独立的参数，训练时只学习了 $[0, \text{max_len}-1]$ 范围内的位置表示。当遇到位置 $\ge \text{max_len}$ 时，没有对应的参数可用。这与正弦编码（确定性函数）和 RoPE/ALiBi（相对距离函数）形成鲜明对比。

3. **问题**：可学习位置嵌入的初始化为什么通常使用较小的标准差（如 0.02）？
   **答案**：如果初始化值过大，位置编码会在训练初期淹没词嵌入的信号，导致模型难以先学习词义再学习位置关系。较小的初始化使位置编码在开始时接近零，让模型逐步学习位置信息的重要性。

4. **问题**：BERT 使用可学习位置嵌入，而 GPT-3 使用什么？为什么不同模型选择不同方案？
   **答案**：GPT-3 也使用可学习位置嵌入。RoPE/ALiBi 等方案在 GPT-3 之后才逐渐成为主流。选择差异主要源于时代背景：早期 Transformer（BERT/GPT-1/2）默认使用可学习嵌入；后来研究发现相对位置编码（RoPE/ALiBi）在长文本和泛化性上更优，因此现代 LLM（LLaMA、Mistral 等）普遍采用 RoPE。

5. **问题**：如果必须用可学习位置嵌入处理超出 max_len 的序列，有什么补救方法？
   **答案**：
   - **插值法**：将训练好的位置嵌入进行线性插值，压缩到更长序列中（如 Positional Interpolation）。
   - **循环使用**：将位置索引对 max_len 取模，复用已有参数（效果通常较差）。
   - **动态扩展**：在推理时动态扩展嵌入矩阵，对新位置进行微调（如 YaRN 方法的部分思路）。

## 代码解析

### 1. nn.Embedding 实现

```python
self.pos_embedding = nn.Embedding(max_len, d_model)
nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
```

- `nn.Embedding` 本质上是一个可学习的查找表，输入整数索引，输出对应向量。
- 初始化使用较小的标准差（0.02），避免位置编码信号过强。

### 2. 前向传播查表

```python
positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
positions = positions.expand(x.size(0), -1)
pos_emb = self.pos_embedding(positions)
x = x + pos_emb
```

- `torch.arange` 生成位置索引，`unsqueeze` 增加 batch 维度。
- `expand` 将索引复制到 batch 维度，无需额外内存。
- `nn.Embedding` 自动将索引转换为对应向量，形状 `(batch, seq_len, d_model)`。

### 3. nn.Parameter 实现（等价方案）

```python
self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
```

- 直接使用 `nn.Parameter` 显式定义参数，形状 `(1, max_len, d_model)`。
- 前向传播时直接切片截取，利用广播机制相加。
- 与 `nn.Embedding` 实现数学等价，但 `nn.Embedding` 更语义化。

## 参考资料

- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
- [Language Models are Unsupervised Multitask Learners (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [Positional Interpolation (Chen et al., 2023)](https://arxiv.org/abs/2306.15595)
