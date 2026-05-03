# Sinusoidal Positional Encoding (正弦位置编码)

## 算法简介

正弦位置编码是 Transformer 原论文《Attention Is All You Need》提出的**绝对位置编码**方案。它通过不同频率的正弦/余弦函数为序列中每个位置生成确定性的位置向量，直接加到词嵌入上，使原本顺序无关的自注意力获得位置感知能力。

## 核心思想

Transformer 的自注意力机制具有**置换等变性**——打乱输入 token 顺序，输出也会相应打乱，但模型本身无法区分 "cat eats fish" 和 "fish eats cat"。因此必须显式注入位置信息。

正弦编码的设计动机：
1. **唯一性**：每个位置 pos 对应唯一的 d_model 维向量，模型能区分不同绝对位置。
2. **相对位置线性**：对于固定偏移 k，$PE_{pos+k}$ 可表示为 $PE_{pos}$ 的线性函数（利用三角和角公式），使模型更容易学习相对位置关系。
3. **多尺度外推**：波长从 $2\pi$ 到 $10000 \cdot 2\pi$ 呈几何级数分布，覆盖多个数量级，对训练时未见过的更长序列有一定泛化能力。
4. **无需学习**：位置编码是确定性函数，不增加可训练参数，减少过拟合风险。

## 数学公式

对于位置 $pos$（从 0 开始）和维度索引 $i$（$0 \le i < d_{\text{model}}/2$）：

$$
PE_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right)
$$

等价写法（便于代码实现）：

$$
PE_{(pos, 2i)} = \sin\left( pos \cdot e^{-\frac{2i \cdot \ln(10000)}{d_{\text{model}}}} \right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left( pos \cdot e^{-\frac{2i \cdot \ln(10000)}{d_{\text{model}}}} \right)
$$

最终输入到 Transformer 的是词嵌入与位置编码之和：

$$
X_{\text{final}} = X_{\text{embed}} + PE
$$

## 时间/空间复杂度

- **时间复杂度**：预计算阶段 $O(\text{max_len} \cdot d_{\text{model}})$；前向传播 $O(\text{seq_len} \cdot d_{\text{model}})$
- **空间复杂度**：$O(\text{max_len} \cdot d_{\text{model}})$ 缓存位置编码矩阵
- **与替代方案对比**：
  - 对比可学习位置嵌入：无额外参数，但灵活性稍差；
  - 对比 RoPE/ALiBi：正弦编码是绝对位置编码，无法显式建模相对位置，长文本外推能力较弱。

## 面试高频考点

1. **问题**：为什么正弦编码使用 $10000^{2i/d_{\text{model}}}$ 作为分母？
   **答案**：这确保波长从低频到高频呈几何级数分布。$i=0$ 时波长最长（约 $2\pi \cdot 10000$），捕捉长程位置模式；$i=d_{\text{model}}/2-1$ 时波长最短（约 $2\pi$），捕捉精细的局部位置变化。底数 10000 是经验值，保证覆盖足够的频率范围。

2. **问题**：正弦编码如何实现相对位置的线性表示？
   **答案**：利用三角和角公式：$\sin(a+b) = \sin a \cos b + \cos a \sin b$，$\cos(a+b) = \cos a \cos b - \sin a \sin b$。因此 $PE_{pos+k}$ 可写成 $PE_{pos}$ 各分量的线性组合，系数仅与 $k$ 有关，与 $pos$ 无关。这意味着模型可以通过学习这些固定系数的权重来捕捉相对位置。

3. **问题**：正弦编码与可学习位置嵌入（Learnable Positional Embedding）相比有何优缺点？
   **答案**：
   - 优点：无额外参数；具有确定性的相对位置归纳偏置；对更长序列有一定外推性。
   - 缺点：每个位置编码独立，无法显式建模 token 间的相对距离；长文本外推效果不如 RoPE/ALiBi；灵活性不如可学习嵌入。

4. **问题**：为什么正弦编码的维度 $d_{\text{model}}$ 必须是偶数？
   **答案**：因为正弦和余弦成对出现，每对占用 2 个维度。若 $d_{\text{model}}$ 为奇数，最后一个维度无法配对，实现上需要特殊处理（如单独补零或重复）。

5. **问题**：如果序列长度超过预计算的 max_len 怎么办？
   **答案**：正弦编码是确定性函数，可以动态计算更长序列的位置编码，无需像可学习嵌入那样受限于训练时的 max_len。但实践中通常设置足够大的 max_len 并预计算，以节省前向传播时间。

## 代码解析

### 1. 预计算位置编码矩阵

```python
pe = torch.zeros(max_len, d_model)
position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
div_term = torch.exp(
    torch.arange(0, d_model, 2, dtype=torch.float32)
    * (-math.log(10000.0) / d_model)
)
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

- `position` 形状为 `(max_len, 1)`，表示每个位置的索引。
- `div_term` 形状为 `(d_model/2,)`，表示每个维度组的角频率分母。
- 利用广播机制，`position * div_term` 的形状自动扩展为 `(max_len, d_model/2)`。
- 偶数列赋正弦值，奇数列赋余弦值。

### 2. 前向传播

```python
x = x + self.pe[:, :seq_len, :]
return self.dropout(x)
```

- 从预计算的缓存中截取前 `seq_len` 个位置编码。
- 通过广播机制加到输入 `x` 上（`x` 形状为 `(batch, seq_len, d_model)`）。
- 最后应用 dropout，防止位置编码过拟合。

## 参考资料

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- [PyTorch Positional Encoding Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
