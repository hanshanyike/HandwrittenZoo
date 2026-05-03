# ReLU / GELU / Swish 激活函数

## 算法简介

激活函数是神经网络引入非线性表达能力的关键组件。ReLU、GELU、Swish 分别代表了三代主流设计：从简单截断到概率加权，再到自门控平滑，每一次演进都在缓解梯度消失、提升训练稳定性与模型容量。

## 核心思想

- **ReLU**: 负值直接归零，正值原样通过。设计动机是"稀疏激活"——让网络在任意时刻只有部分神经元活跃，降低计算量并缓解梯度消失。
- **GELU**: 不再硬性截断，而是以标准正态分布的累积分布函数(CDF)对输入进行加权。输入越小，被"抑制"的概率越高，过渡平滑且处处可导。
- **Swish/SiLU**: 输入自身与 sigmoid 门控的乘积。sigmoid 输出一个 0~1 之间的门控信号，决定输入的通过比例。负值区域保留微弱梯度，避免神经元彻底死亡。

## 数学公式

### ReLU
$$
\text{ReLU}(x) = \max(0, x)
$$

### GELU (精确)
$$
\text{GELU}(x) = x \cdot \Phi(x) = 0.5 \cdot x \cdot \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
$$
其中 $\Phi(x)$ 为标准正态分布的累积分布函数。

### GELU (tanh 近似，PyTorch 默认)
$$
\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot \left(x + 0.044715 \cdot x^3\right)\right)\right)
$$

### Swish / SiLU
$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

## 时间/空间复杂度

- **时间复杂度**: $O(n)$ — 逐元素操作，与输入规模线性相关
- **空间复杂度**: $O(n)$ — 仅需存储输出张量
- **对比**: ReLU 计算量最小（仅需比较），GELU 和 Swish 因含 exp/tanh 计算量略高，但在 GPU 上差异可忽略

## 面试高频考点

1. **问题**: ReLU 的缺陷是什么？如何解决？
   **答案**: ReLU 的负区间梯度恒为零，导致"神经元死亡"(dying ReLU)。解决方案包括：使用 Leaky ReLU（负区间小斜率）、PReLU（可学习斜率）、GELU/Swish（平滑负区间保留梯度）。

2. **问题**: GELU 为什么比 ReLU 更适合 Transformer？
   **答案**: Transformer 深层堆叠对梯度流极度敏感。GELU 平滑且处处可导，负区间保留微弱梯度，避免硬截断带来的训练不稳定；同时概率加权机制与 Dropout 有理论联系，更适合大规模预训练。

3. **问题**: Swish 与 GELU 的异同？
   **答案**: 两者都是平滑、非单调激活函数，负区间均保留梯度。Swish 是输入与 sigmoid 的乘积（自门控），GELU 是输入与高斯 CDF 的乘积（概率加权）。实验上两者性能接近，Swish 在部分视觉任务更优，GELU 在 NLP 任务更常用。

4. **问题**: 为什么大模型几乎不再使用 ReLU？
   **答案**: 大模型参数量大、训练数据多，对优化景观的平滑性要求更高。ReLU 的硬截断会在深层网络中放大梯度稀疏问题，导致训练不稳定；GELU/Swish 的平滑特性使梯度流更稳定，收敛更快、泛化更好。

## 代码解析

### ReLU 实现
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.inplace:
        return x.clamp_(min=0)
    return x.clamp(min=0)
```
使用 `torch.clamp` 将负值截断为零。`inplace` 版本直接修改输入张量，节省显存。

### GELU 实现
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )
    )
```
采用 tanh 近似公式，避免 erf 计算开销。系数 `0.044715` 和 `sqrt(2/pi)` 来自原始论文的拟合结果，与 PyTorch 原生实现逐位对齐。

### Swish 实现
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)
```
最简洁的实现：输入与 sigmoid 门控逐元素相乘。sigmoid 输出范围 (0,1)，起到软门控作用。

## 参考资料

- [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) — Hendrycks & Gimpel, 2016
- [Searching for Activation Functions](https://arxiv.org/abs/1710.05941) — Ramachandran et al., 2017 (Swish)
- [PyTorch nn.GELU 文档](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html)
- [PyTorch nn.SiLU 文档](https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html)
