# QLoRA (Quantized Low-Rank Adaptation)

## 算法简介

QLoRA 是华盛顿大学于 2023 年提出的高效微调方法，它在 LoRA 的基础上引入 **4-bit 量化**，将预训练模型权重压缩到 NF4（Normal Float 4）精度，配合**双量化（Double Quantization）**和**分页优化器（Paged Optimizer）**，首次实现了在单张 48GB GPU 上微调 65B 参数大模型，将微调门槛从数据中心级降低到消费级。

## 核心思想

1. **4-bit 量化基座**：预训练权重以 4-bit 存储，显存占用降至 FP16 的 1/4。
2. **动态反量化**：前向传播时，4-bit 权重动态反量化为 16-bit（BF16/FP16）进行计算，保证数值稳定性。
3. **LoRA 旁路**：只在 16-bit 精度下训练 LoRA 适配器，梯度不流向量化权重。
4. **双量化**：对量化常数（scale）再次量化，进一步降低显存开销。
5. **分页优化器**：当 GPU 显存不足时，自动将优化器状态分页到 CPU 内存。

## 数学公式

### 1. 4-bit 均匀量化

对于权重矩阵 $W$ 的某个 block，量化公式为：

$$
W_{q} = \text{round}\left(\frac{W - z}{s}\right), \quad s = \frac{W_{\max} - W_{\min}}{2^b - 1}, \quad z = W_{\min}
$$

其中 $b=4$，$s$ 为 scale，$z$ 为 zero-point。

反量化：

$$
\hat{W} = W_{q} \cdot s + z
$$

### 2. NF4（Normal Float 4）量化

QLoRA 使用 NF4 而非均匀量化，因为模型权重通常服从零均值正态分布。NF4 的量化级别根据正态分布的分位数设计，使得每个区间的概率质量相等：

$$
Q_{\text{NF4}} = \left\{ q_i \mid q_i = \Phi^{-1}\left(\frac{2i + 1}{2^{b+1}}\right), \quad i = 0, 1, \dots, 15 \right\}
$$

其中 $\Phi^{-1}$ 为标准正态分布的逆 CDF。NF4 在相同 bit 下比均匀量化精度更高。

### 3. 双量化（Double Quantization）

每个 block 的 scale 是 32-bit 浮点数。当 block_size=64 时，scale 的存储开销为：

$$
\text{Overhead}_{\text{single}} = \frac{32}{64} = 0.5 \text{ bit/参数}
$$

双量化将 scale 量化为 8-bit：

$$
\text{Overhead}_{\text{double}} = \frac{8}{64} = 0.125 \text{ bit/参数}
$$

总显存占用：

$$
\text{Memory} = 4 \text{ bits} + 0.125 \text{ bits} = 4.125 \text{ bits/参数}
$$

相比 FP16 的 16 bits，节省约 **3.9 倍**。

### 4. QLoRA 前向传播

$$
h = \underbrace{\text{dequantize}(W_{4\text{bit}})}_{\text{动态反量化}} \cdot x + \underbrace{\frac{\alpha}{r} B A x}_{\text{LoRA 旁路}}
$$

## 时间/空间复杂度

- **时间复杂度**：$O(B \cdot L \cdot d_{model} \cdot r)$ 每层（与 LoRA 相同）
  - 4-bit 反量化引入少量开销，但通常被计算掩盖。
- **空间复杂度**：
  - 基座权重：$O(d_{model}^2 / 2)$（4-bit 存储，等效 FP16 的 1/4）
  - LoRA 参数：$O(d_{model} \cdot r)$
  - 优化器状态：Paged Optimizer 可在 CPU/GPU 间动态分配
- **与 LoRA 对比**：QLoRA 的训练速度略慢（反量化开销），但显存需求降低 3~4 倍。

## 面试高频考点

### Q1: QLoRA 相比 LoRA 最大的改进是什么？

**A**: 最大的改进是**4-bit 量化基座权重**。LoRA 虽然减少了可训练参数，但基座权重仍需以 FP16 加载，大模型（如 65B）需要 130GB+ 显存。QLoRA 将基座压缩到 4-bit，65B 模型仅需约 35GB 显存，配合分页优化器可在单张 48GB GPU 上训练。

### Q2: 为什么 4-bit 量化后还能训练？不会梯度爆炸或消失吗？

**A**: 关键点在于**梯度不流向量化权重**：
1. 基座权重以 4-bit 存储，但前向传播时反量化为 16-bit 计算。
2. 基座权重被冻结（requires_grad=False），梯度只流向 LoRA 参数。
3. LoRA 参数保持 16-bit 精度训练，避免了低精度训练的数值不稳定问题。
4. 量化误差被视为一种“噪声”，由于权重冻结，这种噪声是固定的，不会累积。

### Q3: NF4 和 FP4 有什么区别？为什么 QLoRA 用 NF4？

**A**:
- **FP4（Floating Point 4-bit）**：4-bit 浮点格式，有 1 位符号、2 位指数、1 位尾数（或类似分配）。动态范围大但精度低。
- **NF4（Normal Float 4-bit）**：根据正态分布的分位数设计的非均匀量化级别。对于服从正态分布的权重，NF4 的信息熵损失更小，精度更高。

QLoRA 使用 NF4 是因为预训练模型的权重经过 LayerNorm 等操作后，通常近似服从零均值正态分布，NF4 能更好地保留分布细节。

### Q4: 双量化（Double Quantization）节省了多少显存？

**A**: 以 block_size=64 为例：
- 单量化：每个 block 一个 32-bit scale，开销 = 32/64 = 0.5 bit/参数
- 双量化：scale 量化为 8-bit，开销 = 8/64 = 0.125 bit/参数
- **节省**：0.375 bit/参数，对于 65B 模型约节省 3GB 显存。

虽然绝对数值不大，但对于在显存极限边缘运行的场景至关重要。

### Q5: QLoRA 适合什么场景？有什么局限性？

**A**:
**适合场景**：
- 消费级 GPU（24GB~48GB）微调 7B~70B 大模型
- 快速原型验证和实验迭代
- 多任务场景（每个任务一个 LoRA 适配器，共享量化基座）

**局限性**：
1. **速度较慢**：反量化开销使训练速度比 LoRA 慢 10%~30%。
2. **精度损失**：4-bit 量化对极端敏感的任务（如数学推理）可能有轻微影响。
3. **依赖 bitsandbytes**：生产环境部署需要兼容的推理框架。
4. **不可合并**：4-bit 权重无法直接与 LoRA 合并为单一权重矩阵。

## 代码解析

### 4-bit 量化与反量化

```python
def quantize_to_4bit(weight, block_size=64):
    blocks = weight.view(num_blocks, block_size)
    w_min, w_max = blocks.min(dim=1), blocks.max(dim=1)
    scale = (w_max - w_min) / 15.0
    zero_point = w_min
    quantized = torch.round((blocks - zero_point) / scale).clamp(0, 15)
    return quantized, scale, zero_point
```

分块量化是实际系统的标准做法，每块独立计算 scale 和 zero_point，避免大矩阵中数值范围差异导致的精度损失。

### Linear4Bit 动态反量化

```python
class Linear4Bit(nn.Module):
    def forward(self, x):
        weight = self.get_dequantized_weight()  # 4-bit -> 16-bit
        return F.linear(x, weight, self.bias)
```

前向传播时临时反量化，计算完成后释放，不长期占用 FP16 显存。

### 双量化

```python
class DoubleQuantization:
    def quantize_scale(self, scale):
        s_min, s_max = scale.min(), scale.max()
        scale_of_scale = (s_max - s_min) / 255.0
        quantized_scale = torch.round((scale - s_min) / scale_of_scale).clamp(0, 255)
        return quantized_scale.to(torch.uint8), torch.tensor([scale_of_scale, s_min])
```

对 scale 再次进行 8-bit 量化，将存储开销从 0.5 bit/参数降到 0.125 bit/参数。

## 参考资料

- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — QLoRA 原始论文
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) — 4-bit 量化的官方实现
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA 原始论文
