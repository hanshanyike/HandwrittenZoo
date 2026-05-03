# Batch Normalization (BatchNorm)

## 算法简介

Batch Normalization（批归一化）由 Ioffe & Szegedy 于 2015 年提出，是深度学习领域里程碑式的优化技术。它通过对每个 mini-batch 在通道维度上计算均值和方差进行归一化，有效缓解了**内部协变量偏移（Internal Covariate Shift）**，使得网络可以使用更大的学习率、更快收敛，并具有一定的正则化效果。BatchNorm 是 ResNet、DenseNet 等经典 CNN 架构的核心组件。

## 核心思想

深度网络训练过程中，每一层的输入分布会随着前面层参数更新而不断变化，这被称为“内部协变量偏移”。BatchNorm 的洞察是：**在每一层输入处进行归一化，将其拉回到均值为 0、方差为 1 的标准分布**，从而：
- 缓解梯度消失/爆炸问题；
- 允许使用更大的学习率，加速训练；
- 减少对参数初始化的敏感性；
- 引入轻微噪声，起到正则化作用（类似 Dropout）。

关键设计：**训练时用当前 batch 统计量，推理时用移动平均统计量**，保证训练和推理行为一致。

## 数学公式

对输入特征图 $X \in \mathbb{R}^{B \times C \times H \times W}$（以 2D 为例），BatchNorm 对每个通道 $c$ 独立计算：

**训练阶段**：

$$
\mu_c = \frac{1}{B \cdot H \cdot W} \sum_{b,h,w} X_{b,c,h,w}
$$

$$
\sigma_c^2 = \frac{1}{B \cdot H \cdot W} \sum_{b,h,w} (X_{b,c,h,w} - \mu_c)^2
$$

$$
\hat{X}_{b,c,h,w} = \frac{X_{b,c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}}
$$

$$
Y_{b,c,h,w} = \gamma_c \cdot \hat{X}_{b,c,h,w} + \beta_c
$$

同时更新移动平均：

$$
\text{running\_mean}_c = (1 - \text{momentum}) \cdot \text{running\_mean}_c + \text{momentum} \cdot \mu_c
$$

$$
\text{running\_var}_c = (1 - \text{momentum}) \cdot \text{running\_var}_c + \text{momentum} \cdot \sigma_c^2
$$

**推理阶段**：

$$
Y_{b,c,h,w} = \gamma_c \cdot \frac{X_{b,c,h,w} - \text{running\_mean}_c}{\sqrt{\text{running\_var}_c + \epsilon}} + \beta_c
$$

其中：
- $\mu_c, \sigma_c^2$：当前 batch 在通道 $c$ 上的均值和方差
- $\text{running\_mean}, \text{running\_var}$：训练阶段累积的移动平均统计量
- $\gamma_c, \beta_c$：可学习的缩放和偏移参数
- $\epsilon$：数值稳定项
- $\text{momentum}$：移动平均动量（PyTorch 默认 0.1）

## 时间/空间复杂度

- **时间复杂度**：$O(B \times C \times H \times W)$ —— 每个元素访问常数次
- **空间复杂度**：$O(C)$ —— 存储 $\gamma$、$\beta$、running_mean、running_var
- **与 LayerNorm/RMSNorm 对比**：
  - BatchNorm 在 batch 维度统计，依赖 batch size；
  - 训练和推理行为不同（需维护移动平均）；
  - 对序列长度敏感，不适用于变长序列和自回归模型。

## 面试高频考点

1. **BatchNorm 和 LayerNorm 的根本区别是什么？**
   **答案**：统计维度不同。BN 在 **batch 维度**（跨样本）统计每个通道的均值方差；LN 在 **特征维度**（样本内部）统计。因此 BN 依赖 batch size，LN 与 batch size 无关。

2. **为什么 Transformer 不使用 BatchNorm，而使用 LayerNorm？**
   **答案**：主要原因有四：① **序列长度可变**：不同句子长度不同，padding 会污染 BN 的 batch 统计量；② **自注意力机制**：同一 batch 中同一位置的 token 可能来自完全不同的句子，跨样本统计无意义；③ **batch size 敏感**：大模型分布式训练中单卡 batch size 很小，BN 统计不稳定；④ **训练和推理不一致**：BN 推理依赖移动平均，而 LN 训练和推理逻辑完全一致。

3. **BatchNorm 训练时和推理时的行为有什么不同？**
   **答案**：训练时使用**当前 batch 的统计量**（均值、方差）进行归一化，并更新移动平均；推理时使用**训练阶段累积的移动平均统计量**，不再依赖当前 batch，保证输出确定性。

4. **BatchNorm 的 momentum 参数是什么意思？**
   **答案**：PyTorch 中 `momentum` 定义更新比例：`running = (1 - momentum) * running + momentum * batch`。默认值 0.1 表示新 batch 统计量占 10%，历史占 90%。注意这与 SGD 优化器中的 momentum 含义不同。

5. **BatchNorm 为什么能起到正则化效果？**
   **答案**：因为每个 batch 的统计量（均值、方差）带有随机性，归一化过程相当于给网络输入引入了微小噪声，这种噪声与 Dropout 类似，能够抑制过拟合，起到正则化作用。

6. **BatchNorm 应该放在激活函数之前还是之后？**
   **答案**：原始论文推荐放在激活函数之前（即 Linear -> BN -> ReLU）。因为 BN 的输入如果经过 ReLU 会全部非负，导致分布偏移；放在激活之前可以更好地稳定输入分布。但实际中也有放在之后的情况，需根据具体任务验证。

## 代码解析

### 训练与推理分支

```python
if self.training:
    mean = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, unbiased=False, keepdim=True)
    # 更新移动平均...
else:
    mean = self.running_mean.view(...)
    var = self.running_var.view(...)
```

- `self.training` 是 PyTorch `nn.Module` 的内置属性，由 `.train()` 和 `.eval()` 控制；
- 推理时必须使用 `running_mean`/`running_var`，否则输出会随 batch 变化，导致结果不确定。

### 移动平均更新

```python
with torch.no_grad():
    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
```

- 使用 `torch.no_grad()` 避免移动平均参与反向传播；
- `register_buffer` 确保状态会被 `state_dict()` 保存和加载。

### 广播机制

```python
mean = self.running_mean.view(1, -1, 1)  # 适配 (B, C, L)
x_norm = (x - mean) / torch.sqrt(var + self.eps)
```

- 通过 `view` 调整统计量形状，利用 PyTorch 广播机制完成逐元素运算；
- 不同维度输入（2D/3D/4D）只需调整 view 的形状即可复用同一逻辑。

## 参考资料

- [Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", ICML 2015](https://arxiv.org/abs/1502.03167)
- [PyTorch 官方文档 - torch.nn.BatchNorm1d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
- [PyTorch 官方文档 - torch.nn.BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
- [Why Batch Norm works? ( deeplearning.ai )](https://www.coursera.org/lecture/deep-neural-network/why-does-batch-norm-work-RAnbP)
