# ResNet (Residual Network)

## 算法简介
ResNet（残差网络）是 2015 年 ImageNet 竞赛冠军，由何恺明等人提出。它通过引入**跳跃连接（Skip Connection）**解决了深层神经网络的退化问题（Degradation Problem），使得训练超过 100 层甚至 1000 层的网络成为可能，是深度学习历史上最具影响力的架构之一。

## 核心思想
传统观点认为，更深的网络应该具有更强的表达能力。然而实验发现，当网络深度超过一定层数后，训练集准确率反而下降——这不是过拟合，而是**退化问题**。

ResNet 的关键洞察是：
1. **残差学习**：与其让网络直接学习底层映射 $H(x)$，不如让它学习残差 $F(x) = H(x) - x$。这样原始映射就变成了 $H(x) = F(x) + x$。
2. **恒等映射的易学习性**：如果某一层最优就是恒等映射（什么都不做），那么让残差 $F(x) \to 0$ 比让网络直接学习 $H(x) = x$ 更容易。
3. **梯度高速公路**：跳跃连接形成了一条梯度回传的捷径，有效缓解了深层网络的梯度消失问题。

## 数学公式

### 残差块输出
$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

其中 $\mathcal{F}$ 是残差映射（即卷积层堆叠），$\mathbf{x}$ 是跳跃连接。

### 带下采样的残差块
当输入输出维度不匹配时（stride > 1 或通道数变化），shortcut 路径需要进行线性投影：
$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + W_s \mathbf{x}$$

其中 $W_s$ 是 1x1 卷积，用于维度对齐。

### BasicBlock（ResNet-18/34）
$$\mathcal{F}(\mathbf{x}) = W_2 \cdot \text{ReLU}(\text{BN}(W_1 \cdot \mathbf{x}))$$

### Bottleneck（ResNet-50/101/152）
$$\mathcal{F}(\mathbf{x}) = W_3 \cdot \text{ReLU}(\text{BN}(W_2 \cdot \text{ReLU}(\text{BN}(W_1 \cdot \mathbf{x}))))$$

其中 $W_1$ 是 1x1 降维，$W_2$ 是 3x3 空间卷积，$W_3$ 是 1x1 升维。

## 时间/空间复杂度
- **时间复杂度**: $O(N \cdot C \cdot H \cdot W \cdot K^2)$，与普通卷积网络同阶，$K$ 为卷积核大小
- **空间复杂度**: $O(L \cdot C \cdot H \cdot W)$，主要由中间特征图决定，$L$ 为网络深度
- **参数量对比**:
  - ResNet-18: ~11.7M
  - ResNet-34: ~21.8M
  - ResNet-50: ~25.6M（Bottleneck 设计使深层网络参数量反而可控）
  - ResNet-101: ~44.5M
  - ResNet-152: ~60.2M
- **与替代方案对比**:
  - VGG-16: ~138M 参数，ResNet-50 精度更高但参数量仅为其 1/5
  - Plain Network（无残差连接）：同深度下训练误差显著高于 ResNet

## 面试高频考点

1. **ResNet 解决了什么问题？退化问题与过拟合有什么区别？**
   **答案**: ResNet 解决了深层网络的**退化问题**（Degradation）：随着网络加深，训练集准确率反而下降。这不是过拟合（过拟合是训练集好、测试集差），而是优化困难——深层网络难以学习到恒等映射。ResNet 通过残差连接让网络可以轻松地"选择"只使用恒等映射。

2. **为什么残差连接能缓解梯度消失？**
   **答案**: 反向传播时，残差块的梯度有两条路径：一条经过卷积层（可能梯度变小），另一条直接通过跳跃连接（梯度为 1）。因此即使某条路径梯度消失，另一条路径仍能传递梯度，形成"梯度高速公路"。数学上，若残差映射导数为 $\frac{\partial \mathcal{F}}{\partial x}$，则总梯度为 $1 + \frac{\partial \mathcal{F}}{\partial x}$，至少保留了 1。

3. **BasicBlock 和 Bottleneck 的区别？**
   **答案**: BasicBlock 用于 ResNet-18/34，由两个 3x3 卷积组成，expansion=1；Bottleneck 用于 ResNet-50/101/152，由 1x1-3x3-1x1 三个卷积组成，中间维度压缩为 1/4，expansion=4。Bottleneck 用更少的参数量实现了更深的网络。

4. **1x1 卷积在 Bottleneck 中的作用？**
   **答案**: 两个作用：(1) **降维**：减少 3x3 卷积的输入通道数，大幅降低计算量；(2) **升维**：恢复通道数以匹配残差连接。这种"先降后升"的设计是 bottleneck 的核心，能在保持模型容量的同时减少 FLOPs。

5. **ResNet 的权重初始化为什么用 Kaiming 初始化？**
   **答案**: Kaiming (He) 初始化专为 ReLU 激活设计，考虑了 ReLU 截断负值导致的方差衰减。它根据 fan_in 或 fan_out 调整初始化标准差，使得前向传播时各层激活值的方差保持恒定，避免梯度爆炸或消失。

6. **如果输入输出通道数不同，shortcut 如何处理？**
   **答案**: 有两种方案：(1) 使用 1x1 卷积进行投影（projection），调整通道数和空间尺寸；(2) 使用零填充（zero-padding）补全通道。ResNet 论文中主要使用投影 shortcut，实验表明投影比零填充效果更好。

## 代码解析
- `BasicBlock`: ResNet-18/34 的浅层残差块，两个 3x3 卷积 + 残差连接。
- `Bottleneck`: ResNet-50/101/152 的深层残差块，1x1-3x3-1x1 结构 + expansion=4。
- `downsample`: 当 `stride != 1` 或通道数不匹配时，对 shortcut 路径进行 1x1 卷积投影。
- `_make_layer`: 构建一个残差阶段，第一个块可能下采样，后续块保持尺寸。
- `_initialize_weights`: Kaiming 正态初始化，BatchNorm 的 gamma 初始化为 1、beta 为 0。
- `resnet18/resnet34/resnet50`: 工厂函数，按论文配置构建不同深度的 ResNet。

## 参考资料
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (CVPR 2016)
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) (ECCV 2016) —— ResNet v2，改进残差单元设计
- [The Illustrated ResNet](http://jalammar.github.io/illustrated-resnet/)
- [PyTorch Official ResNet Implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
