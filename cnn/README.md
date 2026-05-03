# CNN 经典架构 (Classic CNN Architectures)

本目录包含计算机视觉领域最具影响力的经典卷积神经网络架构的从零实现。

## 目录结构

| 文件 | 说明 | 面试频率 |
|------|------|----------|
| [resnet.py](resnet.py) + [resnet.md](resnet.md) | ResNet 残差网络（跳跃连接） | 极高 |
| [vgg.py](vgg.py) + [vgg.md](vgg.md) | VGG 网络（小卷积核深度堆叠） | 高 |

## 架构演进脉络

```
LeNet (1998) → AlexNet (2012) → VGG (2014) → ResNet (2015) → DenseNet / EfficientNet ...
                ↑                ↑              ↑
           ReLU+Dropout     3x3深度堆叠      残差连接
           GPU训练           规整设计         可训练百层+
```

### VGG (2014)
- **核心贡献**：证明小卷积核（3x3）深度堆叠的有效性
- **关键设计**：感受野等效替换、规整的 Block 结构
- **主要缺点**：参数量巨大（~138M），全连接层占主导

### ResNet (2015)
- **核心贡献**：跳跃连接解决深层网络退化问题
- **关键设计**：残差学习、BasicBlock / Bottleneck、1x1 卷积降维
- **主要优势**：可训练 100+ 层，参数量少（ResNet-50 仅 ~25M），精度高

## 快速开始

```python
from cnn.resnet import resnet18, resnet50
from cnn.vgg import vgg16

# 创建模型
model = resnet50(num_classes=10)
# 或
model = vgg16(num_classes=10)

# 前向传播
import torch
x = torch.randn(2, 3, 224, 224)
out = model(x)
print(out.shape)  # (2, 10)
```

## 面试重点
- **感受野计算**：VGG 中 3x3 堆叠与 5x5/7x7 的等效关系
- **退化问题**：ResNet 为什么能解决？与梯度消失的区别？
- **参数量对比**：VGG-16 vs ResNet-50 的参数量和精度差异
- **1x1 卷积的作用**：降维/升维、通道混合、计算量优化
