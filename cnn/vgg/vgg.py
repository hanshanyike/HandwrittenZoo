"""
VGGNet (Visual Geometry Group Network)

Algorithm: VGG 网络 —— 使用非常小的 3x3 卷积核深度堆叠，证明增加网络深度
         能显著提升性能，开启了"深层网络"时代。

Core Idea:
- 用多个 3x3 卷积层替代大卷积核（如 5x5、7x7），在保持相同感受野的同时
  增加非线性激活次数，提升模型表达能力。
- 网络结构极其规整：由多个 VGG Block 组成，每个 Block 内部通道数相同，
  Block 之间通道数翻倍、空间尺寸减半。
- 设计简洁优雅，易于理解和扩展，是后续许多网络（如 ResNet）的对比基准。

Complexity:
    - 时间复杂度: O(N * C * H * W * K^2) —— 深层小卷积核堆叠
    - 空间复杂度: O(L * C * H * W) —— 特征图占主导
    - 参数量: VGG-16 ~138M, VGG-19 ~144M（参数量巨大是主要缺点）

Interview Frequency: 高（CNN 基础架构，感受野计算是常考点）

References:
    - Simonyan & Zisserman, "Very Deep Convolutional Networks for Large-Scale Image Recognition", ICLR 2015
"""

import torch
import torch.nn as nn


class VGGBlock(nn.Module):
    """
    VGG 网络的基本构建块。

    结构:
        重复 num_convs 次: (conv3x3 -> relu) -> maxpool2x2

    核心设计:
        - 每个 Block 内部使用相同数量的通道数
        - 所有卷积核均为 3x3，padding=1 保持空间尺寸
        - Block 末尾使用 2x2 MaxPool 进行下采样（空间减半）
        - 每个 Block 的卷积层数逐渐增加（从 2 层到 4 层）

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数（Block 内所有卷积层的通道数）
        num_convs: Block 内卷积层的数量
    """

    def __init__(self, in_channels: int, out_channels: int, num_convs: int):
        super().__init__()
        layers = []

        # 构建 Block 内的卷积层序列
        for i in range(num_convs):
            # 第一个卷积层负责通道数变换，后续卷积层保持通道数不变
            conv_in = in_channels if i == 0 else out_channels
            layers.append(
                nn.Conv2d(
                    conv_in, out_channels,
                    kernel_size=3, padding=1, bias=False  # bias=False 因为后面有 BN
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))  # 现代实现通常加入 BN 加速训练
            layers.append(nn.ReLU(inplace=True))

        # Block 末尾的下采样层：2x2 MaxPool，stride=2
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入特征图，形状 (batch, in_channels, H, W)
        Returns:
            输出特征图，形状 (batch, out_channels, H//2, W//2)
        """
        return self.block(x)


class VGG(nn.Module):
    """
    VGG 完整网络实现。

    架构组成:
        1. 特征提取层（features）: 多个 VGGBlock 堆叠
        2. 全局平均池化（avgpool）: 将空间维度压缩为 1x1（替代原始实现中的全连接层前展平）
        3. 分类器（classifier）: 三层全连接 + Dropout

    经典配置:
        - VGG-11: [1, 1, 2, 2, 2]  —— 11 层权重层（8 卷积 + 3 全连接）
        - VGG-13: [2, 2, 2, 2, 2]  —— 13 层权重层（10 卷积 + 3 全连接）
        - VGG-16: [2, 2, 3, 3, 3]  —— 16 层权重层（13 卷积 + 3 全连接）
        - VGG-19: [2, 2, 4, 4, 4]  —— 19 层权重层（16 卷积 + 3 全连接）

    Args:
        conv_nums: 每个 VGGBlock 的卷积层数量列表
        num_classes: 分类类别数，默认 1000（ImageNet）
    """

    # 每个 Block 的输出通道数配置
    CHANNELS = [64, 128, 256, 512, 512]

    def __init__(
        self,
        conv_nums: list[int],
        num_classes: int = 1000,
    ):
        super().__init__()
        assert len(conv_nums) == len(self.CHANNELS), (
            f"conv_nums 长度必须为 {len(self.CHANNELS)}"
        )

        # 构建特征提取层：多个 VGGBlock 顺序连接
        features = []
        in_channels = 3  # RGB 输入
        for num_convs, out_channels in zip(conv_nums, self.CHANNELS):
            features.append(
                VGGBlock(in_channels, out_channels, num_convs)
            )
            in_channels = out_channels  # 下一个 Block 的输入通道数

        self.features = nn.Sequential(*features)

        # 全局平均池化：将 7x7 特征图压缩为 1x1
        # 原始 VGG 论文使用展平后接 4096 维全连接，这里使用 AdaptiveAvgPool2d 更灵活
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 分类器：三层全连接 + ReLU + Dropout
        # 输入维度为最后一个 Block 的输出通道数（512）
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming 初始化（适用于 ReLU）和全连接层 Xavier 初始化。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用 Kaiming 初始化
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层使用 Xavier 初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入图像，形状 (batch, 3, H, W)
        Returns:
            分类 logits，形状 (batch, num_classes)
        """
        # 特征提取
        x = self.features(x)

        # 全局池化
        x = self.avgpool(x)

        # 展平后送入全连接分类器
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def vgg11(num_classes: int = 1000) -> VGG:
    """构建 VGG-11（配置: [1, 1, 2, 2, 2]）。"""
    return VGG([1, 1, 2, 2, 2], num_classes=num_classes)


def vgg13(num_classes: int = 1000) -> VGG:
    """构建 VGG-13（配置: [2, 2, 2, 2, 2]）。"""
    return VGG([2, 2, 2, 2, 2], num_classes=num_classes)


def vgg16(num_classes: int = 1000) -> VGG:
    """构建 VGG-16（配置: [2, 2, 3, 3, 3]）。"""
    return VGG([2, 2, 3, 3, 3], num_classes=num_classes)


def vgg19(num_classes: int = 1000) -> VGG:
    """构建 VGG-19（配置: [2, 2, 4, 4, 4]）。"""
    return VGG([2, 2, 4, 4, 4], num_classes=num_classes)


if __name__ == "__main__":
    # ==================== 自测区域 ====================
    torch.manual_seed(42)

    batch_size = 2
    num_classes = 10
    x = torch.randn(batch_size, 3, 224, 224)

    # 1. 测试单个 VGGBlock
    print("=" * 60)
    print("Test 1: VGGBlock forward pass")
    block = VGGBlock(in_channels=3, out_channels=64, num_convs=2)
    x_block = torch.randn(batch_size, 3, 224, 224)
    out_block = block(x_block)
    # 经过 MaxPool2d(2,2) 后空间尺寸减半
    assert out_block.shape == (batch_size, 64, 112, 112)
    print(f"VGGBlock input shape:  {x_block.shape}")
    print(f"VGGBlock output shape: {out_block.shape}")

    # 2. 测试 VGG-11
    print("=" * 60)
    print("Test 2: VGG-11")
    model_11 = vgg11(num_classes=num_classes)
    out_11 = model_11(x)
    assert out_11.shape == (batch_size, num_classes)
    print(f"VGG-11 output shape: {out_11.shape}")
    total_params_11 = sum(p.numel() for p in model_11.parameters())
    print(f"VGG-11 total parameters: {total_params_11:,}")

    # 3. 测试 VGG-16
    print("=" * 60)
    print("Test 3: VGG-16")
    model_16 = vgg16(num_classes=num_classes)
    out_16 = model_16(x)
    assert out_16.shape == (batch_size, num_classes)
    print(f"VGG-16 output shape: {out_16.shape}")
    total_params_16 = sum(p.numel() for p in model_16.parameters())
    print(f"VGG-16 total parameters: {total_params_16:,}")

    # 4. 测试 VGG-19
    print("=" * 60)
    print("Test 4: VGG-19")
    model_19 = vgg19(num_classes=num_classes)
    out_19 = model_19(x)
    assert out_19.shape == (batch_size, num_classes)
    print(f"VGG-19 output shape: {out_19.shape}")
    total_params_19 = sum(p.numel() for p in model_19.parameters())
    print(f"VGG-19 total parameters: {total_params_19:,}")

    # 5. 验证中间特征图尺寸变化
    print("=" * 60)
    print("Test 5: Feature map size tracking")
    test_x = torch.randn(1, 3, 224, 224)
    print(f"Input:  {test_x.shape}")
    for i, block in enumerate(model_16.features):
        test_x = block(test_x)
        print(f"After Block {i+1}: {test_x.shape}")

    # 6. 验证梯度回传
    print("=" * 60)
    print("Test 6: Gradient backpropagation")
    loss = out_16.sum()
    loss.backward()
    has_grad = any(p.grad is not None for p in model_16.parameters())
    assert has_grad, "No gradients computed!"
    print("Gradient backpropagation: OK")

    print("=" * 60)
    print("All VGG tests passed!")
