"""
ResNet (Residual Network)

Algorithm: 残差网络 —— 通过引入跳跃连接（Skip Connection）解决深层网络的梯度消失/退化问题，
         使得训练数百层甚至上千层的网络成为可能。

Core Idea:
- 传统深层网络随着层数增加会出现“退化问题”（Degradation），即训练集准确率反而下降。
- ResNet 提出“残差学习”：让网络学习输入与输出之间的残差映射 F(x) = H(x) - x，
  而非直接学习底层映射 H(x)。
- 通过恒等映射（Identity Mapping）的跳跃连接，梯度可以直接回传，缓解梯度消失。

Complexity:
    - 时间复杂度: O(N * C * H * W * K^2) —— 与标准卷积网络同阶，N为样本数
    - 空间复杂度: O(L * C * H * W) —— L为网络深度，主要由中间特征图决定
    - 参数量: ResNet-18 约 11.7M, ResNet-50 约 25.6M

Interview Frequency: 极高（CNN 面试必考，跳跃连接是核心考点）

References:
    - He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    ResNet-18/34 使用的浅层残差块（Basic Block）。

    结构:
        conv3x3 -> bn -> relu -> conv3x3 -> bn -> (+shortcut) -> relu

    核心设计:
        - 两个 3x3 卷积层，中间夹 BatchNorm 和 ReLU
        - 跳跃连接（shortcut）将输入直接加到第二个卷积的输出上
        - 当 stride != 1 或通道数变化时，shortcut 使用 1x1 卷积进行下采样
    """

    # 浅层块的扩展系数为 1（输出通道数 = 输入通道数）
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()

        # 第一个 3x3 卷积：可能带有 stride 进行下采样
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 第二个 3x3 卷积：保持空间尺寸不变
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 下采样层：当输入输出尺寸不匹配时，对 shortcut 路径进行维度对齐
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，实现残差连接。

        Args:
            x: 输入特征图，形状 (batch, in_channels, H, W)
        Returns:
            输出特征图，形状 (batch, out_channels, H', W')
        """
        identity = x  # 保存输入用于残差连接

        # 主路径：conv1 -> bn -> relu -> conv2 -> bn
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # shortcut 路径：若尺寸不匹配则进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接：主路径输出与 shortcut 相加
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    ResNet-50/101/152 使用的深层残差块（Bottleneck Block）。

    结构:
        conv1x1(降维) -> bn -> relu ->
        conv3x3(空间变换) -> bn -> relu ->
        conv1x1(升维) -> bn -> (+shortcut) -> relu

    核心设计:
        - 1x1 卷积先降维再升维，减少 3x3 卷积的计算量（bottleneck 思想）
        - 中间层维度为 out_channels // 4，因此 expansion = 4
        - 相比 BasicBlock 参数量更少，适合更深网络
    """

    # 深层块的扩展系数为 4（输出通道数 = 输入通道数 * 4）
    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()

        # 1x1 卷积降维：将通道数从 in_channels 降到 out_channels
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 卷积进行空间特征提取，可能带有 stride
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 卷积升维：将通道数从 out_channels 升回 out_channels * expansion
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，实现 bottleneck 残差连接。

        Args:
            x: 输入特征图，形状 (batch, in_channels, H, W)
        Returns:
            输出特征图，形状 (batch, out_channels*expansion, H', W')
        """
        identity = x

        # 1x1 降维
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 3x3 空间变换
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # 1x1 升维
        out = self.conv3(out)
        out = self.bn3(out)

        # shortcut 维度对齐
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差连接
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet 完整网络实现。

    架构组成:
        1. 初始卷积层（conv1）: 7x7 大卷积核，stride=2，快速降低空间维度
        2. 最大池化（maxpool）: 3x3，stride=2，进一步下采样
        3. 四个残差阶段（layer1-4）: 每个阶段由多个残差块堆叠
        4. 全局平均池化（avgpool）: 将空间维度压缩为 1x1
        5. 全连接层（fc）: 输出分类结果

    Args:
        block: 残差块类型，BasicBlock 或 Bottleneck
        layers: 每个阶段的残差块数量，如 [2, 2, 2, 2] 对应 ResNet-18
        num_classes: 分类类别数，默认 1000（ImageNet）
    """

    def __init__(
        self,
        block: type[nn.Module],
        layers: list[int],
        num_classes: int = 1000,
    ):
        super().__init__()
        self.in_channels = 64  # 初始通道数，跟踪每个阶段的输入通道

        # 初始卷积层：7x7 大卷积核快速提取低级特征并下采样
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 最大池化：进一步将空间维度减半
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个残差阶段，通道数逐渐翻倍，空间维度逐渐减半
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 全局平均池化 + 全连接分类头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化：Kaiming 初始化适用于 ReLU 激活
        self._initialize_weights()

    def _make_layer(
        self,
        block: type[nn.Module],
        out_channels: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """
        构建一个残差阶段（layer），包含多个残差块。

        Args:
            block: 残差块类
            out_channels: 该阶段的基础输出通道数
            blocks: 该阶段的残差块数量
            stride: 第一个残差块的步长（用于下采样）
        Returns:
            nn.Sequential 包含所有残差块
        """
        downsample = None

        # 当 stride != 1 或通道数变化时，需要对 shortcut 进行下采样
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        # 第一个残差块可能带有下采样
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample)
        )

        # 更新输入通道数为当前阶段的输出通道数
        self.in_channels = out_channels * block.expansion

        # 后续残差块保持尺寸不变，不再下采样
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        """He (Kaiming) 初始化，适用于 ReLU 激活函数。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming 正态初始化，mode='fan_out' 表示基于输出通道数计算
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm 的 gamma 初始化为 1，beta 初始化为 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入图像，形状 (batch, 3, H, W)
        Returns:
            分类 logits，形状 (batch, num_classes)
        """
        # 初始卷积 + 池化
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 四个残差阶段
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局池化 + 分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(num_classes: int = 1000) -> ResNet:
    """构建 ResNet-18（使用 BasicBlock，每阶段 [2,2,2,2]）。"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet34(num_classes: int = 1000) -> ResNet:
    """构建 ResNet-34（使用 BasicBlock，每阶段 [3,4,6,3]）。"""
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes: int = 1000) -> ResNet:
    """构建 ResNet-50（使用 Bottleneck，每阶段 [3,4,6,3]）。"""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


if __name__ == "__main__":
    # ==================== 自测区域 ====================
    torch.manual_seed(42)

    batch_size = 2
    num_classes = 10
    x = torch.randn(batch_size, 3, 224, 224)

    # 1. 测试 BasicBlock（ResNet-18/34 使用）
    print("=" * 60)
    print("Test 1: BasicBlock forward pass")
    basic_block = BasicBlock(in_channels=64, out_channels=64, stride=1)
    x_basic = torch.randn(batch_size, 64, 56, 56)
    out_basic = basic_block(x_basic)
    assert out_basic.shape == (batch_size, 64, 56, 56)
    print(f"BasicBlock input shape:  {x_basic.shape}")
    print(f"BasicBlock output shape: {out_basic.shape}")

    # 测试带下采样的 BasicBlock
    basic_block_down = BasicBlock(
        in_channels=64, out_channels=128, stride=2,
        downsample=nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(128),
        )
    )
    out_basic_down = basic_block_down(x_basic)
    assert out_basic_down.shape == (batch_size, 128, 28, 28)
    print(f"BasicBlock (stride=2) output shape: {out_basic_down.shape}")

    # 2. 测试 Bottleneck（ResNet-50/101/152 使用）
    print("=" * 60)
    print("Test 2: Bottleneck forward pass")
    # Bottleneck 的 expansion=4，当 in_channels != out_channels * expansion 时需要 downsample
    bottleneck = Bottleneck(
        in_channels=64, out_channels=64, stride=1,
        downsample=nn.Sequential(
            nn.Conv2d(64, 64 * Bottleneck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(64 * Bottleneck.expansion),
        )
    )
    x_bottleneck = torch.randn(batch_size, 64, 56, 56)
    out_bottleneck = bottleneck(x_bottleneck)
    assert out_bottleneck.shape == (batch_size, 256, 56, 56)  # 64 * 4 = 256
    print(f"Bottleneck input shape:  {x_bottleneck.shape}")
    print(f"Bottleneck output shape: {out_bottleneck.shape}")

    # 3. 测试完整 ResNet-18
    print("=" * 60)
    print("Test 3: ResNet-18")
    model_18 = resnet18(num_classes=num_classes)
    out_18 = model_18(x)
    assert out_18.shape == (batch_size, num_classes)
    print(f"ResNet-18 output shape: {out_18.shape}")

    # 统计参数量
    total_params = sum(p.numel() for p in model_18.parameters())
    print(f"ResNet-18 total parameters: {total_params:,}")

    # 4. 测试完整 ResNet-50
    print("=" * 60)
    print("Test 4: ResNet-50")
    model_50 = resnet50(num_classes=num_classes)
    out_50 = model_50(x)
    assert out_50.shape == (batch_size, num_classes)
    print(f"ResNet-50 output shape: {out_50.shape}")

    total_params_50 = sum(p.numel() for p in model_50.parameters())
    print(f"ResNet-50 total parameters: {total_params_50:,}")

    # 5. 验证梯度回传
    print("=" * 60)
    print("Test 5: Gradient backpropagation")
    loss = out_50.sum()
    loss.backward()
    has_grad = any(p.grad is not None for p in model_50.parameters())
    assert has_grad, "No gradients computed!"
    print("Gradient backpropagation: OK")

    print("=" * 60)
    print("All ResNet tests passed!")
