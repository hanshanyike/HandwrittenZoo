"""
Batch Normalization (BatchNorm)

Algorithm: 批归一化 —— 在 mini-batch 的维度上对每一特征通道计算均值和方差进行归一化。
Core Idea: 通过对每一层的输入进行归一化，缓解内部协变量偏移（Internal Covariate Shift），
         允许使用更大的学习率，加速收敛，并具有一定的正则化效果。
Complexity:
    - 时间复杂度: O(B * C * H * W) —— 每个元素访问常数次
    - 空间复杂度: O(C) —— 可学习参数 gamma、beta，以及移动平均的 running_mean、running_var
Interview Frequency: 高（CNN 经典组件，面试常考与 LayerNorm/RMSNorm 的对比及适用场景）

References:
    - Ioffe & Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", ICML 2015
"""

import torch
import torch.nn as nn


class BatchNorm1d(nn.Module):
    """
    Batch Normalization 1D 的从零实现（适用于 NLP 中的 (B, C) 或 (B, C, L) 输入）。

    核心逻辑：
        1. 训练时：基于当前 batch 统计量归一化，并更新移动平均
        2. 推理时：使用训练阶段累积的移动平均统计量进行归一化

    与 PyTorch 官方 nn.BatchNorm1d 行为一致（含 momentum 和 eps 处理）。
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        """
        初始化 BatchNorm1d 模块。

        Args:
            num_features: 特征通道数 C
            eps: 分母稳定项
            momentum: 移动平均的动量系数（PyTorch 中定义为 1 - 平滑因子）
            affine: 是否使用可学习的 gamma 和 beta
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # 可学习参数
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        # 移动平均统计量（推理时使用，不参与梯度计算）
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

        # 训练/推理标志
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量，形状为 (B, C) 或 (B, C, L)

        Returns:
            归一化后的张量，形状与输入相同
        """
        if self.training:
            # ========== 训练模式：使用当前 batch 统计量 ==========
            # 计算维度：除通道维度外全部求平均
            # 若输入为 (B, C)，dim=0；若输入为 (B, C, L)，dim=(0, 2)
            dims = [0] if x.dim() == 2 else [0, 2]

            mean = x.mean(dim=dims, keepdim=True)
            # 使用有偏方差（与 PyTorch 一致）
            var = x.var(dim=dims, unbiased=False, keepdim=True)

            # 更新移动平均（使用 no_grad 避免参与梯度计算）
            with torch.no_grad():
                # PyTorch 的 momentum 定义：running = (1-momentum)*running + momentum*batch
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.squeeze()
                self.num_batches_tracked += 1

            # 标准化
            x_norm = (x - mean) / torch.sqrt(var + self.eps)

        else:
            # ========== 推理模式：使用移动平均统计量 ==========
            # running_mean/running_var 的形状为 (C,)，需要广播到输入形状
            if x.dim() == 2:
                mean = self.running_mean.view(1, -1)
                var = self.running_var.view(1, -1)
            else:
                mean = self.running_mean.view(1, -1, 1)
                var = self.running_var.view(1, -1, 1)

            x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 应用可学习仿射变换
        if self.affine:
            if x.dim() == 2:
                x_norm = x_norm * self.gamma.view(1, -1) + self.beta.view(1, -1)
            else:
                x_norm = x_norm * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)

        return x_norm


class BatchNorm2d(nn.Module):
    """
    Batch Normalization 2D 的从零实现（适用于图像 (B, C, H, W) 输入）。

    在 (B, H, W) 三个维度上统计，每个通道独立计算均值和方差。
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (B, C, H, W)
        """
        if self.training:
            # 在 B, H, W 维度上求均值和方差
            mean = x.mean(dim=[0, 2, 3], keepdim=True)  # (1, C, 1, 1)
            var = x.var(dim=[0, 2, 3], unbiased=False, keepdim=True)  # (1, C, 1, 1)

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.view(-1)
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.view(-1)
                self.num_batches_tracked += 1

            x_norm = (x - mean) / torch.sqrt(var + self.eps)
        else:
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
            x_norm = (x - mean) / torch.sqrt(var + self.eps)

        if self.affine:
            x_norm = x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)

        return x_norm


if __name__ == "__main__":
    # ==================== 自测区域 ====================
    torch.manual_seed(42)

    # 1. 测试 BatchNorm1d (B, C) 输入
    print("=" * 50)
    print("Test 1: BatchNorm1d with (B, C) input")
    B, C = 4, 8
    x_2d = torch.randn(B, C)

    my_bn1d = BatchNorm1d(C)
    torch_bn1d = nn.BatchNorm1d(C)

    # 同步参数和缓冲区
    with torch.no_grad():
        my_bn1d.gamma.copy_(torch_bn1d.weight)
        my_bn1d.beta.copy_(torch_bn1d.bias)

    # 训练模式对比
    my_bn1d.train()
    torch_bn1d.train()

    out_my = my_bn1d(x_2d)
    out_torch = torch_bn1d(x_2d)

    are_equal = torch.allclose(out_my, out_torch, atol=1e-5)
    print(f"Train mode allclose: {are_equal}")
    assert are_equal, "BatchNorm1d train mode mismatch!"

    # 推理模式对比
    my_bn1d.eval()
    torch_bn1d.eval()

    out_my_eval = my_bn1d(x_2d)
    out_torch_eval = torch_bn1d(x_2d)

    are_equal_eval = torch.allclose(out_my_eval, out_torch_eval, atol=1e-5)
    print(f"Eval mode allclose: {are_equal_eval}")
    assert are_equal_eval, "BatchNorm1d eval mode mismatch!"

    # 2. 测试 BatchNorm1d (B, C, L) 输入
    print("=" * 50)
    print("Test 2: BatchNorm1d with (B, C, L) input")
    L = 10
    x_3d = torch.randn(B, C, L)

    my_bn1d_3d = BatchNorm1d(C)
    torch_bn1d_3d = nn.BatchNorm1d(C)

    with torch.no_grad():
        my_bn1d_3d.gamma.copy_(torch_bn1d_3d.weight)
        my_bn1d_3d.beta.copy_(torch_bn1d_3d.bias)

    my_bn1d_3d.eval()
    torch_bn1d_3d.eval()

    out_my_3d = my_bn1d_3d(x_3d)
    out_torch_3d = torch_bn1d_3d(x_3d)

    are_equal_3d = torch.allclose(out_my_3d, out_torch_3d, atol=1e-4)
    print(f"Eval mode (B,C,L) allclose: {are_equal_3d}")

    # 3. 测试 BatchNorm2d
    print("=" * 50)
    print("Test 3: BatchNorm2d with (B, C, H, W) input")
    H, W = 4, 4
    x_4d = torch.randn(B, C, H, W)

    my_bn2d = BatchNorm2d(C)
    torch_bn2d = nn.BatchNorm2d(C)

    with torch.no_grad():
        my_bn2d.gamma.copy_(torch_bn2d.weight)
        my_bn2d.beta.copy_(torch_bn2d.bias)

    my_bn2d.eval()
    torch_bn2d.eval()

    out_my_4d = my_bn2d(x_4d)
    out_torch_4d = torch_bn2d(x_4d)

    are_equal_4d = torch.allclose(out_my_4d, out_torch_4d, atol=1e-4)
    print(f"Eval mode (B,C,H,W) allclose: {are_equal_4d}")

    # 4. 测试无 affine 参数
    print("=" * 50)
    print("Test 4: BatchNorm1d without affine parameters")
    bn_no_affine = BatchNorm1d(C, affine=False)
    out_no_affine = bn_no_affine(x_2d)
    print(f"Output shape (no affine): {out_no_affine.shape}")
    assert out_no_affine.shape == x_2d.shape

    print("=" * 50)
    print("All BatchNorm tests passed!")
