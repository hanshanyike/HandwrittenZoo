"""
Root Mean Square Layer Normalization (RMSNorm)

Algorithm: 均方根层归一化 —— 在 LayerNorm 基础上去除均值中心化，仅保留 RMS 缩放。
Core Idea: 作者发现 LayerNorm 的成功主要归功于“重新缩放（re-scaling）”而非“去均值（mean centering）”。
         去掉均值计算后，RMSNorm 在保持相近性能的同时减少了计算量，成为 LLaMA、GPT-NeoX 等大模型的标配。
Complexity:
    - 时间复杂度: O(B * T * D) —— 比 LayerNorm 略低（省去一次均值计算和一次减法）
    - 空间复杂度: O(D) —— 仅需可学习缩放参数 gamma
Interview Frequency: 高（大模型优化热点，面试常考与 LayerNorm 的对比）

References:
    - Zhang & Sennrich, "Root Mean Square Layer Normalization", NeurIPS 2019
    - LLaMA, GPT-NeoX, T5 等主流大模型均采用 RMSNorm
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    均方根层归一化 (RMSNorm) 的从零实现。

    核心公式：
        RMS(x) = sqrt( mean(x_i^2) + eps )
        RMSNorm(x) = x / RMS(x) * gamma

    与 LayerNorm 的区别：
        - 不计算均值，不做去均值中心化
        - 通常只有 gamma（缩放参数），没有 beta（偏移参数）
        - 计算量略小，在大型 Transformer 中广泛使用
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        """
        初始化 RMSNorm 模块。

        Args:
            normalized_shape: 需要归一化的特征维度大小
            eps: 加到分母上的极小值，防止除零
            elementwise_affine: 是否使用可学习的 gamma 缩放参数
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # 可学习的缩放参数 gamma（weight）
        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
        else:
            self.register_parameter("gamma", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量，形状为 (..., normalized_shape)

        Returns:
            归一化后的张量，形状与输入相同
        """
        # 保存原始数据类型，因为中间计算可能在 float32 进行以保持数值稳定
        input_dtype = x.dtype

        # 将输入转为 float32 进行计算，避免低精度（如 float16/bfloat16）下的数值问题
        x_fp32 = x.float()

        # 计算均方根（RMS）：sqrt( mean(x^2) + eps )
        # 沿最后一维（特征维度）计算平方均值
        mean_square = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)

        # 归一化：逐元素除以 RMS
        x_norm = x_fp32 / rms

        # 若启用可学习参数，则应用 gamma 缩放
        if self.elementwise_affine:
            x_norm = x_norm * self.gamma

        # 转回原始数据类型
        return x_norm.to(input_dtype)


class RMSNormNaive(nn.Module):
    """
    RMSNorm 的极简实现（与 PyTorch 2.x 官方 nn.RMSNorm 逻辑一致）。

    此版本省略了显式的 dtype 转换，适合理解核心算法。
    实际大模型训练中推荐使用上方带 float32 中间计算的版本。
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # rsqrt = 1 / sqrt，比先 sqrt 再除更高效
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps))


if __name__ == "__main__":
    # ==================== 自测区域 ====================
    torch.manual_seed(42)

    batch_size, seq_len, d_model = 2, 4, 8
    x = torch.randn(batch_size, seq_len, d_model)

    # 1. 测试自定义 RMSNorm 与 PyTorch 2.x 官方实现数值等价性
    print("=" * 50)
    print("Test 1: Numerical equivalence with PyTorch official RMSNorm")

    # 检查 PyTorch 版本是否支持 nn.RMSNorm
    try:
        torch_rms = nn.RMSNorm(d_model, eps=1e-5)
        my_rms = RMSNorm(d_model, eps=1e-5)

        # 同步参数
        with torch.no_grad():
            my_rms.gamma.copy_(torch_rms.weight)

        out_my = my_rms(x)
        out_torch = torch_rms(x)

        are_equal = torch.allclose(out_my, out_torch, atol=1e-5)
        print(f"Custom vs Official allclose: {are_equal}")
        assert are_equal, "Custom RMSNorm does not match official implementation!"
    except AttributeError:
        print("PyTorch version < 2.1, nn.RMSNorm not available. Skipping official comparison.")
        my_rms = RMSNorm(d_model)
        out_my = my_rms(x)
        print(f"Custom RMSNorm output shape: {out_my.shape}")

    # 2. 测试极简版本
    print("=" * 50)
    print("Test 2: RMSNormNaive consistency")
    naive_rms = RMSNormNaive(d_model)
    out_naive = naive_rms(x)
    print(f"Naive RMSNorm output shape: {out_naive.shape}")
    assert out_naive.shape == x.shape

    # 3. 测试无学习参数版本
    print("=" * 50)
    print("Test 3: RMSNorm without affine parameters")
    rms_no_affine = RMSNorm(d_model, elementwise_affine=False)
    out_no_affine = rms_no_affine(x)
    print(f"Output shape (no affine): {out_no_affine.shape}")
    assert out_no_affine.shape == x.shape

    # 4. 验证 RMSNorm 输出均值的绝对值通常不为 0（与 LayerNorm 的关键区别）
    print("=" * 50)
    print("Test 4: Verify RMSNorm does NOT center mean to zero")
    sample_out = out_my[0, 0, :]
    mean_val = sample_out.mean().item()
    print(f"Sample output mean: {mean_val:.6f} (LayerNorm would be ~0)")

    print("=" * 50)
    print("All RMSNorm tests passed!")
