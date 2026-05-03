"""
Quantization (Basic)

Algorithm: Post-Training Quantization (PTQ) for LLM weights and activations.
Core Idea: Map high-precision floating-point tensors (FP32/FP16) to low-bit
    integers (INT8/INT4) using scale and zero-point, reducing memory footprint
    and accelerating inference via integer arithmetic.
Time Complexity: O(n) per tensor for calibration/quantization; inference uses
    lower-bit ops which are 2~4x faster on supported hardware.
Space Complexity: INT8 reduces weight size by 4x vs FP32; INT4 by 8x.
Interview Frequency: High — essential model compression technique in production.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 工具函数：对称 / 非对称量化与反量化
# ---------------------------------------------------------------------------

def symmetric_quantize(
    x: torch.Tensor,
    bits: int = 8,
    dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对称量化：假设浮点范围关于 0 对称，映射到整数对称区间 [-Q_max, Q_max]。

    公式：
        scale = max(|x|) / (2^{bits-1} - 1)
        x_q = round(x / scale)

    Args:
        x: 输入浮点张量
        bits: 量化位数（8 或 4）
        dim: 若指定，则沿该维度逐通道（per-channel）计算 scale；否则全局（per-tensor）

    Returns:
        x_q: 量化后的整型张量（以 float 容器存储，实际取整数值）
        scale: 缩放因子，形状与 x 相同或沿 dim 压缩
    """
    qmax = 2 ** (bits - 1) - 1

    if dim is None:
        # per-tensor：取全局最大绝对值
        abs_max = x.abs().max()
        scale = abs_max / qmax
        # 防止除零
        if scale == 0:
            scale = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        x_q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)
    else:
        # per-channel：沿指定维度求最大绝对值，并保持维度用于广播
        abs_max = x.abs().max(dim=dim, keepdim=True)[0]
        scale = abs_max / qmax
        scale[scale == 0] = 1.0
        x_q = torch.clamp(torch.round(x / scale), -qmax - 1, qmax)

    return x_q, scale


def symmetric_dequantize(x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """对称反量化：x = x_q * scale"""
    return x_q * scale


def asymmetric_quantize(
    x: torch.Tensor,
    bits: int = 8,
    dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    非对称量化：不假设关于 0 对称，使用 zero-point 将任意浮点区间映射到整型区间。

    公式：
        x_q = round((x - x_min) / scale) + z
        scale = (x_max - x_min) / (2^{bits} - 1)
        z = round(-x_min / scale)

    适用于激活值分布明显偏向正或负的场景（如 ReLU 输出全为非负）。

    Args:
        x: 输入浮点张量
        bits: 量化位数
        dim: 若指定，则逐通道计算 scale 和 zero_point

    Returns:
        x_q: 量化后的整型张量（取值范围 [0, 2^{bits}-1]）
        scale: 缩放因子
        zero_point: 零点偏移
    """
    qmax = 2 ** bits - 1

    if dim is None:
        x_min = x.min()
        x_max = x.max()
        scale = (x_max - x_min) / qmax
        if scale == 0:
            scale = torch.tensor(1.0, dtype=x.dtype, device=x.device)
            zero_point = torch.tensor(0, dtype=x.dtype, device=x.device)
        else:
            zero_point = torch.round(-x_min / scale)
        x_q = torch.clamp(torch.round(x / scale) + zero_point, 0, qmax)
    else:
        x_min = x.min(dim=dim, keepdim=True)[0]
        x_max = x.max(dim=dim, keepdim=True)[0]
        scale = (x_max - x_min) / qmax
        scale[scale == 0] = 1.0
        zero_point = torch.round(-x_min / scale)
        x_q = torch.clamp(torch.round(x / scale) + zero_point, 0, qmax)

    return x_q, scale, zero_point


def asymmetric_dequantize(
    x_q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    """非对称反量化：x = (x_q - zero_point) * scale"""
    return (x_q - zero_point) * scale


# ---------------------------------------------------------------------------
# 模拟量化线性层：仅用于教学演示，展示量化如何应用于 nn.Linear
# ---------------------------------------------------------------------------

class QuantizedLinear(nn.Module):
    """
    简化的仅权重量化线性层（Weight-Only Quantization）。

    实际部署中，INT8/INT4 权重会与 INT8 激活或 FP16 激活相乘，
    并通过底层 kernel（如 CUTLASS、Marlin）加速。这里只做数值模拟。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        symmetric: bool = True,
        per_channel: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel

        # 注册浮点权重（训练后通过 quantize_weight 进行量化）
        self.register_buffer("weight_fp", torch.randn(out_features, in_features))
        # 量化后的权重、scale、zero_point
        self.register_buffer("weight_q", None)
        self.register_buffer("w_scale", None)
        self.register_buffer("w_zero_point", None)
        self._quantized = False

    def quantize_weight(self):
        """对当前浮点权重执行量化，并保存参数。"""
        dim = 0 if self.per_channel else None  # 输出通道维度
        if self.symmetric:
            w_q, scale = symmetric_quantize(self.weight_fp, bits=self.bits, dim=dim)
            self.register_buffer("weight_q", w_q)
            self.register_buffer("w_scale", scale)
        else:
            w_q, scale, zp = asymmetric_quantize(self.weight_fp, bits=self.bits, dim=dim)
            self.register_buffer("weight_q", w_q)
            self.register_buffer("w_scale", scale)
            self.register_buffer("w_zero_point", zp)
        self._quantized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。若已量化，则反量化回浮点后再做矩阵乘（模拟量化推理）。

        真实推理中不会反量化回 FP32，而是直接用 INT kernel 计算：
            y = x @ (weight_q * scale)   (symmetric)
        """
        if not self._quantized:
            return F.linear(x, self.weight_fp)

        if self.symmetric:
            w_fp = symmetric_dequantize(self.weight_q, self.w_scale)
        else:
            w_fp = asymmetric_dequantize(self.weight_q, self.w_scale, self.w_zero_point)

        return F.linear(x, w_fp)


# 需要在这里导入 F，因为前面定义时还未用到
import torch.nn.functional as F


def compute_quantization_error(
    x: torch.Tensor, bits: int = 8, symmetric: bool = True
) -> float:
    """
    计算量化-反量化后的均方误差（MSE），用于评估量化精度损失。

    Returns:
        mse: 浮点数
    """
    if symmetric:
        x_q, scale = symmetric_quantize(x, bits=bits)
        x_dq = symmetric_dequantize(x_q, scale)
    else:
        x_q, scale, zp = asymmetric_quantize(x, bits=bits)
        x_dq = asymmetric_dequantize(x_q, scale, zp)
    mse = ((x - x_dq) ** 2).mean().item()
    return mse


if __name__ == "__main__":
    torch.manual_seed(42)

    # 1. 对称 / 非对称量化演示
    x = torch.randn(10, 20)
    print("=== Per-Tensor Quantization Demo ===")
    for bits in (8, 4):
        for sym in (True, False):
            mse = compute_quantization_error(x, bits=bits, symmetric=sym)
            mode = "sym" if sym else "asym"
            print(f"  INT{bits} {mode:4s} MSE: {mse:.6f}")

    # 2. 逐通道量化演示
    weight = torch.randn(32, 64)  # out_features x in_features
    print("\n=== Per-Channel Weight Quantization ===")
    w_q_sym, s_sym = symmetric_quantize(weight, bits=8, dim=0)
    w_dq_sym = symmetric_dequantize(w_q_sym, s_sym)
    print(f"  INT8 sym  per-channel MSE: {((weight - w_dq_sym) ** 2).mean().item():.6f}")

    w_q_asym, s_asym, zp_asym = asymmetric_quantize(weight, bits=8, dim=0)
    w_dq_asym = asymmetric_dequantize(w_q_asym, s_asym, zp_asym)
    print(f"  INT8 asym per-channel MSE: {((weight - w_dq_asym) ** 2).mean().item():.6f}")

    # 3. QuantizedLinear 自测
    print("\n=== QuantizedLinear Self-Test ===")
    lin = QuantizedLinear(64, 32, bits=8, symmetric=True, per_channel=True)
    lin.quantize_weight()
    inp = torch.randn(4, 64)
    out_q = lin(inp)
    # 与浮点结果对比
    lin_fp = nn.Linear(64, 32, bias=False)
    lin_fp.weight.data = lin.weight_fp
    out_fp = lin_fp(inp)
    diff = (out_q - out_fp).abs().max().item()
    print(f"  Max diff between quantized and fp output: {diff:.6f}")

    print("\nQuantization self-test passed.")
