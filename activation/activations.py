"""
Activation Functions: ReLU, GELU, Swish (SiLU)

一句话描述: 深度学习中最常用的三种激活函数，从简单到平滑，逐代提升梯度流与表达能力。

核心思想:
- ReLU: 负值截断为零，计算极简，缓解梯度消失。
- GELU: 以标准正态分布的累积分布函数(CDF)对输入进行加权，平滑且非单调。
- Swish/SiLU: 输入与 sigmoid 的乘积，自门控机制，负值区域保留微弱梯度。

时间/空间复杂度:
- 时间复杂度: O(n) — 逐元素操作
- 空间复杂度: O(n) — 输出张量

面试频率: 极高 (GELU/Swish 为大模型基础考点)
"""

import math
import torch
import torch.nn as nn


class ReLU(nn.Module):
    """
    ReLU (Rectified Linear Unit) 激活函数。

    数学公式:
        ReLU(x) = max(0, x)

    特性:
        - 正区间线性，负区间恒为零
        - 计算量极小，收敛速度快
        - 缺陷: 负区间神经元"死亡"，梯度永远为零
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用 torch.clamp 实现 ReLU，直观且兼容 inplace
        if self.inplace:
            return x.clamp_(min=0)
        return x.clamp(min=0)


class GELU(nn.Module):
    """
    GELU (Gaussian Error Linear Unit) 激活函数。

    数学公式（精确）:
        GELU(x) = x * Φ(x) = 0.5 * x * (1 + erf(x / sqrt(2)))

    数学公式（tanh 近似，PyTorch 默认）:
        GELU(x) ≈ 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x^3) ))

    特性:
        - 平滑、非单调，处处可导
        - 负值区域保留微弱梯度，缓解神经元死亡
        - BERT、GPT、T5 等模型的默认激活函数
    """

    def __init__(self, approximate: str = "tanh"):
        super().__init__()
        # approximate 可取 "none"(精确) 或 "tanh"(近似)
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.approximate == "tanh":
            # tanh 近似实现，与 PyTorch 原生 gelu 保持一致
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
        else:
            # 精确实现，基于误差函数 erf
            return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Swish(nn.Module):
    """
    Swish (又称 SiLU, Sigmoid Linear Unit) 激活函数。

    数学公式:
        Swish(x) = x * σ(x) = x / (1 + exp(-x))

    特性:
        - 自门控: sigmoid 输出作为门控信号，输入本身作为被门控值
        - 平滑、非单调，负值区域梯度非零
        - Google Brain 提出，EfficientNet、LLaMA 等模型使用
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 逐元素乘积: x * sigmoid(x)
        return x * torch.sigmoid(x)


if __name__ == "__main__":
    # 简单自测: 验证形状、数值趋势及与 PyTorch 原生的对比
    torch.manual_seed(42)
    x = torch.randn(4, 8)

    # ReLU 测试
    relu = ReLU()
    relu_out = relu(x)
    assert relu_out.shape == x.shape
    assert (relu_out >= 0).all()
    print("ReLU 输出范围:", relu_out.min().item(), "~", relu_out.max().item())

    # GELU 测试
    gelu = GELU(approximate="tanh")
    gelu_out = gelu(x)
    assert gelu_out.shape == x.shape
    # 与 PyTorch 原生 F.gelu 对比
    ref_gelu = torch.nn.functional.gelu(x, approximate="tanh")
    assert torch.allclose(gelu_out, ref_gelu, atol=1e-6)
    print("GELU 与 PyTorch 原生差异:", (gelu_out - ref_gelu).abs().max().item())

    # Swish 测试
    swish = Swish()
    swish_out = swish(x)
    assert swish_out.shape == x.shape
    # 与 PyTorch 原生 F.silu 对比
    ref_silu = torch.nn.functional.silu(x)
    assert torch.allclose(swish_out, ref_silu, atol=1e-6)
    print("Swish 与 PyTorch 原生差异:", (swish_out - ref_silu).abs().max().item())

    print("所有激活函数自测通过!")
