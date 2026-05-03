"""
SwiGLU (Swish Gated Linear Unit) Activation

一句话描述: LLaMA、PaLM 等主流大模型的标准 FFN 激活单元，门控机制与 Swish 激活的结合。

核心思想:
SwiGLU 是 GLU (Gated Linear Unit) 的 Swish 变体。它将输入通过两个并行的线性投影，
一个经过 Swish(SiLU) 激活后与另一个逐元素相乘，形成门控结构。
相比单一激活函数，SwiGLU 具有更强的表达能力，但引入了三矩阵乘法和参数量陷阱。

时间/空间复杂度:
- 时间复杂度: O(n) — 逐元素门控操作
- 空间复杂度: O(n) — 中间门控张量

面试频率: 极高 (LLaMA 架构必考点，参数计数陷阱)
"""

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """
    SwiGLU 激活模块（纯激活视角，不含输出投影）。

    数学公式:
        SwiGLU(x, W, V) = Swish(xW) ⊙ (xV)

    其中:
        - xW 为门控分支 (gate)，经过 Swish 激活
        - xV 为值分支 (value)，原样保留
        - ⊙ 表示逐元素相乘 (Hadamard product)

    参数:
        d_model: 输入/输出维度
        d_ff: 中间投影维度 (注意 SwiGLU 场景下的特殊处理)
        bias: 是否使用偏置项

    注意:
        在完整 FFN 中，SwiGLU 后通常还需接一个输出线性层 W2。
        本模块仅实现门控激活部分，方便与不同 FFN 结构组合。
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # 两个并行线性投影: gate 分支与 value 分支
        # 在标准 SwiGLU FFN 中，这两个投影的输出维度均为 d_ff
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.value_proj = nn.Linear(d_model, d_ff, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x: 输入张量，形状 (..., d_model)

        返回:
            门控激活后的张量，形状 (..., d_ff)
        """
        # gate 分支: 经过 Swish (SiLU) 激活
        gate = torch.nn.functional.silu(self.gate_proj(x))
        # value 分支: 原样保留
        value = self.value_proj(x)
        # 逐元素相乘，完成门控
        return gate * value


class SwiGLUCombined(nn.Module):
    """
    SwiGLU 门控激活 + 输出投影的完整组合（单模块简化版）。

    数学公式:
        y = (Swish(xW_g) ⊙ (xW_v)) W_o

    参数:
        d_model: 输入/输出维度
        d_ff: 中间投影维度 (gate/value 分支的输出维度)
        bias: 是否在线性层中使用偏置

    注意:
        此模块包含三个线性层 (W_g, W_v, W_o)，参数量为 3 * d_model * d_ff。
        若要与标准 ReLU FFN (参数量 2 * d_model * d_ff) 保持总参数量一致，
        需令 d_ff = 2/3 * d_ff_original，这是面试中最常见的参数计数陷阱。
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # gate 投影: d_model -> d_ff
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        # value 投影: d_model -> d_ff
        self.value_proj = nn.Linear(d_model, d_ff, bias=bias)
        # 输出投影: d_ff -> d_model
        self.out_proj = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 门控激活部分
        gate = torch.nn.functional.silu(self.gate_proj(x))
        value = self.value_proj(x)
        hidden = gate * value
        # 输出投影，将维度映射回 d_model
        return self.out_proj(hidden)


def compute_swiglu_ffn_params(d_model: int, d_ff: int, bias: bool = False) -> int:
    """
    计算 SwiGLU FFN 的总参数量，用于验证参数计数陷阱。

    参数:
        d_model: 输入/输出维度
        d_ff: 中间维度
        bias: 是否含偏置

    返回:
        总参数量
    """
    # 三个矩阵: W_g(d_model, d_ff), W_v(d_model, d_ff), W_o(d_ff, d_model)
    mat_params = 3 * d_model * d_ff
    bias_params = 3 * d_ff if bias else 0
    return mat_params + bias_params


def compute_relu_ffn_params(d_model: int, d_ff: int, bias: bool = False) -> int:
    """
    计算标准 ReLU FFN 的总参数量，用于与 SwiGLU 对比。

    参数:
        d_model: 输入/输出维度
        d_ff: 中间维度
        bias: 是否含偏置

    返回:
        总参数量
    """
    # 两个矩阵: W1(d_model, d_ff), W2(d_ff, d_model)
    mat_params = 2 * d_model * d_ff
    bias_params = d_ff + d_model if bias else 0
    return mat_params + bias_params


if __name__ == "__main__":
    torch.manual_seed(42)
    d_model = 512
    # 面试陷阱: 若 d_ff 与 ReLU FFN 相同，SwiGLU 参数量会多 50%
    d_ff = 2048

    x = torch.randn(2, 16, d_model)  # (batch, seq_len, d_model)

    # 测试 SwiGLU 纯激活模块
    swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, bias=False)
    out = swiglu(x)
    assert out.shape == (2, 16, d_ff)
    print("SwiGLU 激活输出形状:", out.shape)

    # 测试 SwiGLU 完整模块 (含输出投影)
    swiglu_combined = SwiGLUCombined(d_model=d_model, d_ff=d_ff, bias=False)
    out2 = swiglu_combined(x)
    assert out2.shape == (2, 16, d_model)
    print("SwiGLU 完整模块输出形状:", out2.shape)

    # 参数计数陷阱演示
    relu_params = compute_relu_ffn_params(d_model, d_ff, bias=False)
    swiglu_params = compute_swiglu_ffn_params(d_model, d_ff, bias=False)
    print(f"ReLU FFN 参数量: {relu_params:,}")
    print(f"SwiGLU FFN 参数量 (同 d_ff): {swiglu_params:,}")
    print(f"SwiGLU 是 ReLU 的 {swiglu_params / relu_params:.2f} 倍")

    # 保持总参数量一致的正确做法: d_ff_swiglu = 2/3 * d_ff_relu
    d_ff_swiglu = int(2 * d_ff / 3)
    swiglu_params_matched = compute_swiglu_ffn_params(d_model, d_ff_swiglu, bias=False)
    print(f"\n若令 SwiGLU 的 d_ff = 2/3 * {d_ff} = {d_ff_swiglu}")
    print(f"则 SwiGLU 参数量: {swiglu_params_matched:,}")
    print(f"与 ReLU FFN 参数量 {relu_params:,} 接近一致")

    print("\nSwiGLU 自测通过!")
