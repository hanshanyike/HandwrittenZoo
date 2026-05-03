"""
Feed-Forward Network: Standard FFN vs SwiGLU FFN

一句话描述: Transformer 中位置级前馈网络的两种主流实现——标准 ReLU FFN 与 SwiGLU FFN 的对比实现。

核心思想:
- 标准 FFN: 两个线性层夹一个 ReLU/GELU，结构简单，参数量 2*d_model*d_ff。
- SwiGLU FFN: 三个线性层配合门控激活，表达力强，参数量 3*d_model*d_ff。
  为保持总参数量一致，SwiGLU 的中间维度需压缩为 2/3。

时间/空间复杂度:
- 标准 FFN: 时间 O(batch*seq*d_model*d_ff), 空间 O(batch*seq*d_ff)
- SwiGLU FFN: 时间 O(batch*seq*d_model*d_ff), 空间 O(batch*seq*d_ff) (调整 d_ff 后)

面试频率: 极高 (架构选型与参数计数必考)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardReLU(nn.Module):
    """
    标准 ReLU FFN (Transformer 原始论文版本)。

    结构:
        x -> Linear(d_model, d_ff) -> ReLU -> Linear(d_ff, d_model) -> y

    数学公式:
        FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    参数量:
        2 * d_model * d_ff + (d_ff + d_model) 若使用偏置
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # 升维投影: d_model -> d_ff
        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        # 降维投影: d_ff -> d_model
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一步: 升维并激活
        hidden = F.relu(self.fc1(x))
        hidden = self.dropout(hidden)
        # 第二步: 降维映射回原始维度
        return self.fc2(hidden)


class FeedForwardGELU(nn.Module):
    """
    GELU FFN (BERT、GPT 等模型的标准配置)。

    结构:
        x -> Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model) -> y

    数学公式:
        FFN(x) = \text{GELU}(x W_1 + b_1) W_2 + b_2

    参数量:
        与 ReLU FFN 相同: 2 * d_model * d_ff
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.fc1 = nn.Linear(d_model, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用 GELU 替代 ReLU，平滑性更好
        hidden = F.gelu(self.fc1(x))
        hidden = self.dropout(hidden)
        return self.fc2(hidden)


class FeedForwardSwiGLU(nn.Module):
    """
    SwiGLU FFN (LLaMA、PaLM、Mistral 等模型的标准配置)。

    结构:
        x ─┬─> Linear(d_model, d_ff) ──> SiLU ──┐
           │                                    * ──> Linear(d_ff, d_model) -> y
           └─> Linear(d_model, d_ff) ───────────┘

    数学公式:
        FFN_SwiGLU(x) = (SiLU(x W_g) * (x W_v)) W_o

    参数量:
        3 * d_model * d_ff  (三个线性层)
        若要与标准 FFN 保持相同参数量，应设置 d_ff = 2/3 * d_ff_standard

    注意:
        现代大模型通常设置 bias=False 以节省参数量。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # gate 投影: 输入 -> d_ff，后续经过 SiLU 作为门控信号
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        # value 投影: 输入 -> d_ff，提供被门控的值
        self.value_proj = nn.Linear(d_model, d_ff, bias=bias)
        # 输出投影: d_ff -> d_model，将门控结果映射回输入维度
        self.out_proj = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gate 分支: SiLU 激活
        gate = F.silu(self.gate_proj(x))
        # value 分支: 原样保留
        value = self.value_proj(x)
        # 门控相乘: 逐元素选择特征
        hidden = gate * value
        hidden = self.dropout(hidden)
        # 输出投影
        return self.out_proj(hidden)


def count_parameters(module: nn.Module) -> int:
    """
    统计模块的可训练参数量。

    参数:
        module: PyTorch 模块

    返回:
        可训练参数总数
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def create_matched_swiglu(d_model: int, d_ff_standard: int, **kwargs) -> FeedForwardSwiGLU:
    """
    创建与标准 FFN 参数量对齐的 SwiGLU FFN。

    参数:
        d_model: 输入/输出维度
        d_ff_standard: 标准 FFN 的中间维度 (如 4*d_model)
        **kwargs: 传递给 FeedForwardSwiGLU 的其他参数

    返回:
        参数量近似相等的 SwiGLU FFN 实例

    原理:
        标准 FFN 参数量: 2 * d_model * d_ff_standard
        SwiGLU 参数量: 3 * d_model * d_ff_swiglu
        令两者相等: d_ff_swiglu = (2/3) * d_ff_standard
    """
    d_ff_swiglu = int(2 * d_ff_standard / 3)
    return FeedForwardSwiGLU(d_model=d_model, d_ff=d_ff_swiglu, **kwargs)


if __name__ == "__main__":
    torch.manual_seed(42)
    d_model = 512
    d_ff = 2048
    batch, seq_len = 2, 16

    x = torch.randn(batch, seq_len, d_model)

    # 1. 测试 ReLU FFN
    ffn_relu = FeedForwardReLU(d_model, d_ff, dropout=0.1, bias=True)
    out_relu = ffn_relu(x)
    assert out_relu.shape == (batch, seq_len, d_model)
    params_relu = count_parameters(ffn_relu)
    print(f"ReLU FFN: 输出形状 {out_relu.shape}, 参数量 {params_relu:,}")

    # 2. 测试 GELU FFN
    ffn_gelu = FeedForwardGELU(d_model, d_ff, dropout=0.1, bias=True)
    out_gelu = ffn_gelu(x)
    assert out_gelu.shape == (batch, seq_len, d_model)
    params_gelu = count_parameters(ffn_gelu)
    print(f"GELU FFN: 输出形状 {out_gelu.shape}, 参数量 {params_gelu:,}")

    # 3. 测试 SwiGLU FFN (同 d_ff，参数量更多)
    ffn_swiglu = FeedForwardSwiGLU(d_model, d_ff, dropout=0.1, bias=False)
    out_swiglu = ffn_swiglu(x)
    assert out_swiglu.shape == (batch, seq_len, d_model)
    params_swiglu = count_parameters(ffn_swiglu)
    print(f"SwiGLU FFN (同 d_ff): 输出形状 {out_swiglu.shape}, 参数量 {params_swiglu:,}")

    # 4. 测试参数量对齐的 SwiGLU FFN
    ffn_swiglu_matched = create_matched_swiglu(d_model, d_ff, dropout=0.1, bias=False)
    out_swiglu_matched = ffn_swiglu_matched(x)
    params_swiglu_matched = count_parameters(ffn_swiglu_matched)
    print(
        f"SwiGLU FFN (2/3 d_ff): 输出形状 {out_swiglu_matched.shape}, "
        f"参数量 {params_swiglu_matched:,}, d_ff={ffn_swiglu_matched.d_ff}"
    )

    # 验证参数量对齐
    assert abs(params_swiglu_matched - params_relu) < d_model * 10  # 允许取整误差
    print(f"\n参数量对齐验证: ReLU={params_relu:,}, SwiGLU(2/3)={params_swiglu_matched:,}, 差异={abs(params_swiglu_matched - params_relu):,}")

    print("\n所有 FFN 自测通过!")
