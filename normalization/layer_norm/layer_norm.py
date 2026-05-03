"""
Layer Normalization (LayerNorm)

Algorithm: 层归一化 —— 对单个样本的特征维度进行归一化，使模型在不同批量大小和序列长度下保持稳定。
Core Idea: 与 BatchNorm 跨样本统计不同，LayerNorm 仅在单个样本内部计算均值和方差，
         因此与 batch size 无关，特别适合 Transformer、RNN 等序列模型。
Complexity:
    - 时间复杂度: O(B * T * D) —— 每个元素仅被访问常数次
    - 空间复杂度: O(D) —— 可学习参数 gamma 和 beta 的维度
Interview Frequency: 极高（Transformer 核心组件，面试必考）

References:
    - Ba et al., "Layer Normalization", 2016
    - Vaswani et al., "Attention Is All You Need", NeurIPS 2017
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    层归一化 (Layer Normalization) 的从零实现。

    对输入张量的最后 `normalized_shape` 维度进行归一化：
        1. 计算该维度上的均值 mean 和方差 var
        2. 做标准化: (x - mean) / sqrt(var + eps)
        3. 做可学习的仿射变换: gamma * x_norm + beta

    与 PyTorch 官方 `nn.LayerNorm` 数值等价（使用有偏方差 unbiased=False）。
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ):
        """
        初始化 LayerNorm 模块。

        Args:
            normalized_shape: 需要归一化的特征维度大小（如 embedding_dim）
            eps: 加到分母上的极小值，防止除零，默认 1e-5
            elementwise_affine: 是否使用可学习的 gamma 缩放参数
            bias: 是否使用可学习的 beta 偏移参数（仅在 elementwise_affine=True 时生效）
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # 可学习的缩放参数 gamma（对应论文中的 weight / scale）
        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            # 可学习的偏移参数 beta（对应论文中的 bias / shift）
            if bias:
                self.beta = nn.Parameter(torch.zeros(normalized_shape))
            else:
                self.register_parameter("beta", None)
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量，形状为 (..., normalized_shape)，
               例如 NLP 中常见的 (batch_size, seq_len, embedding_dim)

        Returns:
            归一化后的张量，形状与输入相同
        """
        # 计算最后 normalized_shape 维度的均值
        # keepdim=True 保持维度对齐，便于广播减法
        mean = x.mean(dim=-1, keepdim=True)

        # 计算有偏方差（与 PyTorch 官方实现保持一致）
        # unbiased=False 表示除以 N 而非 N-1
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # 标准化：减去均值并除以标准差（加 eps 防止数值不稳定）
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # 若启用可学习仿射变换，则应用 gamma 和 beta
        if self.elementwise_affine:
            x_norm = x_norm * self.gamma
            if self.beta is not None:
                x_norm = x_norm + self.beta

        return x_norm


class PreNormTransformerBlock(nn.Module):
    """
    Pre-Norm (前置归一化) Transformer 子层示例。

    结构: x + Sublayer(LayerNorm(x))
    即先将输入归一化，再送入子层（Attention 或 FFN），最后做残差连接。

    优势: 训练更稳定，梯度在深层网络中传播更顺畅，是现代大模型（GPT、LLaMA 等）的主流选择。
    """

    def __init__(self, d_model: int, sublayer: nn.Module):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm: 先归一化，再经过子层，最后残差连接
        return x + self.sublayer(self.norm(x))


class PostNormTransformerBlock(nn.Module):
    """
    Post-Norm (后置归一化) Transformer 子层示例。

    结构: LayerNorm(x + Sublayer(x))
    即先做残差连接，再对结果进行归一化。

    特点: 原始 Transformer 论文采用此结构，但深层网络训练时梯度爆炸/消失风险更高。
    """

    def __init__(self, d_model: int, sublayer: nn.Module):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.sublayer = sublayer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Post-Norm: 先经过子层并残差连接，再归一化
        return self.norm(x + self.sublayer(x))


if __name__ == "__main__":
    # ==================== 自测区域 ====================
    torch.manual_seed(42)

    batch_size, seq_len, d_model = 2, 4, 8
    x = torch.randn(batch_size, seq_len, d_model)

    # 1. 测试自定义 LayerNorm 与官方实现数值等价性
    print("=" * 50)
    print("Test 1: Numerical equivalence with PyTorch official LayerNorm")
    my_ln = LayerNorm(d_model)
    torch_ln = nn.LayerNorm(d_model)

    # 同步参数以确保公平比较
    with torch.no_grad():
        my_ln.gamma.copy_(torch_ln.weight)
        my_ln.beta.copy_(torch_ln.bias)

    out_my = my_ln(x)
    out_torch = torch_ln(x)

    are_equal = torch.allclose(out_my, out_torch, atol=1e-6)
    print(f"Custom vs Official allclose: {are_equal}")
    assert are_equal, "Custom LayerNorm does not match official implementation!"

    # 2. 测试 Pre-Norm 与 Post-Norm 结构
    print("=" * 50)
    print("Test 2: Pre-Norm and Post-Norm block shapes")

    dummy_sublayer = nn.Linear(d_model, d_model)

    pre_block = PreNormTransformerBlock(d_model, dummy_sublayer)
    post_block = PostNormTransformerBlock(d_model, dummy_sublayer)

    out_pre = pre_block(x)
    out_post = post_block(x)

    print(f"Input shape:  {x.shape}")
    print(f"Pre-Norm output shape:  {out_pre.shape}")
    print(f"Post-Norm output shape: {out_post.shape}")
    assert out_pre.shape == x.shape
    assert out_post.shape == x.shape

    # 3. 测试 elementwise_affine=False（无学习参数）
    print("=" * 50)
    print("Test 3: LayerNorm without affine parameters")
    ln_no_affine = LayerNorm(d_model, elementwise_affine=False)
    out_no_affine = ln_no_affine(x)
    print(f"Output shape (no affine): {out_no_affine.shape}")
    assert out_no_affine.shape == x.shape

    print("=" * 50)
    print("All LayerNorm tests passed!")
