"""
Attention with Linear Biases (ALiBi, 线性偏置注意力)

Algorithm: Attention with Linear Biases (ALiBi)
One-line: 不添加位置嵌入到词嵌入，而是在 attention score 上直接添加
         与 query-key 距离成线性关系的负偏置，实现相对位置编码并增强长度外推。

Core idea:
    传统位置编码（正弦、可学习、RoPE）都需要修改输入嵌入或 Q/K 表示。
    ALiBi 的核心洞察是：位置信息只需要影响 attention 的分布，因此可以直接
    在 attention score（即 QK^T / sqrt(d_k)）上添加一个与距离相关的偏置。
    具体地，对于 query 位置 i 和 key 位置 j，添加偏置 m * (-|i-j|)。
    其中 m 是每个 attention head 特有的固定斜率（非学习参数）。
    距离越远的 key，受到的负偏置越大，softmax 后的权重自然衰减。
    ALiBi 被 MPT、BLOOM、Baichuan 等模型采用，以优异的长度外推能力著称。

Complexity:
    - 时间复杂度：O(n_heads * seq_len * seq_len) 生成偏置矩阵；与 attention 同阶
    - 空间复杂度：O(n_heads * max_seq_len * max_seq_len) 缓存，或动态生成

Interview frequency: High (外推能力考点，MPT/Baichuan 核心)
"""

import math
import torch
import torch.nn as nn


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """
    计算 ALiBi 中每个 attention head 的斜率 m。

    当 n_heads 是 2 的幂时，斜率构成等比数列：
        m_h = 2^(-8 * h / n_heads),  h = 1, 2, ..., n_heads
    当 n_heads 不是 2 的幂时，先计算最接近的较小 2 的幂的斜率，
    再对 2*closest_power_of_2 取偶数索引补充剩余 head。

    参数:
        n_heads (int): 注意力头数。

    返回:
        Tensor: 形状为 (n_heads,) 的斜率张量。
    """

    def get_slopes_power_of_2(n: int) -> list[float]:
        """当 n 是 2 的幂时的斜率计算。"""
        # 公式: start = 2^(-(2^-(log2(n) - 3))) = 2^(-8/n)
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    # 判断 n_heads 是否是 2 的幂
    if math.log2(n_heads).is_integer():
        slopes = get_slopes_power_of_2(n_heads)
    else:
        # 找到最接近的较小 2 的幂
        closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
        # 先计算 closest_power_of_2 个斜率
        slopes = get_slopes_power_of_2(closest_power_of_2)
        # 剩余 head 的斜率：从 2*closest_power_of_2 的斜率中取偶数索引
        extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)
        extra_slopes = extra_slopes[0::2][: n_heads - closest_power_of_2]
        slopes.extend(extra_slopes)

    return torch.tensor(slopes, dtype=torch.float32)


def build_alibi_bias(n_heads: int, seq_len: int) -> torch.Tensor:
    """
    构建 ALiBi 偏置矩阵。

    参数:
        n_heads (int): 注意力头数。
        seq_len (int): 序列长度。

    返回:
        Tensor: 形状为 (n_heads, seq_len, seq_len) 的偏置矩阵。
                在因果注意力中，通常与上三角 mask 结合使用。
    """
    # 计算每个 head 的斜率: (n_heads,)
    slopes = get_alibi_slopes(n_heads)

    # 生成位置索引: 0, 1, ..., seq_len-1
    pos = torch.arange(seq_len, dtype=torch.float32)

    # 计算距离矩阵: -|i - j|，形状 (seq_len, seq_len)
    # 使用广播: pos[:, None] - pos[None, :] -> (seq_len, seq_len)
    distance_matrix = -torch.abs(pos.unsqueeze(1) - pos.unsqueeze(0))

    # 将斜率与距离矩阵相乘，得到每个 head 的偏置
    # slopes: (n_heads, 1, 1) * distance_matrix: (1, seq_len, seq_len) -> (n_heads, seq_len, seq_len)
    alibi_bias = slopes.view(n_heads, 1, 1) * distance_matrix.unsqueeze(0)

    return alibi_bias


class ALiBiPositionalBias(nn.Module):
    """
    ALiBi 位置偏置模块。

    参数:
        n_heads (int): 注意力头数。
        max_seq_len (int): 预计算的最大序列长度。
        causal (bool): 是否使用因果掩码（自回归模型通常设为 True）。
    """

    def __init__(self, n_heads: int, max_seq_len: int = 2048, causal: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.causal = causal

        # 预计算斜率并注册为 buffer
        slopes = get_alibi_slopes(n_heads)
        self.register_buffer("slopes", slopes.view(n_heads, 1, 1))

        # 预计算距离矩阵的模板: (max_seq_len, max_seq_len)
        pos = torch.arange(max_seq_len, dtype=torch.float32)
        distance_matrix = -torch.abs(pos.unsqueeze(1) - pos.unsqueeze(0))
        self.register_buffer("distance_matrix", distance_matrix)

        # 若使用因果注意力，预计算上三角 mask（True 表示需要 mask 的位置）
        if causal:
            causal_mask = torch.triu(
                torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1
            )
            self.register_buffer("causal_mask", causal_mask)
        else:
            self.causal_mask = None

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        生成当前序列长度的 ALiBi 偏置矩阵。

        参数:
            seq_len (int): 当前序列长度。

        返回:
            Tensor: 形状为 (n_heads, seq_len, seq_len) 的偏置矩阵。
                    若 causal=True，上三角已被置为负无穷（-inf）。
        """
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"输入序列长度 {seq_len} 超过预计算的最大长度 {self.max_seq_len}"
            )

        # 截取当前序列长度的距离矩阵: (seq_len, seq_len)
        dist = self.distance_matrix[:seq_len, :seq_len]

        # 计算偏置: (n_heads, seq_len, seq_len)
        bias = self.slopes * dist.unsqueeze(0)

        # 若使用因果注意力，将上三角置为 -inf
        if self.causal and self.causal_mask is not None:
            mask = self.causal_mask[:seq_len, :seq_len]
            bias = bias.masked_fill(mask.unsqueeze(0), float("-inf"))

        return bias


class ALiBiMultiHeadAttention(nn.Module):
    """
    带 ALiBi 偏置的简化多头注意力（用于演示 ALiBi 的集成方式）。

    参数:
        d_model (int): 模型维度。
        n_heads (int): 注意力头数。
        max_seq_len (int): 最大序列长度。
        dropout (float): Dropout 概率。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # ALiBi 偏置模块
        self.alibi = ALiBiPositionalBias(n_heads, max_seq_len, causal=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x (Tensor): 输入张量，形状 (batch, seq_len, d_model)。

        返回:
            Tensor: 输出张量，形状 (batch, seq_len, d_model)。
        """
        batch_size, seq_len, _ = x.shape

        # 投影并分头: (batch, seq_len, n_heads, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # 转置为 (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算 attention score: (batch, n_heads, seq_len, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 添加 ALiBi 偏置: (n_heads, seq_len, seq_len)
        alibi_bias = self.alibi(seq_len)
        scores = scores + alibi_bias.unsqueeze(0)

        # Softmax 并 dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和: (batch, n_heads, seq_len, head_dim)
        out = torch.matmul(attn_weights, v)

        # 合并多头并投影: (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)


if __name__ == "__main__":
    # ------------------- 自测代码 -------------------
    n_heads, seq_len = 8, 16
    alibi_bias_module = ALiBiPositionalBias(n_heads=n_heads, max_seq_len=512, causal=True)

    # 生成偏置矩阵
    bias = alibi_bias_module(seq_len)
    print("ALiBi 偏置矩阵形状:", bias.shape)  # (8, 16, 16)
    assert bias.shape == (n_heads, seq_len, seq_len)

    # 验证因果 mask：上三角应为 -inf
    if alibi_bias_module.causal:
        assert torch.isinf(bias[0, 0, 1]) and bias[0, 0, 1] < 0, "因果 mask 上三角应为 -inf"
        print("因果 mask 验证通过：上三角为 -inf")

    # 验证斜率单调性：head 索引越大，斜率应越小（绝对值）
    slopes = alibi_bias_module.slopes.squeeze()
    for i in range(n_heads - 1):
        assert slopes[i] > slopes[i + 1], "斜率应随 head 索引增大而减小"
    print("斜率单调性验证通过")

    # 验证距离特性：对角线（距离为0）的偏置应为 0
    diag = torch.diagonal(bias, dim1=-2, dim2=-1)  # (n_heads, seq_len)
    assert torch.allclose(diag, torch.zeros_like(diag)), "对角线（距离为0）偏置应为0"
    print("对角线偏置验证通过：距离为0时偏置为0")

    # 验证多头注意力模块
    d_model = 64
    mha = ALiBiMultiHeadAttention(d_model=d_model, n_heads=n_heads, max_seq_len=512)
    x = torch.randn(2, seq_len, d_model)
    out = mha(x)
    print("ALiBi MHA 输出形状:", out.shape)
    assert out.shape == x.shape

    print("ALiBi 自测通过！")
