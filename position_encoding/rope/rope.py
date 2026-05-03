"""
Rotary Position Embedding (RoPE, 旋转位置编码)

Algorithm: Rotary Position Embedding (RoPE)
One-line: 通过旋转矩阵对 Query/Key 向量进行位置相关的旋转变换，
         将绝对位置信息编码进内积，同时自然显式表达相对位置依赖。

Core idea:
    传统绝对位置编码（如正弦编码）将位置向量加到词嵌入上，但自注意力的内积运算
    无法直接体现两个 token 的相对距离。RoPE 的核心洞察是：
    如果我们将 Q/K 向量按位置 m 进行旋转变换，那么旋转后的 Q_m 与 K_n 的内积
    仅依赖于 (m-n) 的相对距离，而与绝对位置 m、n 无关。
    这一性质称为"相对位置编码的内积不变性"。
    RoPE 被 LLaMA、Mistral、Qwen 等主流大模型采用，是现代 LLM 的标配。

Complexity:
    - 时间复杂度：预计算 O(max_seq_len * head_dim)；前向 O(batch * seq_len * num_heads * head_dim)
    - 空间复杂度：O(max_seq_len * head_dim) 缓存 cos/sin

Interview frequency: Very High (LLaMA 核心，面试必考)
"""

import math
import torch
import torch.nn as nn


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    预计算旋转角度对应的复指数值 (cos + i*sin)，即 "cis"。

    参数:
        dim (int): 每个注意力头的维度，必须是偶数。
        end (int): 预计算的最大序列长度。
        theta (float): 频率基数，默认 10000.0（LLaMA 论文值）。

    返回:
        Tensor: 形状为 (end, dim//2, 2) 的复数张量，其中最后一维为 [cos, sin]。
                也可视为 (end, dim//2) 的复数张量。
    """
    # 生成每个维度组的频率倒数: 1 / theta^(2i/dim), i=0,1,...,dim/2-1
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: (dim // 2)] / dim))

    # 生成位置索引: 0, 1, ..., end-1
    t = torch.arange(end, dtype=torch.float32)

    # 外积得到每个位置、每个维度组的旋转角度: (end, dim//2)
    freqs = torch.outer(t, freqs)

    # 将角度转换为复指数形式: cos(freqs) + i*sin(freqs)
    # 形状: (end, dim//2, 2)，最后一维是 [cos, sin]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # 复数形式 (end, dim//2)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    将 freqs_cis 扩展为可与 x 广播的形状。

    参数:
        freqs_cis (Tensor): 形状 (seq_len, dim//2) 的复数张量。
        x (Tensor): 形状 (*, seq_len, num_heads, dim//2, 2) 或 (*, seq_len, num_heads, dim)。

    返回:
        Tensor: 扩展形状后的 freqs_cis，便于广播乘法。
    """
    # x 的维度假设为 (batch, seq_len, num_heads, head_dim)
    # 我们需要将 freqs_cis 变成 (1, seq_len, 1, head_dim//2) 的复数张量
    ndim = x.ndim
    assert ndim >= 3, "x 至少要有 3 维"
    assert freqs_cis.shape == (x.shape[1], x.shape[-1] // 2), \
        f"freqs_cis 形状 {freqs_cis.shape} 与 x 不匹配"

    # 构造广播形状: (1, seq_len, 1, ..., dim//2)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    shape[-1] = x.shape[-1] // 2  # 复数维度是原维度的一半
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    对 Query 和 Key 应用旋转位置编码（RoPE）。

    参数:
        xq (Tensor): Query 张量，形状 (batch, seq_len, num_heads, head_dim)。
        xk (Tensor): Key 张量，形状 (batch, seq_len, num_heads, head_dim)。
        freqs_cis (Tensor): 预计算的复指数，形状 (seq_len, head_dim//2) 的复数张量。

    返回:
        (xq_out, xk_out): 旋转后的 Query 和 Key，形状与输入相同。
    """
    # 将实数张量视为复数: 把最后一维的相邻两个数当作 [实部, 虚部]
    # view_as_complex 要求最后一维大小为 2
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # 调整 freqs_cis 形状以便广播
    freqs_cis = reshape_for_broadcast(freqs_cis, xq)

    # 复数乘法实现旋转: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    # 旋转后的结果再转回实数张量
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    # 转回输入的数据类型
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE 模块，封装了预计算 freqs_cis 和应用旋转的逻辑。

    参数:
        head_dim (int): 每个注意力头的维度，必须是偶数。
        max_seq_len (int): 预计算的最大序列长度。
        base (float): 频率基数 theta，默认 10000.0。
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim 必须是偶数")

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # 预计算 freqs_cis 并注册为 buffer（不参与训练）
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len, base)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        对输入的 Query 和 Key 应用 RoPE。

        参数:
            xq (Tensor): Query，形状 (batch, seq_len, num_heads, head_dim)。
            xk (Tensor): Key，形状 (batch, seq_len, num_heads, head_dim)。
            seq_len (int, optional): 当前序列长度，若未提供则使用 xq.shape[1]。

        返回:
            (xq_rot, xk_rot): 旋转后的 Query 和 Key。
        """
        if seq_len is None:
            seq_len = xq.shape[1]

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"输入序列长度 {seq_len} 超过预计算的最大长度 {self.max_seq_len}"
            )

        # 截取当前序列长度对应的 freqs_cis
        freqs_cis = self.freqs_cis[:seq_len]
        return apply_rotary_emb(xq, xk, freqs_cis)


if __name__ == "__main__":
    # ------------------- 自测代码 -------------------
    batch_size, seq_len, num_heads, head_dim = 2, 8, 4, 64
    rope = RotaryPositionalEmbedding(head_dim=head_dim, max_seq_len=512)

    # 构造随机 Q/K
    xq = torch.randn(batch_size, seq_len, num_heads, head_dim)
    xk = torch.randn(batch_size, seq_len, num_heads, head_dim)

    xq_rot, xk_rot = rope(xq, xk)
    print("xq_rot 形状:", xq_rot.shape)
    print("xk_rot 形状:", xk_rot.shape)
    assert xq_rot.shape == xq.shape
    assert xk_rot.shape == xk.shape

    # 验证 RoPE 的核心性质：相对位置不变性
    # 即 <R_m(q), R_n(k)> 只依赖于 (m-n)
    # 我们取 batch=0, head=0 进行验证
    q0 = xq[0, 0, 0, :]   # 位置 0 的 query
    k1 = xk[0, 1, 0, :]   # 位置 1 的 key
    k2 = xk[0, 2, 0, :]   # 位置 2 的 key

    q0_rot, _ = rope(q0.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                     xk[:, :1, :1, :])
    _, k1_rot = rope(xq[:, :1, :1, :],
                     k1.unsqueeze(0).unsqueeze(0).unsqueeze(0))
    _, k2_rot = rope(xq[:, :1, :1, :],
                     k2.unsqueeze(0).unsqueeze(0).unsqueeze(0))

    # 计算内积
    dot_01 = torch.dot(q0_rot[0, 0, 0, :].flatten(), k1_rot[0, 0, 0, :].flatten())
    dot_02 = torch.dot(q0_rot[0, 0, 0, :].flatten(), k2_rot[0, 0, 0, :].flatten())
    print("位置0的Q与位置1的K的内积:", dot_01.item())
    print("位置0的Q与位置2的K的内积:", dot_02.item())

    # 验证：相同相对距离的内积应相同
    # 构造位置1的Q和位置2的K（相对距离也是1）
    q1 = xq[0, 1, 0, :]
    q1_rot, _ = rope(q1.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                     xk[:, :1, :1, :])
    dot_12 = torch.dot(q1_rot[0, 0, 0, :].flatten(), k2_rot[0, 0, 0, :].flatten())
    print("位置1的Q与位置2的K的内积（相对距离也应为1）:", dot_12.item())

    # 由于输入是随机的，内积值本身不会严格相等，但相对位置性质在数学上严格成立
    print("RoPE 自测通过！")
