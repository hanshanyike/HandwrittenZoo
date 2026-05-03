"""
Sinusoidal Positional Encoding (正弦位置编码)

Algorithm: Sinusoidal Absolute Positional Encoding
One-line: 使用不同频率的正弦/余弦函数为序列中每个位置生成唯一的位置向量，
         并直接加到词嵌入上，使 Transformer 获得顺序感知能力。

Core idea:
    Transformer 的自注意力机制本身对输入顺序不敏感（置换等变性）。
    为了让模型区分 "I eat fish" 和 "fish eat I"，必须在输入中显式注入位置信息。
    正弦编码利用不同波长（从 2π 到 10000·2π）的 sinusoid，使得每个位置 pos 的编码
    在 d_model 维空间中具有唯一"指纹"，同时满足：
    1) 唯一性：不同位置的编码向量不同；
    2) 相对位置线性：PE_{pos+k} 可表示为 PE_pos 的线性函数，便于模型学习相对位置；
    3) 外推性：波长覆盖多个数量级，对训练时未见过的更长序列有一定泛化能力。

Complexity:
    - 时间复杂度：O(max_len * d_model) 预计算，前向传播 O(seq_len * d_model)
    - 空间复杂度：O(max_len * d_model) 缓存

Interview frequency: High (Transformer 基础必考点)
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    标准正弦绝对位置编码（来自 Attention Is All You Need）。

    参数:
        d_model (int): 词嵌入维度，必须是偶数。
        max_len (int): 预计算的最大序列长度，默认 5000。
        dropout (float): 在位置编码与词嵌入相加后应用的 dropout 概率。
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model 必须是偶数，因为正弦/余弦成对出现")

        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        # ------------------------------------------------------------------
        # 预计算位置编码矩阵 pe，形状为 (max_len, d_model)
        # 使用 register_buffer 使其不参与梯度更新，且随模型保存/加载
        # ------------------------------------------------------------------
        pe = torch.zeros(max_len, d_model)

        # position: [max_len, 1]，表示每个位置的索引 0, 1, 2, ..., max_len-1
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        # div_term: [d_model/2]，表示每个维度组的角频率分母
        # 公式：10000^(2i/d_model) = exp(2i * -ln(10000) / d_model)
        # 这里 i 取 0, 1, ..., d_model/2 - 1
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        # 偶数维度 (2i) 使用正弦，奇数维度 (2i+1) 使用余弦
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数列

        # 增加 batch 维度，变为 (1, max_len, d_model)，便于广播到 (batch, seq, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将位置编码加到输入嵌入上。

        参数:
            x (Tensor): 输入张量，形状 (batch_size, seq_len, d_model)。

        返回:
            Tensor: 加上位置编码并经过 dropout 后的张量，形状与输入相同。
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"输入序列长度 {seq_len} 超过预计算的最大长度 {self.max_len}"
            )

        # 直接截取前 seq_len 个位置编码，利用广播机制加到 x 上
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


if __name__ == "__main__":
    # ------------------- 自测代码 -------------------
    batch_size, seq_len, d_model = 2, 10, 64
    pe_layer = SinusoidalPositionalEncoding(d_model=d_model, max_len=512)

    # 构造随机输入
    x = torch.randn(batch_size, seq_len, d_model)
    out = pe_layer(x)

    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
    assert out.shape == x.shape, "输出形状应与输入一致"

    # 验证位置编码的唯一性：不同位置的编码向量应不同
    pe_matrix = pe_layer.pe.squeeze(0)  # (max_len, d_model)
    assert not torch.allclose(pe_matrix[0], pe_matrix[1], atol=1e-6), "位置0与位置1的编码应不同"

    # 验证相对位置线性性质：PE_{pos+k} 可由 PE_pos 线性表示（通过三角恒等式）
    # 这里仅做数值验证：相邻位置编码的差异应具有规律性
    diff = pe_matrix[1] - pe_matrix[0]
    diff2 = pe_matrix[2] - pe_matrix[1]
    # 由于 sin/cos 的非线性，差值并不严格相等，但应呈现平滑变化
    print("相邻位置编码差值的 L2 范数:", diff.norm().item())

    # 验证 dropout 是否生效（训练模式下输出应包含零）
    pe_layer.train()
    out_train = pe_layer(x)
    zero_ratio = (out_train == 0).float().mean().item()
    print(f"训练模式下输出中零元素比例（应接近 dropout={pe_layer.dropout.p}）: {zero_ratio:.4f}")

    print("SinusoidalPositionalEncoding 自测通过！")
