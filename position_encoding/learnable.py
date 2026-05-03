"""
Learnable Positional Embedding (可学习位置嵌入)

Algorithm: Learnable Positional Embedding
One-line: 将每个位置的位置向量作为可训练参数，与词嵌入相加，
         让模型通过数据驱动的方式自主学习最优的位置表示。

Core idea:
    正弦编码是手工设计的确定性函数，虽然具有相对位置线性性质，
    但可能不是最优的位置表示。可学习位置嵌入的核心思想是：
    将位置信息也参数化，让模型通过反向传播和大量数据自动学习
    最适合当前任务的位置表示。
    这种方法更灵活，但增加了可训练参数，且对训练时未见过的
    更长序列缺乏外推能力（超出 max_len 的位置没有对应参数）。
    BERT、GPT-2、ViT 等模型采用此方案。

Complexity:
    - 时间复杂度：O(seq_len * d_model)，前向传播仅为查表和相加
    - 空间复杂度：O(max_len * d_model) 参数
    - 参数量：max_len * d_model

Interview frequency: Medium (与正弦编码对比常考)
"""

import torch
import torch.nn as nn


class LearnablePositionalEncoding(nn.Module):
    """
    可学习位置编码，每个位置对应一个可训练的 d_model 维向量。

    参数:
        d_model (int): 词嵌入维度。
        max_len (int): 最大序列长度（决定位置嵌入参数的数量）。
        dropout (float): 在位置编码与词嵌入相加后应用的 dropout 概率。
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        # ------------------------------------------------------------------
        # 可学习的位置嵌入矩阵，形状为 (max_len, d_model)
        # 使用 nn.Embedding 实现，每个位置索引对应一个 d_model 维向量
        # ------------------------------------------------------------------
        self.pos_embedding = nn.Embedding(max_len, d_model)

        # 初始化：通常使用较小的随机值或正弦编码初始化
        # 这里使用标准正态分布初始化，标准差与 d_model 相关
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将可学习位置编码加到输入嵌入上。

        参数:
            x (Tensor): 输入张量，形状 (batch_size, seq_len, d_model)。

        返回:
            Tensor: 加上位置编码并经过 dropout 后的张量，形状与输入相同。
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"输入序列长度 {seq_len} 超过最大可学习长度 {self.max_len}"
            )

        # 生成位置索引: 0, 1, ..., seq_len-1
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        # 扩展为 (batch_size, seq_len)，利用广播机制
        positions = positions.expand(x.size(0), -1)

        # 查表获取位置嵌入: (batch_size, seq_len, d_model)
        pos_emb = self.pos_embedding(positions)

        # 位置编码与词嵌入相加
        x = x + pos_emb
        return self.dropout(x)


class LearnablePositionalEncodingParam(nn.Module):
    """
    另一种实现方式：直接使用 nn.Parameter 定义位置编码参数。

    与 nn.Embedding 实现等价，但显式暴露参数，便于某些特殊初始化策略。

    参数:
        d_model (int): 词嵌入维度。
        max_len (int): 最大序列长度。
        dropout (float): Dropout 概率。
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        # 直接使用 Parameter，形状 (1, max_len, d_model)，便于广播
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"输入序列长度 {seq_len} 超过最大可学习长度 {self.max_len}"
            )

        # 直接截取前 seq_len 个位置参数，广播相加
        x = x + self.pos_emb[:, :seq_len, :]
        return self.dropout(x)


if __name__ == "__main__":
    # ------------------- 自测代码 -------------------
    batch_size, seq_len, d_model = 2, 10, 64

    # 测试 nn.Embedding 实现
    pe_layer = LearnablePositionalEncoding(d_model=d_model, max_len=512)
    x = torch.randn(batch_size, seq_len, d_model)
    out = pe_layer(x)
    print("[Embedding 实现] 输入形状:", x.shape)
    print("[Embedding 实现] 输出形状:", out.shape)
    assert out.shape == x.shape

    # 测试 Parameter 实现
    pe_layer_param = LearnablePositionalEncodingParam(d_model=d_model, max_len=512)
    out_param = pe_layer_param(x)
    print("[Parameter 实现] 输出形状:", out_param.shape)
    assert out_param.shape == x.shape

    # 验证参数是否可学习：检查梯度
    loss = out.sum()
    loss.backward()
    assert pe_layer.pos_embedding.weight.grad is not None, "位置嵌入参数应有梯度"
    print("梯度检查通过：位置嵌入参数参与了反向传播")

    # 验证不同位置的位置编码不同
    pos_indices = torch.tensor([[0, 1, 2]])  # (1, 3)
    emb_0 = pe_layer.pos_embedding(pos_indices[:, 0])
    emb_1 = pe_layer.pos_embedding(pos_indices[:, 1])
    emb_2 = pe_layer.pos_embedding(pos_indices[:, 2])
    assert not torch.allclose(emb_0, emb_1, atol=1e-6), "位置0与位置1的嵌入应不同"
    assert not torch.allclose(emb_1, emb_2, atol=1e-6), "位置1与位置2的嵌入应不同"
    print("位置唯一性验证通过")

    # 验证超出 max_len 时抛出异常
    try:
        x_long = torch.randn(batch_size, 600, d_model)
        pe_layer(x_long)
        assert False, "应抛出长度超限异常"
    except ValueError as e:
        print(f"长度超限异常捕获成功: {e}")

    # 验证 dropout 是否生效（训练模式下）
    pe_layer.train()
    out_train = pe_layer(x)
    zero_ratio = (out_train == 0).float().mean().item()
    print(f"训练模式下零元素比例（应接近 dropout={pe_layer.dropout.p}）: {zero_ratio:.4f}")

    print("LearnablePositionalEncoding 自测通过！")
