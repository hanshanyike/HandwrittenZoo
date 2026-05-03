"""
Multi-Head Attention (MHA)

Standard Multi-Head Attention as introduced in "Attention Is All You Need" (NeurIPS 2017).
Splits the input into multiple attention heads, allowing the model to jointly attend to
information from different representation subspaces at different positions.

Core idea:
- Project input into h heads of Q, K, V separately.
- Compute scaled dot-product attention in parallel for each head.
- Concatenate heads and apply a final linear projection.

Time complexity: O(batch * seq_len^2 * d_model)
Space complexity: O(batch * seq_len^2 * num_heads) for the attention scores matrix.

Interview frequency: High
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    单头缩放点积注意力。
    计算 Q 与 K 的相似度，再用 softmax 归一化后对 V 加权求和。
    """

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        参数:
            query: (batch, num_heads, seq_len, head_dim)
            key:   (batch, num_heads, seq_len, head_dim)
            value: (batch, num_heads, seq_len, head_dim)
            mask:  (batch, 1, seq_len, seq_len) 或兼容形状，True 表示需要 mask 的位置
        返回:
            output: (batch, num_heads, seq_len, head_dim)
        """
        # 取 head_dim 用于缩放，防止点积过大导致 softmax 梯度消失
        d_k = query.size(-1)
        # 计算相似度分数: (batch, heads, seq, seq)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 若提供 mask，则在 softmax 前将对应位置设为极小值
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # softmax 归一化得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和得到输出
        output = torch.matmul(attn_weights, value)
        return output


class MultiHeadAttention(nn.Module):
    """
    标准多头注意力模块。
    将输入投影到 num_heads 个头，并行计算注意力，再拼接并投影回 d_model。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # 确保 d_model 能被 num_heads 整除，以便均分每个头的维度
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 每个头的维度

        # 定义 Q、K、V 的线性投影层，将输入从 d_model 投影到 d_model
        # 虽然这里输出维度仍是 d_model，但后续会按 head 切分
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 最终输出投影，将拼接后的多头结果映射回 d_model
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        参数:
            query: (batch, seq_len, d_model)
            key:   (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask:  可选，(batch, 1, seq_len, seq_len)
        返回:
            output: (batch, seq_len, d_model)
        """
        batch_size = query.size(0)

        # 1) 线性投影: (batch, seq, d_model)
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # 2) 分头: 将 d_model 拆成 num_heads 个 head_dim
        #    目标形状: (batch, num_heads, seq_len, head_dim)
        #    view + transpose 比 reshape 更直观，确保内存布局正确
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) 计算缩放点积注意力
        attn_output = self.attention(q, k, v, mask)

        # 4) 拼接多头: (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)

        # 5) 最终线性投影
        output = self.w_o(attn_output)
        output = self.dropout(output)
        return output


if __name__ == "__main__":
    # 自测：构造随机输入，验证输出形状正确
    batch_size, seq_len, d_model, num_heads = 2, 10, 64, 8
    mha = MultiHeadAttention(d_model, num_heads)

    x = torch.randn(batch_size, seq_len, d_model)
    out = mha(x, x, x)  # 自注意力场景
    assert out.shape == (batch_size, seq_len, d_model)
    print("MultiHeadAttention 自测通过，输出形状:", out.shape)
