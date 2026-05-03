"""
Self-Attention (Standalone)

Self-Attention is the foundational building block of the Transformer.
In self-attention, the Query, Key, and Value all come from the same input sequence,
allowing each token to attend to every other token in the sequence.

Core idea:
- For each token, compute its relationship (attention score) with all other tokens.
- Use these scores to build a weighted sum of all token representations.
- This creates a context-aware representation for each position.

Time complexity: O(batch * seq_len^2 * d_model)
Space complexity: O(batch * seq_len^2 * num_heads) for attention scores

Interview frequency: High
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    独立的自注意力模块。
    Q、K、V 均来自同一输入，通过三个独立的线性层投影得到。
    """

    def __init__(self, d_model: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        # 单头自注意力时 num_heads=1；多头时与 MHA 等价
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 单头场景下，这三个投影分别将输入映射到 d_model（即 head_dim）
        # 多头场景下，输出仍为 d_model，后续按头切分
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 输出投影
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        参数:
            x:    (batch, seq_len, d_model)
            mask: 可选，(batch, 1, seq_len, seq_len) 或 (batch, seq_len, seq_len)
        返回:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 1) 投影得到 Q、K、V
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 2) 若为多 head，重塑为 (batch, num_heads, seq_len, head_dim)
        if self.num_heads > 1:
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            # 单头时增加 head 维度便于统一计算: (batch, 1, seq_len, head_dim)
            q = q.unsqueeze(1)
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)

        # 3) 缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # 兼容不同 mask 形状
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        # 4) 拼接 head 并输出投影
        if self.num_heads > 1:
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        else:
            attn_output = attn_output.squeeze(1)

        output = self.w_o(attn_output)
        output = self.dropout(output)
        return output


if __name__ == "__main__":
    # 测试单头自注意力
    batch_size, seq_len, d_model = 2, 10, 64
    sa = SelfAttention(d_model, num_heads=1)
    x = torch.randn(batch_size, seq_len, d_model)
    out = sa(x)
    assert out.shape == (batch_size, seq_len, d_model)
    print("SelfAttention (单头) 自测通过，输出形状:", out.shape)

    # 测试多头自注意力
    sa_mh = SelfAttention(d_model, num_heads=8)
    out_mh = sa_mh(x)
    assert out_mh.shape == (batch_size, seq_len, d_model)
    print("SelfAttention (多头) 自测通过，输出形状:", out_mh.shape)
