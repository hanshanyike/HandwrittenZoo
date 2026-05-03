"""
Cross-Attention (Standalone)

Cross-Attention allows one sequence (the decoder) to attend to another sequence
(the encoder output). It is a core component of the original Transformer encoder-decoder
architecture, used in models like BART, T5, and the original "Attention Is All You Need".

Core idea:
- Query comes from the target sequence (decoder hidden states).
- Key and Value come from the source sequence (encoder output).
- This lets the decoder "look up" relevant information from the encoder representation
  when generating each target token.

Time complexity: O(batch * tgt_len * src_len * d_model)
Space complexity: O(batch * num_heads * tgt_len * src_len) for attention scores

Interview frequency: High
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    独立的交叉注意力模块。
    Query 来自目标序列（如 Decoder），Key 和 Value 来自源序列（如 Encoder 输出）。
    """

    def __init__(self, d_model: int, num_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Query 投影: 目标序列 -> d_model
        self.w_q = nn.Linear(d_model, d_model)
        # Key / Value 投影: 源序列 -> d_model
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 输出投影
        self.w_o = nn.Linear(d_model, d_model)
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
            query: (batch, tgt_len, d_model)   — 目标序列（如 Decoder 隐藏状态）
            key:   (batch, src_len, d_model)   — 源序列（如 Encoder 输出）
            value: (batch, src_len, d_model)   — 源序列（通常与 key 相同来源）
            mask:  可选，(batch, 1, tgt_len, src_len) 或 (batch, tgt_len, src_len)
        返回:
            output: (batch, tgt_len, d_model)
        """
        batch_size = query.size(0)
        tgt_len = query.size(1)
        src_len = key.size(1)

        # 1) 投影: Q 来自 query，K/V 来自 key/value
        q = self.w_q(query)
        k = self.w_k(key)
        v = self.w_v(value)

        # 2) 分头处理
        if self.num_heads > 1:
            q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            q = q.unsqueeze(1)   # (batch, 1, tgt_len, head_dim)
            k = k.unsqueeze(1)   # (batch, 1, src_len, head_dim)
            v = v.unsqueeze(1)   # (batch, 1, src_len, head_dim)

        # 3) 缩放点积注意力
        #    scores: (batch, heads, tgt_len, src_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        #    attn_output: (batch, heads, tgt_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # 4) 拼接 head 并输出投影
        if self.num_heads > 1:
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, tgt_len, self.d_model)
        else:
            attn_output = attn_output.squeeze(1)

        output = self.w_o(attn_output)
        output = self.dropout(output)
        return output


if __name__ == "__main__":
    batch_size, tgt_len, src_len, d_model = 2, 8, 12, 64

    # 单头交叉注意力
    ca = CrossAttention(d_model, num_heads=1)
    q = torch.randn(batch_size, tgt_len, d_model)
    kv = torch.randn(batch_size, src_len, d_model)
    out = ca(q, kv, kv)
    assert out.shape == (batch_size, tgt_len, d_model)
    print("CrossAttention (单头) 自测通过，输出形状:", out.shape)

    # 多头交叉注意力
    ca_mh = CrossAttention(d_model, num_heads=8)
    out_mh = ca_mh(q, kv, kv)
    assert out_mh.shape == (batch_size, tgt_len, d_model)
    print("CrossAttention (多头) 自测通过，输出形状:", out_mh.shape)
