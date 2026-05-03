"""
Multi-Query Attention (MQA)

Multi-Query Attention is an efficient attention variant introduced to drastically
reduce memory bandwidth and KV cache size during inference. All query heads share
a single key head and a single value head.

Core idea:
- Keep multiple query heads for expressive power.
- Share one key head and one value head across all query heads.
- This reduces KV cache to 1/h of standard MHA, significantly boosting inference throughput.

Time complexity: O(batch * seq_len^2 * d_model)  (same as MHA during training)
Space complexity: O(batch * seq_len * d_model) for KV cache during inference (much smaller than MHA)

Interview frequency: High
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQueryAttention(nn.Module):
    """
    多查询注意力（MQA）模块。
    所有查询头共享同一组 Key 和 Value 投影，推理时 KV Cache 大幅减少。
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Query 仍按多头投影，每个头有独立的查询向量
        self.w_q = nn.Linear(d_model, d_model)

        # Key 和 Value 只保留单头投影，输出维度为 head_dim
        self.w_k = nn.Linear(d_model, self.head_dim)
        self.w_v = nn.Linear(d_model, self.head_dim)

        # 输出投影将多头拼接后的结果映射回 d_model
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
            query: (batch, seq_len, d_model)
            key:   (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask:  可选，(batch, 1, seq_len, seq_len)
        返回:
            output: (batch, seq_len, d_model)
        """
        batch_size = query.size(0)

        # 1) Query 投影并分头: (batch, num_heads, seq_len, head_dim)
        q = self.w_q(query)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 2) Key / Value 投影为单头: (batch, 1, seq_len, head_dim)
        #    通过 unsqueeze(1) 增加 head 维度，便于广播到所有 query heads
        k = self.w_k(key).unsqueeze(1)   # (batch, 1, seq_len, head_dim)
        v = self.w_v(value).unsqueeze(1) # (batch, 1, seq_len, head_dim)

        # 3) 计算缩放点积注意力
        #    Q: (batch, heads, seq, head_dim) @ K^T: (batch, 1, head_dim, seq)
        #    -> scores: (batch, heads, seq, seq)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 4) 加权求和: (batch, heads, seq, seq) @ (batch, 1, seq, head_dim)
        #    -> (batch, heads, seq, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # 5) 拼接多头并投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        output = self.w_o(attn_output)
        output = self.dropout(output)
        return output


if __name__ == "__main__":
    batch_size, seq_len, d_model, num_heads = 2, 10, 64, 8
    mqa = MultiQueryAttention(d_model, num_heads)

    x = torch.randn(batch_size, seq_len, d_model)
    out = mqa(x, x, x)
    assert out.shape == (batch_size, seq_len, d_model)
    print("MultiQueryAttention 自测通过，输出形状:", out.shape)
