"""
Grouped-Query Attention (GQA)

Grouped-Query Attention is a generalization of MQA and MHA, first popularized by
LLaMA-2 and Mistral. Query heads are divided into groups, and each group shares
a single key head and a single value head.

Core idea:
- MHA has 1 K/V head per Q head (too much KV cache).
- MQA has 1 K/V head for all Q heads (may hurt quality).
- GQA uses an intermediate number of K/V head groups, trading off cache size and quality.

Time complexity: O(batch * seq_len^2 * d_model)
Space complexity: O(batch * seq_len * num_kv_heads * head_dim) for KV cache

Interview frequency: High
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力（GQA）模块。
    将 num_heads 个 query heads 分成 num_kv_heads 组，每组共享一组 K/V。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        assert num_heads % num_kv_heads == 0, "num_heads 必须能被 num_kv_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads          # 查询头总数
        self.num_kv_heads = num_kv_heads    # Key/Value 的组数（头数）
        self.head_dim = d_model // num_heads
        # 每个 K/V 头需要服务多少个 Q 头
        self.num_queries_per_kv = num_heads // num_kv_heads

        # Query 投影到完整的多头维度
        self.w_q = nn.Linear(d_model, d_model)

        # Key / Value 投影到 num_kv_heads 个头，每个头 head_dim 维
        self.w_k = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.head_dim)

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

        # 2) Key / Value 投影并分头: (batch, num_kv_heads, seq_len, head_dim)
        k = self.w_k(key)
        k = k.view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(value)
        v = v.view(batch_size, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 3) 为广播做准备：将 K/V 的每个头复制 num_queries_per_kv 次
        #    使得 K/V 的 head 维度与 Q 对齐: (batch, num_heads, seq_len, head_dim)
        #    使用 repeat_interleave 比显式 expand 更直观
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # 4) 计算缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        # 5) 拼接多头并投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        output = self.w_o(attn_output)
        output = self.dropout(output)
        return output


if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 10, 64
    num_heads, num_kv_heads = 8, 2
    gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)

    x = torch.randn(batch_size, seq_len, d_model)
    out = gqa(x, x, x)
    assert out.shape == (batch_size, seq_len, d_model)
    print("GroupedQueryAttention 自测通过，输出形状:", out.shape)
