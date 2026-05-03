"""
Simplified FlashAttention (Tiling + Online Softmax)

FlashAttention is an IO-aware exact attention algorithm that computes attention
without materializing the full N x N attention score matrix. It uses two key ideas:
1. Tiling: Process attention in small blocks that fit in fast SRAM.
2. Online Softmax: Incrementally update softmax statistics without storing full rows.

This implementation is a simplified, pure-Python/PyTorch educational version.
Production implementations use fused CUDA kernels (e.g., FlashAttention-2/3).

Core idea:
- Split Q into blocks (tiles) along the sequence dimension.
- Stream K and V blocks one at a time.
- For each Q block, maintain running max (m) and running sum (l) of exponentials.
- Update the output accumulator block by block without storing the full attention matrix.

Time complexity: O(batch * seq_len^2 * d_model)  (same FLOPs as standard attention)
Space complexity: O(batch * seq_len * d_model)   (linear in seq_len, no NxN matrix!)

Interview frequency: Very High
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedFlashAttention(nn.Module):
    """
    简化版 FlashAttention，用于教学演示。
    通过分块（tiling）和在线 softmax 避免显式存储完整的注意力分数矩阵。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        block_size: int = 64,
        causal: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.causal = causal

        # 标准 MHA 的投影层，FlashAttention 只改变计算方式，不改变结构
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _flash_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        单头 FlashAttention 前向实现（在线 softmax + 分块）。
        参数:
            q, k, v: 均为 (batch, num_heads, seq_len, head_dim)
        返回:
            output: (batch, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        block_size = self.block_size

        # 初始化输出矩阵 O，以及在线 softmax 的运行统计量
        # m: 当前已见分数的最大值 (batch, heads, seq_len)
        # l: 当前已见 exp(score - m) 的累加和 (batch, heads, seq_len)
        o = torch.zeros_like(q)
        m = torch.full((batch_size, num_heads, seq_len), float("-inf"), device=q.device, dtype=q.dtype)
        l = torch.zeros((batch_size, num_heads, seq_len), device=q.device, dtype=q.dtype)

        # 将 K、V 按 block_size 分块遍历
        for kv_start in range(0, seq_len, block_size):
            kv_end = min(kv_start + block_size, seq_len)
            k_block = k[:, :, kv_start:kv_end, :]   # (batch, heads, block, head_dim)
            v_block = v[:, :, kv_start:kv_end, :]   # (batch, heads, block, head_dim)

            # 计算当前 Q 与 K 块的分数: (batch, heads, seq_len, block)
            scores = torch.matmul(q, k_block.transpose(-2, -1)) / math.sqrt(head_dim)

            # 若启用因果掩码，将未来位置设为 -inf
            if self.causal:
                # 构造列掩码: 当前 K 块的位置索引
                col_idx = torch.arange(kv_start, kv_end, device=q.device)
                # 行掩码: Q 的位置索引
                row_idx = torch.arange(seq_len, device=q.device)
                # mask: (seq_len, block)，True 表示需要保留（row >= col）
                causal_mask = row_idx.unsqueeze(1) >= col_idx.unsqueeze(0)
                scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            # ---- 在线 softmax 更新 ----
            # 1. 计算当前块每行的最大值
            block_max = torch.max(scores, dim=-1).values  # (batch, heads, seq_len)

            # 2. 更新全局最大值 m_new = max(m_old, block_max)
            m_new = torch.maximum(m, block_max)

            # 3. 计算修正因子 exp(m_old - m_new)，用于缩放旧的 l 和 o
            exp_diff = torch.exp(m - m_new)  # (batch, heads, seq_len)

            # 4. 缩放旧的累加和 l，并加上当前块的贡献
            #    当前块需要先减去新的全局最大值，再取 exp
            exp_scores = torch.exp(scores - m_new.unsqueeze(-1))  # (batch, heads, seq_len, block)
            l_new = l * exp_diff + torch.sum(exp_scores, dim=-1)   # (batch, heads, seq_len)

            # 5. 更新输出累加器 o
            #    旧的 o 需要按 exp_diff 缩放，再加上当前块的加权 v_block
            o = o * exp_diff.unsqueeze(-1) + torch.matmul(exp_scores, v_block)

            # 6. 更新运行统计量
            m, l = m_new, l_new

        # 最终归一化: o / l
        output = o / l.unsqueeze(-1)
        return output

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
            mask:  可选，当前简化版主要依赖 causal 参数，额外 mask 暂未融合
        返回:
            output: (batch, seq_len, d_model)
        """
        batch_size = query.size(0)

        # 1) 线性投影并分头
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # 2) FlashAttention 分块计算
        attn_output = self._flash_attention_forward(q, k, v)

        # 3) 拼接多头并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        output = self.w_o(attn_output)
        output = self.dropout(output)
        return output


if __name__ == "__main__":
    batch_size, seq_len, d_model, num_heads = 2, 128, 64, 8
    flash_attn = SimplifiedFlashAttention(d_model, num_heads, block_size=32, causal=True)

    x = torch.randn(batch_size, seq_len, d_model)
    out = flash_attn(x, x, x)
    assert out.shape == (batch_size, seq_len, d_model)
    print("SimplifiedFlashAttention 自测通过，输出形状:", out.shape)

    # 数值正确性验证：与标准注意力对比（非因果）
    flash_attn_eval = SimplifiedFlashAttention(d_model, num_heads, block_size=32, causal=False)
    flash_attn_eval.eval()
    with torch.no_grad():
        flash_out = flash_attn_eval(x, x, x)

    # 构造标准 MHA 参考输出
    class StdMHA(nn.Module):
        def __init__(self, d, h):
            super().__init__()
            self.w_q = nn.Linear(d, d)
            self.w_k = nn.Linear(d, d)
            self.w_v = nn.Linear(d, d)
            self.w_o = nn.Linear(d, d)

        def forward(self, q, k, v):
            b = q.size(0)
            h = self.w_q(q).view(b, -1, num_heads, d_model // num_heads).transpose(1, 2)
            kk = self.w_k(k).view(b, -1, num_heads, d_model // num_heads).transpose(1, 2)
            vv = self.w_v(v).view(b, -1, num_heads, d_model // num_heads).transpose(1, 2)
            scores = torch.matmul(h, kk.transpose(-2, -1)) / math.sqrt(d_model // num_heads)
            attn = torch.matmul(torch.softmax(scores, dim=-1), vv)
            attn = attn.transpose(1, 2).contiguous().view(b, -1, d_model)
            return self.w_o(attn)

    std_mha = StdMHA(d_model, num_heads)
    # 复制权重以确保一致性
    std_mha.w_q.weight = flash_attn_eval.w_q.weight
    std_mha.w_k.weight = flash_attn_eval.w_k.weight
    std_mha.w_v.weight = flash_attn_eval.w_v.weight
    std_mha.w_o.weight = flash_attn_eval.w_o.weight
    std_mha.eval()
    with torch.no_grad():
        std_out = std_mha(x, x, x)

    diff = torch.max(torch.abs(flash_out - std_out)).item()
    print(f"与标准注意力的最大绝对误差: {diff:.6f} (应接近 0)")
