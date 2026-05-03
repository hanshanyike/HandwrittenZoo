"""
Multi-Head Latent Attention (MLA)

MLA is the core attention innovation in DeepSeek-V2/V3, designed to drastically
reduce KV cache size during inference while maintaining or even surpassing MHA quality.
It uses low-rank joint compression for keys and values, plus decoupled RoPE.

Core idea:
- Compress hidden states into a low-dimensional latent vector c_t^KV for K/V jointly.
- Up-project c_t^KV back to full K/V dimensions only when needed.
- Apply decoupled RoPE: a small separate K^R with RoPE to carry position info,
  while the compressed K^C remains position-agnostic and can be absorbed into W^Q/W^O.

Time complexity: O(batch * seq_len^2 * d_model)
Space complexity: O(batch * seq_len * (kv_lora_rank + qk_rope_head_dim)) for KV cache

Interview frequency: Very High (DeepSeek hotspot)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLatentAttention(nn.Module):
    """
    多头潜在注意力（MLA）模块，参考 DeepSeek-V2 设计。
    为简化教学实现，此处省略了完整的矩阵吸收（absorption）优化，
    但保留了低秩压缩与解耦 RoPE 的核心结构。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_lora_rank: int | None = None,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        v_head_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # 若未指定 v_head_dim，默认与 head_dim 相同（DeepSeek 中通常也如此）
        self.v_head_dim = v_head_dim if v_head_dim is not None else self.head_dim

        # Query 低秩压缩维度，训练时省激活显存；推理时可吸收
        self.q_lora_rank = q_lora_rank if q_lora_rank is not None else d_model // 2
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim

        # --- Query 低秩压缩路径 ---
        # 下投影: d_model -> q_lora_rank
        self.w_dq = nn.Linear(d_model, self.q_lora_rank, bias=False)
        # 上投影: q_lora_rank -> num_heads * head_dim (得到 q^C)
        self.w_uq = nn.Linear(self.q_lora_rank, num_heads * self.head_dim, bias=False)

        # 解耦 RoPE 的 Query 投影: q_lora_rank -> num_heads * qk_rope_head_dim
        self.w_qr = nn.Linear(self.q_lora_rank, num_heads * qk_rope_head_dim, bias=False)

        # --- KV 低秩联合压缩路径 ---
        # 下投影: d_model -> kv_lora_rank (联合压缩向量 c_t^KV)
        self.w_dkv = nn.Linear(d_model, kv_lora_rank, bias=False)
        # K 上投影: kv_lora_rank -> num_heads * head_dim (得到 k^C)
        self.w_uk = nn.Linear(kv_lora_rank, num_heads * self.head_dim, bias=False)
        # V 上投影: kv_lora_rank -> num_heads * v_head_dim (得到 v^C)
        self.w_uv = nn.Linear(kv_lora_rank, num_heads * self.v_head_dim, bias=False)

        # 解耦 RoPE 的 Key 投影: d_model -> qk_rope_head_dim (共享单头 k^R)
        self.w_kr = nn.Linear(d_model, qk_rope_head_dim, bias=False)

        # 输出投影: num_heads * v_head_dim -> d_model
        self.w_o = nn.Linear(num_heads * self.v_head_dim, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        简化的 RoPE（旋转位置编码）实现。
        实际 DeepSeek 使用更精细的 RoPE 配置（如 YaRN 扩展），
        此处仅演示核心旋转思想。
        参数:
            x: (batch, num_heads, seq_len, head_dim)
        返回:
            roped_x: 同形状
        """
        # 取最后一维作为旋转维度（此处简化为对 qk_rope_head_dim 做旋转）
        d = x.size(-1)
        # 构造频率向量 theta_i = 10000^(-2i/d)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d, 2, device=x.device).float() / d))
        # 位置索引
        pos = torch.arange(seq_len, device=x.device).float()
        # 外积得到角度: (seq_len, d/2)
        angles = torch.einsum("i,j->ij", pos, inv_freq)
        # 构造旋转矩阵的 sin/cos: (seq_len, d/2)
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        # 将 x 拆分为相邻两维一组进行旋转
        x1 = x[..., 0::2]  # 偶数索引
        x2 = x[..., 1::2]  # 奇数索引

        # 广播 sin/cos 到 (batch, heads, seq_len, d/2)
        sin = sin.unsqueeze(0).unsqueeze(0)
        cos = cos.unsqueeze(0).unsqueeze(0)

        # 应用旋转: [x1, x2] * [cos, -sin; sin, cos]
        roped_x1 = x1 * cos - x2 * sin
        roped_x2 = x1 * sin + x2 * cos

        # 交错合并回原始维度
        roped = torch.stack([roped_x1, roped_x2], dim=-1).flatten(-2)
        return roped

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        参数:
            hidden_states: (batch, seq_len, d_model)
            mask: 可选，(batch, 1, seq_len, seq_len)
        返回:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # ===== 1. Query 低秩压缩 =====
        c_q = self.w_dq(hidden_states)                     # (batch, seq, q_lora_rank)
        q_c = self.w_uq(c_q)                               # (batch, seq, num_heads * head_dim)
        q_r = self.w_qr(c_q)                               # (batch, seq, num_heads * qk_rope_head_dim)

        # 分头并转置
        q_c = q_c.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q_r = q_r.view(batch_size, seq_len, self.num_heads, self.qk_rope_head_dim).transpose(1, 2)

        # 对解耦的 q_r 应用 RoPE
        q_r = self._apply_rope(q_r, seq_len)

        # ===== 2. KV 低秩联合压缩 =====
        c_kv = self.w_dkv(hidden_states)                   # (batch, seq, kv_lora_rank)
        k_c = self.w_uk(c_kv)                              # (batch, seq, num_heads * head_dim)
        v_c = self.w_uv(c_kv)                              # (batch, seq, num_heads * v_head_dim)

        k_c = k_c.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_c = v_c.view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2)

        # ===== 3. 解耦 RoPE 的 Key =====
        k_r = self.w_kr(hidden_states)                     # (batch, seq, qk_rope_head_dim)
        # 扩展为多头共享（DeepSeek 中 k^R 是单头的，这里广播到所有头）
        k_r = k_r.unsqueeze(1)                             # (batch, 1, seq, qk_rope_head_dim)
        k_r = k_r.expand(batch_size, self.num_heads, seq_len, self.qk_rope_head_dim)
        k_r = self._apply_rope(k_r, seq_len)

        # ===== 4. 拼接解耦部分 =====
        # q = [q_c; q_r] -> (batch, heads, seq, head_dim + qk_rope_head_dim)
        q = torch.cat([q_c, q_r], dim=-1)
        # k = [k_c; k_r] -> (batch, heads, seq, head_dim + qk_rope_head_dim)
        k = torch.cat([k_c, k_r], dim=-1)

        # ===== 5. 缩放点积注意力 =====
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 注意: V 只来自 v_c，没有 v_r（RoPE 只影响 Q/K 的分数计算）
        attn_output = torch.matmul(attn_weights, v_c)      # (batch, heads, seq, v_head_dim)

        # ===== 6. 拼接并输出投影 =====
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        output = self.w_o(attn_output)
        output = self.dropout(output)
        return output


if __name__ == "__main__":
    batch_size, seq_len, d_model, num_heads = 2, 10, 512, 8
    mla = MultiHeadLatentAttention(
        d_model=d_model,
        num_heads=num_heads,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
    )

    x = torch.randn(batch_size, seq_len, d_model)
    out = mla(x)
    assert out.shape == (batch_size, seq_len, d_model)
    print("MultiHeadLatentAttention 自测通过，输出形状:", out.shape)
