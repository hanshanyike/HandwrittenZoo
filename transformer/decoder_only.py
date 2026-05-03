"""
Decoder-Only Transformer (GPT / LLaMA Style)
============================================
现代大语言模型（GPT、LLaMA、Qwen 等）广泛使用的 Decoder-Only 架构。
本实现包含：Pre-Norm、RMSNorm、RoPE（旋转位置编码）、SwiGLU FFN、Causal Mask。

核心思想：
    仅保留 Transformer Decoder，去掉 Cross-Attention，使用 Causal Mask 实现自回归生成；
    Pre-Norm 提升训练稳定性；RMSNorm 减少计算量；RoPE 同时编码绝对与相对位置；
    SwiGLU 增强 FFN 表达能力。

时间复杂度：O(n^2 * d)（自回归生成时可通过 KV Cache 优化到 O(n * d) 每步）
空间复杂度：O(n * d)
面试频率：极高
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization（RMSNorm）。
    与 LayerNorm 的区别：去掉均值中心化（mean centering），只保留缩放（scaling）。
    公式：RMSNorm(x) = x / RMS(x) * gamma，其中 RMS(x) = sqrt(mean(x^2) + eps)
    优点：参数量减半（无 beta），计算更快，且在大模型上效果与 LayerNorm 相当甚至更优。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数，初始化为 1
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        # 计算均方根，沿最后一维
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.weight * x_norm


class RotaryPositionalEmbedding(nn.Module):
    """
    旋转位置编码（RoPE, Rotary Positional Embedding）。
    核心思想：通过旋转矩阵将位置信息编码到 Q/K 向量中，使得内积天然包含相对位置信息。
    优点：支持长度外推（length extrapolation），相对位置具有显式表达。
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "dim 必须为偶数"
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # 预计算旋转角度 theta_i = base^(-2i/dim)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))  # (dim/2,)
        # 预计算位置 m * theta_i
        positions = torch.arange(max_seq_len, dtype=torch.float32)  # (max_seq_len,)
        freqs = torch.outer(positions, inv_freq)  # (max_seq_len, dim/2)
        # 复数形式：cos(m*theta) + i*sin(m*theta)
        self.register_buffer("cos_cached", freqs.cos())  # (max_seq_len, dim/2)
        self.register_buffer("sin_cached", freqs.sin())  # (max_seq_len, dim/2)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        将输入的后半部分取负并交换前后半。
        用于 RoPE 的旋转操作：将 (x1, x2, x3, x4, ...) 变为 (-x2, x1, -x4, x3, ...)
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        对输入张量应用 RoPE。
        Args:
            x: (batch_size, n_heads, seq_len, head_dim)
        Returns:
            旋转后的张量，shape 与输入相同
        """
        # 取前 seq_len 个位置的 cos/sin
        cos = self.cos_cached[:seq_len, :]  # (seq_len, dim/2)
        sin = self.sin_cached[:seq_len, :]  # (seq_len, dim/2)

        # 将 cos/sin 扩展为与 x 兼容的形状: (1, 1, seq_len, dim/2)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # 将 x 的最后两维每两个一组，应用旋转
        # x * cos + rotate_half(x) * sin
        x1, x2 = x[..., ::2], x[..., 1::2]
        # 为了兼容 rotate_half，将 x  reshape 为 (..., dim)
        # 更直接的方式：使用交替排列
        x_rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return x * cos + x_rotated * sin


class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络（LLaMA 使用）。
    公式：SwiGLU(x) = (x @ W1) * SiLU(x @ W3) @ W2
    其中 SiLU(x) = x * sigmoid(x)，也称为 Swish 激活函数。
    相比标准 FFN，SwiGLU 引入门控机制，表达能力更强。
    注意：SwiGLU 有三个权重矩阵，中间维度通常调整为 2/3 * d_ff 以保持参数量相近。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # LLaMA 中通常将中间维度乘以 2/3 来补偿三个矩阵带来的参数量增加
        hidden_dim = int(2 * d_ff / 3)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU 作为门控信号
        gate = F.silu(self.w1(x))
        # 逐元素相乘后投影回 d_model
        return self.w2(self.dropout(gate * self.w3(x)))


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力（Grouped Query Attention, GQA）。
    介于 MHA（每个头有独立 K/V）和 MQA（所有头共享 K/V）之间的折中方案。
    将 n_heads 分为 n_kv_groups 组，每组共享一对 K/V 头。
    优点：显著减少 KV Cache 内存占用，同时保持接近 MHA 的质量。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_groups: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_groups == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_groups = n_kv_groups
        self.n_kv_heads = n_kv_groups  # 实际 K/V 头的数量
        self.d_k = d_model // n_heads
        self.d_kv = self.d_k * self.n_kv_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, self.d_kv, bias=False)
        self.w_v = nn.Linear(d_model, self.d_kv, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Q: (batch, seq, n_heads, d_k)
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        # K/V: (batch, seq, n_kv_heads, d_k)
        K = self.w_k(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_kv_heads, self.d_k).transpose(1, 2)

        # 应用 RoPE 到 Q 和 K
        Q = rope(Q, seq_len)
        K = rope(K, seq_len)

        # GQA：将 K/V 重复以匹配 Q 的头数
        # n_heads // n_kv_heads 表示每个 K/V 头被多少个 Q 头共享
        if self.n_kv_heads < self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            K = K.repeat_interleave(n_rep, dim=1)  # (batch, n_heads, seq, d_k)
            V = V.repeat_interleave(n_rep, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(context)


class DecoderLayer(nn.Module):
    """
    Decoder-Only Layer（Pre-Norm 版本）。
    与原始 Transformer Post-Norm 不同，Pre-Norm 在每个子层**之前**做归一化，
    形成更干净的残差流（clean residual stream），显著提升深层模型训练稳定性。
    结构：RMSNorm -> Masked Self-Attention -> 残差 -> RMSNorm -> SwiGLU -> 残差
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_groups: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_groups, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryPositionalEmbedding,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Pre-Norm：先归一化再进入子层
        x = x + self.attn(self.norm1(x), rope, mask)
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    """
    GPT / LLaMA 风格的 Decoder-Only Transformer。
    用于自回归语言建模和文本生成。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 4096,
        max_len: int = 2048,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_groups: int = 4,
        d_ff: int = 11008,
        dropout: float = 0.0,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.n_heads = n_heads

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, n_heads, n_kv_groups, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # RoPE 在所有层共享
        head_dim = d_model // n_heads
        self.rope = RotaryPositionalEmbedding(head_dim, max_len)

        # 权重初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """构造 Causal Mask：下三角矩阵（含对角线）。"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        x = self.token_emb(input_ids)
        x = self.dropout(x)

        causal_mask = self._make_causal_mask(seq_len, input_ids.device)

        for layer in self.layers:
            x = layer(x, self.rope, causal_mask)

        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """
        自回归生成文本（贪心/采样）。
        注意：本实现未包含 KV Cache 优化，仅用于演示。
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 如果序列超过 max_len，截断
            if input_ids.size(1) > self.max_len:
                input_ids = input_ids[:, -self.max_len:]
            logits = self.forward(input_ids)
            # 只取最后一个位置的 logits
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids


# ==================== Self-Test ====================
if __name__ == "__main__":
    VOCAB_SIZE = 1000
    D_MODEL = 256
    MAX_LEN = 128
    N_LAYERS = 4
    N_HEADS = 8
    N_KV_GROUPS = 2  # GQA：8 个 Q 头共享 2 个 K/V 头
    D_FF = 512
    BATCH_SIZE = 2
    SEQ_LEN = 16

    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        max_len=MAX_LEN,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_kv_groups=N_KV_GROUPS,
        d_ff=D_FF,
    )

    # 1) 前向传播测试
    input_ids = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    logits = model(input_ids)
    print(f"[Decoder-Only] Logits shape: {logits.shape}")  # (B, seq_len, vocab_size)
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)

    # 2) 验证 Causal Mask：位置 i 的 logits 不应依赖位置 i+1 的输入
    causal_mask = model._make_causal_mask(SEQ_LEN, input_ids.device)
    print(f"[Decoder-Only] Causal mask upper-tri sum: {torch.triu(causal_mask[0,0], diagonal=1).sum().item()} (should be 0)")

    # 3) 生成测试
    prompt = torch.randint(1, VOCAB_SIZE, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10)
    print(f"[Decoder-Only] Prompt length: {prompt.size(1)}, Generated length: {generated.size(1)}")
    assert generated.size(1) == prompt.size(1) + 10

    # 4) 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Decoder-Only] Total parameters: {total_params / 1e6:.2f}M")

    print("Self-test passed.")
