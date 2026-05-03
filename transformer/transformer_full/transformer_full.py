"""
Transformer (Complete Encoder-Decoder)
======================================
从零实现完整的 Encoder-Decoder Transformer（Vaswani et al., 2017）。
包含：Token Embedding、Sinusoidal Positional Encoding、Multi-Head Attention、
Position-wise FFN、LayerNorm、Padding Mask / Causal Mask。

核心思想：
    完全基于自注意力机制（Self-Attention）的序列到序列模型，摒弃 RNN/CNN，
    通过多头注意力在不同表示子空间中并行捕获依赖，配合残差连接与层归一化实现深层训练。

时间复杂度：O(n^2 * d)（序列长度 n，模型维度 d，主要来自 Attention 的 QK^T）
空间复杂度：O(n * d)（每层激活值）
面试频率：极高
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Embedding):
    """词嵌入层：将离散的 token 索引映射为稠密向量。"""

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int = 0):
        # padding_idx 保证填充位置不参与梯度更新，避免干扰训练
        super().__init__(vocab_size, d_model, padding_idx=padding_idx)


class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码（Sinusoidal Positional Encoding）。
    优点：可外推到训练时未见过的更长序列；每个位置是唯一的；相对位置可通过线性变换得到。
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # 预计算位置编码矩阵，训练时不更新
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        # 分母项：10000^(2i/d_model)，取对数避免数值溢出
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为 buffer，不作为模型参数，但会随模型保存/加载
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            与 x 同 shape 的位置编码张量
        """
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]


class MultiHeadAttention(nn.Module):
    """
    多头自注意力（Multi-Head Self-Attention）。
    将 Q/K/V 投影到 h 个头，分别计算 Scaled Dot-Product Attention，再拼接并线性投影。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 线性投影层：将输入映射到 Q、K、V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # 输出投影层：将拼接后的多头结果映射回 d_model
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key:   (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            mask:  广播兼容 (batch_size, 1, seq_len_q, seq_len_k) 或 (1, 1, seq_len_q, seq_len_k)
        Returns:
            output: (batch_size, seq_len_q, d_model)
        """
        batch_size = query.size(0)

        # 1) 线性投影并分头
        # (batch, seq, d_model) -> (batch, seq, heads, d_k) -> (batch, heads, seq, d_k)
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2) Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, heads, q_len, k_len)

        # 3) 应用 mask（Padding Mask 或 Causal Mask）
        if mask is not None:
            # 将 mask 为 0 的位置填充极大负值，使 softmax 后接近 0
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)  # 沿 key 维度归一化
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)  # (batch, heads, q_len, d_k)

        # 4) 拼接多头并线性投影
        # (batch, heads, q_len, d_k) -> (batch, q_len, heads, d_k) -> (batch, q_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)
        return output


class PositionwiseFeedForward(nn.Module):
    """
    位置前馈网络（FFN）：对每个位置独立应用两层线性变换 + 激活。
    原始论文使用 ReLU，现代实现常用 GELU。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一层升维 -> 激活 -> Dropout -> 第二层降维
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class LayerNorm(nn.Module):
    """
    层归一化（Layer Normalization）。
    对单个样本的所有特征做归一化，稳定深层网络训练。
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))   # 可学习缩放
        self.beta = nn.Parameter(torch.zeros(d_model))   # 可学习偏移
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        # 归一化后使用 gamma/beta 进行仿射变换
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class EncoderLayer(nn.Module):
    """
    Transformer Encoder 层（Post-Norm 版本）。
    结构：Multi-Head Self-Attention -> 残差 -> LayerNorm -> FFN -> 残差 -> LayerNorm
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # 子层1：多头自注意力 + 残差连接 + LayerNorm
        _x = x
        x = self.self_attn(x, x, x, mask=src_mask)
        x = self.dropout1(x)
        x = self.norm1(_x + x)

        # 子层2：FFN + 残差连接 + LayerNorm
        _x = x
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = self.norm2(_x + x)
        return x


class DecoderLayer(nn.Module):
    """
    Transformer Decoder 层（Post-Norm 版本）。
    结构：
        1) Masked Self-Attention -> 残差 -> LayerNorm
        2) Cross-Attention (Q from decoder, K/V from encoder) -> 残差 -> LayerNorm
        3) FFN -> 残差 -> LayerNorm
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # 子层1：带 Mask 的自注意力（防止看到未来信息）
        _x = x
        x = self.masked_self_attn(x, x, x, mask=tgt_mask)
        x = self.dropout1(x)
        x = self.norm1(_x + x)

        # 子层2：Cross-Attention（Decoder 查询 Encoder 输出）
        _x = x
        x = self.cross_attn(query=x, key=enc_output, value=enc_output, mask=src_mask)
        x = self.dropout2(x)
        x = self.norm2(_x + x)

        # 子层3：FFN
        _x = x
        x = self.feed_forward(x)
        x = self.dropout3(x)
        x = self.norm3(_x + x)
        return x


class TransformerEncoder(nn.Module):
    """由 N 个 EncoderLayer 堆叠而成的编码器。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_emb = TokenEmbedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # Embedding + Positional Encoding
        x = self.token_emb(src) * math.sqrt(self.d_model)  # 缩放以匹配位置编码量级
        x = x + self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class TransformerDecoder(nn.Module):
    """由 N 个 DecoderLayer 堆叠而成的解码器，最后接线性投影到词表。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_emb = TokenEmbedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.token_emb(tgt) * math.sqrt(self.d_model)
        x = x + self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_mask)
        return self.fc_out(x)


class Transformer(nn.Module):
    """
    完整的 Encoder-Decoder Transformer。
    用于机器翻译等 Seq2Seq 任务。
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
        d_model: int = 512,
        max_len: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, max_len, n_layers, n_heads, d_ff, dropout, src_pad_idx
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, max_len, n_layers, n_heads, d_ff, dropout, tgt_pad_idx
        )

        # 参数初始化（Xavier uniform）
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """
        构造 Padding Mask（Encoder 使用）。
        形状: (batch_size, 1, 1, src_seq_len)
        """
        return (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """
        构造 Decoder 的联合 Mask：Padding Mask + Causal Mask。
        形状: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        """
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
        tgt_len = tgt.size(1)
        # 下三角矩阵：当前位置只能看到自己和之前的位置
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
        return tgt_pad_mask & causal_mask  # 广播后逐元素与

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_output, tgt_mask, src_mask)
        return output


# ==================== Self-Test ====================
if __name__ == "__main__":
    # 超参数
    SRC_VOCAB = 1000
    TGT_VOCAB = 1000
    D_MODEL = 512
    N_LAYERS = 6
    N_HEADS = 8
    D_FF = 2048
    BATCH_SIZE = 2
    SRC_LEN = 10
    TGT_LEN = 12

    model = Transformer(
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
    )

    src = torch.randint(1, SRC_VOCAB, (BATCH_SIZE, SRC_LEN))
    tgt = torch.randint(1, TGT_VOCAB, (BATCH_SIZE, TGT_LEN))
    # 模拟填充位置
    src[0, -2:] = 0
    tgt[0, -3:] = 0

    out = model(src, tgt)
    print(f"[Transformer Full] Output shape: {out.shape}")  # (B, TGT_LEN, TGT_VOCAB)
    assert out.shape == (BATCH_SIZE, TGT_LEN, TGT_VOCAB)

    # 验证 Causal Mask：最后一个时间步不应看到未来
    tgt_mask = model.make_tgt_mask(tgt)
    print(f"[Transformer Full] Causal mask upper-tri sum: {torch.triu(tgt_mask[0,0], diagonal=1).sum().item()} (should be 0)")
    print("Self-test passed.")
