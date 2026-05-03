"""
BERT (Bidirectional Encoder Representations from Transformers)
===============================================================
基于 Transformer Encoder 的双向预训练语言模型（Devlin et al., 2019）。
本实现包含：Token/Position/Segment Embedding、Transformer Encoder Stack、
MLM（Masked Language Modeling）与 NSP（Next Sentence Prediction）任务演示。

核心思想：
    通过深层双向 Transformer Encoder 学习上下文表示；
    预训练任务 MLM 强制模型根据上下文预测被遮罩的词，NSP 学习句子间关系。

时间复杂度：O(n^2 * d)（与 Transformer Encoder 相同）
空间复杂度：O(n * d)
面试频率：极高
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class BertEmbedding(nn.Module):
    """
    BERT 的三重嵌入：Token Embedding + Position Embedding + Segment Embedding。
    与原始 Transformer 不同，BERT 的位置编码是可学习的（Learned Positional Embedding）。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 512,
        segment_size: int = 2,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        # BERT 使用可学习的位置嵌入，而非正弦编码
        self.pos_emb = nn.Embedding(max_len, d_model)
        # 句子 A/B 的段嵌入，用于区分两个句子
        self.seg_emb = nn.Embedding(segment_size, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch_size, seq_len)
            segment_ids: (batch_size, seq_len)，0 表示句子 A，1 表示句子 B
        Returns:
            (batch_size, seq_len, d_model)
        """
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(input_ids)  # (batch, seq_len)

        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)

        # 三重嵌入相加
        embeddings = (
            self.token_emb(input_ids)
            + self.pos_emb(pos_ids)
            + self.seg_emb(segment_ids)
        )
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultiHeadAttention(nn.Module):
    """与 Transformer 相同的多头自注意力，支持 Attention Mask。"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(context)


class FeedForward(nn.Module):
    """BERT 的 FFN 使用 GELU 激活函数，而非 ReLU。"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GELU 比 ReLU 更平滑，在预训练语言模型中表现更好
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class BertEncoderLayer(nn.Module):
    """
    BERT Encoder Layer（Post-LN 版本，与原始 Transformer 一致）。
    结构：Self-Attention -> Add&Norm -> FFN -> Add&Norm
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        _x = x
        x = self.self_attn(x, x, x, mask=mask)
        x = self.dropout1(x)
        x = self.norm1(_x + x)

        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(_x + x)
        return x


class BertEncoder(nn.Module):
    """由 N 个 BertEncoderLayer 堆叠而成的编码器。"""

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
        self.embedding = BertEmbedding(vocab_size, d_model, max_len, dropout=dropout, padding_idx=padding_idx)
        self.layers = nn.ModuleList(
            [BertEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.embedding(input_ids, segment_ids)
        # attention_mask: (batch, seq_len) -> 扩展为 (batch, 1, 1, seq_len) 用于多头注意力
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            x = layer(x, attention_mask)
        return x


class BertPooler(nn.Module):
    """
    BERT 的 Pooler：取 [CLS] token 的输出，经过线性层 + Tanh，得到句子表示。
    用于 NSP 和下游分类任务。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, d_model)
        cls_token = hidden_states[:, 0]  # 取第一个 token [CLS]
        pooled = self.dense(cls_token)
        pooled = self.activation(pooled)
        return pooled


class BertModel(nn.Module):
    """
    完整的 BERT Base 模型（不含预训练头）。
    输出最后一层隐藏状态 + Pooler 输出。
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        max_len: int = 512,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.encoder = BertEncoder(
            vocab_size, d_model, max_len, n_layers, n_heads, d_ff, dropout, padding_idx
        )
        self.pooler = BertPooler(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            sequence_output: (batch, seq_len, d_model)
            pooled_output:   (batch, d_model)
        """
        sequence_output = self.encoder(input_ids, segment_ids, attention_mask)
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


class BertPretrainingHeads(nn.Module):
    """
    BERT 预训练的两个任务头：
    1. MLM Head：预测被遮罩的 token（输出 vocab_size 维 logits）
    2. NSP Head：预测两个句子是否连续（二分类）
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        # MLM：先经过与输入 embedding 绑定的变换（这里简化为独立线性层），再映射到词表
        self.mlm_dense = nn.Linear(d_model, d_model)
        self.mlm_activation = nn.GELU()
        self.mlm_layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.mlm_classifier = nn.Linear(d_model, vocab_size)

        # NSP：二分类
        self.nsp_classifier = nn.Linear(d_model, 2)

    def forward(
        self, sequence_output: torch.Tensor, pooled_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # MLM
        mlm_hidden = self.mlm_dense(sequence_output)
        mlm_hidden = self.mlm_activation(mlm_hidden)
        mlm_hidden = self.mlm_layer_norm(mlm_hidden)
        mlm_logits = self.mlm_classifier(mlm_hidden)  # (batch, seq_len, vocab_size)

        # NSP
        nsp_logits = self.nsp_classifier(pooled_output)  # (batch, 2)
        return mlm_logits, nsp_logits


class BertForPretraining(nn.Module):
    """
    带预训练头的完整 BERT，支持 MLM + NSP 联合训练。
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 768,
        max_len: int = 512,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = BertModel(vocab_size, d_model, max_len, n_layers, n_heads, d_ff, dropout)
        self.cls = BertPretrainingHeads(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_output, pooled_output = self.bert(input_ids, segment_ids, attention_mask)
        mlm_logits, nsp_logits = self.cls(sequence_output, pooled_output)
        return mlm_logits, nsp_logits


# ==================== MLM / NSP 数据构造工具 ====================

def create_mlm_labels(
    input_ids: torch.Tensor,
    vocab_size: int,
    mask_token_id: int = 103,
    pad_token_id: int = 0,
    mlm_prob: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    构造 MLM 的输入和标签。
    策略（与原始 BERT 一致）：
      - 80% 概率替换为 [MASK]
      - 10% 概率替换为随机 token
      - 10% 概率保持不变
    标签中未被选中的位置设为 -100（CrossEntropy 的 ignore_index）。

    Args:
        input_ids: (batch, seq_len)
    Returns:
        mlm_input_ids: (batch, seq_len)
        labels: (batch, seq_len)
    """
    labels = input_ids.clone()
    # 构造概率矩阵，排除填充位
    prob_matrix = torch.full(labels.shape, mlm_prob)
    prob_matrix[input_ids == pad_token_id] = 0.0

    # 对每个 token 以 mlm_prob 概率决定是否被选中
    masked_indices = torch.bernoulli(prob_matrix).bool()
    # 标签中未被选中的位置设为 -100（不参与 loss 计算）
    labels[~masked_indices] = -100

    mlm_input_ids = input_ids.clone()
    # 80% 替换为 [MASK]
    mask_replacement = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    mlm_input_ids[mask_replacement] = mask_token_id

    # 10% 替换为随机 token
    random_replacement = (
        torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~mask_replacement
    )
    random_words = torch.randint(1, vocab_size, labels.shape, device=input_ids.device)
    mlm_input_ids[random_replacement] = random_words[random_replacement]

    # 剩余 10% 保持不变
    return mlm_input_ids, labels


def create_nsp_data(
    sentences: list[list[int]],
    vocab_size: int,
    cls_token_id: int = 101,
    sep_token_id: int = 102,
    pad_token_id: int = 0,
    max_len: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    构造 NSP 数据：50% 正例（真实下一句），50% 负例（随机采样下一句）。

    Args:
        sentences: 句子列表，每个句子是 token id 列表
    Returns:
        input_ids: (batch, max_len)
        segment_ids: (batch, max_len)
        attention_mask: (batch, max_len)
        labels: (batch,)，0 表示 IsNext，1 表示 NotNext
    """
    batch_size = len(sentences) // 2
    input_ids_list = []
    segment_ids_list = []
    attention_mask_list = []
    labels_list = []

    for i in range(batch_size):
        sent_a = sentences[2 * i]
        # 50% 概率使用真实下一句，50% 概率随机采样
        if random.random() < 0.5:
            sent_b = sentences[2 * i + 1]
            label = 0  # IsNext
        else:
            rand_idx = random.randint(0, len(sentences) - 1)
            sent_b = sentences[rand_idx]
            label = 1  # NotNext

        # 拼接：[CLS] + sent_a + [SEP] + sent_b + [SEP]
        tokens = [cls_token_id] + sent_a + [sep_token_id] + sent_b + [sep_token_id]
        segment = [0] * (1 + len(sent_a) + 1) + [1] * (len(sent_b) + 1)

        # Padding
        attention_mask = [1] * len(tokens)
        if len(tokens) < max_len:
            pad_len = max_len - len(tokens)
            tokens += [pad_token_id] * pad_len
            segment += [0] * pad_len
            attention_mask += [0] * pad_len
        else:
            tokens = tokens[:max_len]
            segment = segment[:max_len]
            attention_mask = attention_mask[:max_len]

        input_ids_list.append(tokens)
        segment_ids_list.append(segment)
        attention_mask_list.append(attention_mask)
        labels_list.append(label)

    return (
        torch.tensor(input_ids_list, dtype=torch.long),
        torch.tensor(segment_ids_list, dtype=torch.long),
        torch.tensor(attention_mask_list, dtype=torch.long),
        torch.tensor(labels_list, dtype=torch.long),
    )


# ==================== Self-Test ====================
if __name__ == "__main__":
    VOCAB_SIZE = 1000
    D_MODEL = 128
    MAX_LEN = 64
    N_LAYERS = 4
    N_HEADS = 4
    D_FF = 512
    BATCH_SIZE = 2
    SEQ_LEN = 16

    # 1) 测试 BertModel 前向传播
    model = BertModel(vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_len=MAX_LEN,
                      n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF)
    input_ids = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    segment_ids = torch.zeros_like(input_ids)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[0, -3:] = 0  # 模拟 padding

    seq_out, pooled_out = model(input_ids, segment_ids, attention_mask)
    print(f"[BERT] Sequence output shape: {seq_out.shape}")  # (B, seq_len, d_model)
    print(f"[BERT] Pooled output shape: {pooled_out.shape}")  # (B, d_model)

    # 2) 测试 MLM 数据构造
    mlm_input, mlm_labels = create_mlm_labels(input_ids, vocab_size=VOCAB_SIZE)
    print(f"[BERT] MLM input shape: {mlm_input.shape}, labels shape: {mlm_labels.shape}")
    print(f"[BERT] MLM masked positions: {(mlm_labels != -100).sum().item()}")

    # 3) 测试 NSP 数据构造
    dummy_sentences = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]
    nsp_input, nsp_seg, nsp_mask, nsp_labels = create_nsp_data(
        dummy_sentences, vocab_size=VOCAB_SIZE, max_len=MAX_LEN
    )
    print(f"[BERT] NSP input shape: {nsp_input.shape}, labels: {nsp_labels.tolist()}")

    # 4) 测试预训练头
    pretrain_model = BertForPretraining(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, max_len=MAX_LEN,
        n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF
    )
    mlm_logits, nsp_logits = pretrain_model(nsp_input, nsp_seg, nsp_mask)
    print(f"[BERT] MLM logits shape: {mlm_logits.shape}")  # (B, seq_len, vocab_size)
    print(f"[BERT] NSP logits shape: {nsp_logits.shape}")  # (B, 2)

    # 5) 计算一次 MLM + NSP 的 loss
    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, VOCAB_SIZE), mlm_labels.view(-1), ignore_index=-100
    )
    nsp_loss = F.cross_entropy(nsp_logits, nsp_labels)
    total_loss = mlm_loss + nsp_loss
    print(f"[BERT] MLM loss: {mlm_loss.item():.4f}, NSP loss: {nsp_loss.item():.4f}, Total: {total_loss.item():.4f}")
    print("Self-test passed.")
