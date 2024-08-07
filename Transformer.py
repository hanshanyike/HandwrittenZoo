import torch
from torch import nn
import torch.nn.functional as F
import math

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        """
        初始化词嵌入层，用于将词汇表中的每个词映射到一个固定维度的向量空间。

        :param vocab_size: 词汇表中不同词的数量。
        :param d_model: 每个词嵌入向量的维度。
        """
        # padding_idx: 指定的填充词索引，用于序列填充。使用nn.Embedding的初始化
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, maxlen, device):
        """
        初始化位置嵌入层，用于生成序列的位置编码。

        :param d_model: 模型的维度。
        :param maxlen: 序列的最大长度。
        :param device: 指定设备（CPU或GPU）。
        """
        super(PositionalEmbedding, self).__init__()
        # 初始化一个全0的位置编码矩阵，大小为(maxlen, d_model)。
        self.encoding = torch.zeros(maxlen, d_model, device=device)
        # 设置位置编码矩阵的梯度为False，即在训练中不更新位置编码。
        self.encoding.requires_grad_(False)

        # 创建一个从0到maxlen-1的序列，用于计算正弦和余弦函数的位置信息。(maxlen,1)
        pos = torch.arange(0, maxlen, device=device).float().unsqueeze(1)

        # 创建一个从0到d_model,步长为2的序列。(d_model//2)
        _2i = torch.arange(0, d_model, 2, device=device)

        # 填充编码矩阵的0,2,4...列（偶数索引列）为正弦函数的值。(maxlen,d_model//2)
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        # 填充编码矩阵的1,3,5...列（奇数索引列）为余弦函数的值。
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        """
        根据输入序列的长度返回对应长度的位置编码。

        :param x: 输入序列。
        :return: 输入序列长度对应位置编码的子集。
        """
        seq_len = x.shape[1]  # 获取序列长度
        # 返回位置编码矩阵中与序列长度相等的前seq_len行。
        return self.encoding[:seq_len, :]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        初始化Transformer的嵌入层，包括词嵌入和位置嵌入。

        :param vocab_size: 词汇表的大小。
        :param d_model: 模型的维度。
        :param max_len: 序列的最大长度。
        :param drop_prob: Dropout概率。
        :param device: 指定设备（CPU或GPU）。
        """
        super(TransformerEmbedding, self).__init__()
        # 初始化词嵌入层。
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        # 初始化位置嵌入层。
        self.pos_emb = PositionalEmbedding(d_model, max_len, device)
        # 初始化Dropout层，用于正则化以防止过拟合。
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        """
        前向传播函数，将词嵌入和位置嵌入相加，并应用Dropout。

        :param x: 输入的索引序列，形状为(batch_size, seq_length)。
        :return: Dropout处理后的嵌入序列。
        """
        # 通过词嵌入层获取词嵌入。
        tok_emb = self.tok_emb(x)
        # 获取与输入序列长度相匹配的位置嵌入。
        pos_emb = self.pos_emb(x)
        # 将词嵌入和位置嵌入相加。
        emb = tok_emb + pos_emb
        # 应用Dropout并返回结果。
        return self.drop_out(emb)

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-10):
        """
        初始化层归一化模块。

        :param d_model: 每个样本的特征数量。
        :param eps: 用于数值稳定性的小常数，防止除以零。
        """
        super(LayerNorm, self).__init__()
        # 初始化gamma参数，用于缩放标准化的输入。
        self.gamma = nn.Parameter(torch.ones(d_model))
        # 初始化beta参数，用于平移标准化的输入。
        self.beta = nn.Parameter(torch.zeros(d_model))
        # 指定epsilon值，用于数值稳定性。
        self.eps = eps

    def forward(self, x):
        """
        前向传播函数，对输入进行层归一化。

        :param x: 输入张量，形状为(batch_size, seq_length, d_model)。？
        :return: 归一化后的输出张量。
        """
        # 计算输入的均值，保持维度。
        mean = x.mean(-1, keepdim=True)
        # 计算输入的方差，不使用无偏估计，保持维度。
        var = x.var(-1, unbiased=False, keepdim=True)
        # 标准化输入：(x - mean) / sqrt(var + eps)
        out = (x - mean) / torch.sqrt(var + self.eps)
        # 应用缩放和平移参数。
        out = self.gamma * out + self.beta
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        """
        初始化位置感知的前馈网络。

        :param d_model: 输入和输出的维度。
        :param hidden: 隐藏层的维度。
        :param dropout: Dropout概率，用于正则化。
        """
        super(PositionwiseFeedForward, self).__init__()
        # 第一个线性变换层，将输入从d_model映射到隐藏层维度。
        self.fc1 = nn.Linear(d_model, hidden)
        # 第二个线性变换层，将隐藏层的输出映射回d_model维度。
        self.fc2 = nn.Linear(hidden, d_model)
        # Dropout层，用于正则化以防止过拟合。
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播函数，执行前馈网络操作。

        :param x: 输入张量，形状为(batch_size, seq_length, d_model)。
        :return: 前馈网络处理后的输出张量。
        """
        # 应用第一个线性变换层。
        x = self.fc1(x)
        # 应用ReLU激活函数。
        x = F.relu(x)
        # 应用Dropout。
        x = self.dropout(x)
        # 应用第二个线性变换层。
        x = self.fc2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        """
        初始化多头注意力模块。

        :param d_model: 模型的维度。
        :param n_head: 注意力头的数量。
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        # 为查询（Q）、键（K）、值（V）分别创建线性层。
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        # 结果组合后的线性层。
        self.w_combine = nn.Linear(d_model, d_model)
        # Softmax函数用于归一化注意力分数。
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        前向传播函数，执行多头注意力操作。

        :param q: 查询张量。(batch_size, seq_length, d_model)
        :param k: 键张量。(batch_size, seq_length, d_model)
        :param v: 值张量。(batch_size, seq_length, d_model)
        :param mask: 可选的掩码张量，用于防止未来时间步的信息流动。
        :return: 注意力机制处理后的输出张量。
        """
        # batch批大小 seq_length序列长度 dimension维度
        batch, seq_length, dimension = q.shape
        n_d = self.d_model // self.n_head  # 每个头处理的特征维度
        # 通过线性层将查询、键、值投影到d_model维度
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 重塑形状为 (batch, seq_length, n_head, n_d) 再重排列(batch, head, seq_length, n_d)
        q = q.view(batch, seq_length, self.n_head, n_d).permute(0, 2, 1, 3)
        k = k.view(batch, seq_length, self.n_head, n_d).permute(0, 2, 1, 3)
        v = v.view(batch, seq_length, self.n_head, n_d).permute(0, 2, 1, 3)

        # 计算注意力分数并应用缩放 transpose(-2, -1)交换最后两个维度 k的形状为(batch, head, n_d, seq_length)
        score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(n_d)
        # 如果提供了掩码，将其应用于注意力分数
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)  # 对掩码位置使用极大负值
        # 应用Softmax归一化并进行加权求和
        score = self.softmax(score) @ v

        # 重新排列和整合多头注意力的输出 score形状从(batch, head, seq_length, n_d)变为(batch, seq_length, head, n_d)
        score = score.permute(0, 2, 1, 3).contiguous().view(batch, seq_length, dimension)
        # 通过最后的线性层
        output = self.w_combine(score)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob) -> None:
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)

        x = self.drop1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.drop2(x)
        x = self.norm2(x + _x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc, t_mask, s_mask):
        _x = dec
        x = self.attention1(dec, dec, dec, t_mask)  # 下三角掩码

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            _x = x
            x = self.cross_attention(x, enc, enc, s_mask)

            x = self.dropout2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        env_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layer,
        drop_prob,
        device,
    ):
        super(Encoder, self).__init__()

        self.embedding = TransformerEmbedding(
            env_voc_size, d_model, max_len, drop_prob, device
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, ffn_hidden, n_head, drop_prob)
                for _ in range(n_layer)
            ]
        )

    def forward(self, x, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        dec_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layer,
        drop_prob,
        device,
    ):
        super(Decoder, self).__init__()

        self.embedding = TransformerEmbedding(
            dec_voc_size, d_model, max_len, drop_prob, device
        )

        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, ffn_hidden, n_head, drop_prob)
                for _ in range(n_layer)
            ]
        )

        self.fc = nn.Linear(d_model, dec_voc_size)

    def forward(self, dec, enc, t_mask, s_mask):
        dec = self.embedding(dec)
        for layer in self.layers:
            dec = layer(dec, enc, t_mask, s_mask)

        dec = self.fc(dec)

        return dec


class Transformer(nn.Module):
    def __init__(
        self,
        src_pad_idx,
        trg_pad_idx,
        enc_voc_size,
        dec_voc_size,
        max_len,
        d_model,
        n_heads,
        ffn_hidden,
        n_layers,
        drop_prob,
        device,
    ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            enc_voc_size,
            max_len,
            d_model,
            ffn_hidden,
            n_heads,
            n_layers,
            drop_prob,
            device,
        )
        self.decoder = Decoder(
            dec_voc_size,
            max_len,
            d_model,
            ffn_hidden,
            n_heads,
            n_layers,
            drop_prob,
            device,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)

        # (Batch, Time, len_q, len_k)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)

        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)

        mask = q & k
        return mask

    def make_casual_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = (
            torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        )
        return mask

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(
            trg, trg, self.trg_pad_idx, self.trg_pad_idx
        ) * self.make_casual_mask(trg, trg)
        src_trg_mask = self.make_pad_mask(trg, src, self.trg_pad_idx, self.src_pad_idx)

        enc = self.encoder(src, src_mask)
        ouput = self.decoder(trg, enc, trg_mask, src_trg_mask)
        return ouput


def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)