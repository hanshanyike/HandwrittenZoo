"""
Transformer 模块单元测试
测试 Transformer、BERT、Decoder-Only 模型的结构与前向传播
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from transformer.transformer_full import (
    PositionalEncoding,
    TransformerEmbedding,
    MultiHeadAttention,
    PositionwiseFeedForward,
    EncoderLayer,
    DecoderLayer,
    Transformer,
)
from transformer.bert import (
    BertEmbedding,
    BertEncoderLayer,
    BertPooler,
    BertModel,
    BertMLMHead,
    BertNSPHead,
)
from transformer.decoder_only import (
    DecoderOnlyTransformer,
)


class TestTransformerFull:
    """测试完整 Transformer 模型"""

    def test_positional_encoding_shape(self):
        d_model = 64
        max_len = 100
        pe = PositionalEncoding(d_model, max_len)
        x = torch.randn(2, 10, d_model)
        out = pe(x)
        assert out.shape == (2, 10, d_model)

    def test_transformer_embedding(self):
        vocab_size = 100
        d_model = 64
        max_len = 50
        emb = TransformerEmbedding(vocab_size, d_model, max_len)
        x = torch.randint(0, vocab_size, (2, 10))
        out = emb(x)
        assert out.shape == (2, 10, d_model)

    def test_multi_head_attention(self):
        d_model = 64
        num_heads = 8
        mha = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(2, 10, d_model)
        out, attn = mha(x, x, x)
        assert out.shape == (2, 10, d_model)
        assert attn.shape == (2, num_heads, 10, 10)

    def test_positionwise_ffn(self):
        d_model = 64
        d_ff = 256
        ffn = PositionwiseFeedForward(d_model, d_ff)
        x = torch.randn(2, 10, d_model)
        out = ffn(x)
        assert out.shape == (2, 10, d_model)

    def test_encoder_layer(self):
        d_model = 64
        num_heads = 8
        d_ff = 256
        layer = EncoderLayer(d_model, num_heads, d_ff)
        x = torch.randn(2, 10, d_model)
        out = layer(x)
        assert out.shape == (2, 10, d_model)

    def test_decoder_layer(self):
        d_model = 64
        num_heads = 8
        d_ff = 256
        layer = DecoderLayer(d_model, num_heads, d_ff)
        x = torch.randn(2, 10, d_model)
        enc_out = torch.randn(2, 15, d_model)
        out = layer(x, enc_out)
        assert out.shape == (2, 10, d_model)

    def test_transformer_forward(self):
        src_vocab = 100
        tgt_vocab = 100
        d_model = 64
        num_heads = 8
        d_ff = 256
        n_layers = 2
        model = Transformer(src_vocab, tgt_vocab, d_model, num_heads, d_ff, n_layers)
        src = torch.randint(0, src_vocab, (2, 10))
        tgt = torch.randint(0, tgt_vocab, (2, 8))
        out = model(src, tgt)
        assert out.shape == (2, 8, tgt_vocab)


class TestBert:
    """测试 BERT 模型"""

    def test_bert_embedding(self):
        vocab_size = 100
        d_model = 64
        max_len = 50
        emb = BertEmbedding(vocab_size, d_model, max_len)
        input_ids = torch.randint(0, vocab_size, (2, 10))
        token_type_ids = torch.zeros(2, 10, dtype=torch.long)
        out = emb(input_ids, token_type_ids)
        assert out.shape == (2, 10, d_model)

    def test_bert_encoder_layer(self):
        d_model = 64
        num_heads = 8
        d_ff = 256
        layer = BertEncoderLayer(d_model, num_heads, d_ff)
        x = torch.randn(2, 10, d_model)
        out = layer(x)
        assert out.shape == (2, 10, d_model)

    def test_bert_pooler(self):
        d_model = 64
        pooler = BertPooler(d_model)
        x = torch.randn(2, 10, d_model)
        out = pooler(x)
        assert out.shape == (2, d_model)

    def test_bert_model_forward(self):
        vocab_size = 100
        d_model = 64
        num_heads = 8
        d_ff = 256
        n_layers = 2
        model = BertModel(vocab_size, d_model, num_heads, d_ff, n_layers)
        input_ids = torch.randint(0, vocab_size, (2, 10))
        out = model(input_ids)
        assert out.shape == (2, 10, d_model)

    def test_bert_mlm_head(self):
        d_model = 64
        vocab_size = 100
        head = BertMLMHead(d_model, vocab_size)
        x = torch.randn(2, 10, d_model)
        out = head(x)
        assert out.shape == (2, 10, vocab_size)

    def test_bert_nsp_head(self):
        d_model = 64
        head = BertNSPHead(d_model)
        x = torch.randn(2, d_model)
        out = head(x)
        assert out.shape == (2, 2)

    def test_bert_full_forward(self):
        vocab_size = 100
        d_model = 64
        num_heads = 8
        d_ff = 256
        n_layers = 2
        model = BertModel(vocab_size, d_model, num_heads, d_ff, n_layers)
        input_ids = torch.randint(0, vocab_size, (2, 10))
        out = model(input_ids)
        assert out.shape == (2, 10, d_model)


class TestDecoderOnly:
    """测试 Decoder-Only 模型"""

    def test_decoder_only_forward(self):
        vocab_size = 100
        d_model = 64
        num_heads = 8
        d_ff = 256
        n_layers = 2
        max_len = 50
        model = DecoderOnlyTransformer(vocab_size, d_model, num_heads, d_ff, n_layers, max_len)
        x = torch.randint(0, vocab_size, (2, 10))
        out = model(x)
        assert out.shape == (2, 10, vocab_size)

    def test_decoder_only_causal_mask(self):
        vocab_size = 100
        d_model = 64
        num_heads = 8
        d_ff = 256
        n_layers = 2
        max_len = 50
        model = DecoderOnlyTransformer(vocab_size, d_model, num_heads, d_ff, n_layers, max_len)
        mask = model._make_causal_mask(5)
        assert mask.shape == (1, 1, 5, 5)
        # 验证因果性：上三角为 0
        assert torch.all(mask[0, 0, 0, 1:] == 0)
        assert torch.all(mask[0, 0, :, 0] == 1)

    def test_decoder_only_parameter_count(self):
        vocab_size = 100
        d_model = 64
        num_heads = 8
        d_ff = 256
        n_layers = 2
        max_len = 50
        model = DecoderOnlyTransformer(vocab_size, d_model, num_heads, d_ff, n_layers, max_len)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
