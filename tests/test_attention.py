"""
Attention 模块单元测试
测试 MHA、MQA、GQA、MLA、FlashAttention、Self/Cross Attention
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from attention.multi_head_attention import MultiHeadAttention
from attention.multi_query_attention import MultiQueryAttention
from attention.grouped_query_attention import GroupedQueryAttention
from attention.multi_head_latent_attention import MultiHeadLatentAttention
from attention.flash_attention import FlashAttention
from attention.self_attention import SelfAttention
from attention.cross_attention import CrossAttention


class TestMultiHeadAttention:
    """测试多头注意力 (MHA)"""

    def test_output_shape(self):
        batch, seq_len, d_model, num_heads = 2, 10, 64, 8
        mha = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch, seq_len, d_model)
        out, attn = mha(x, x, x)
        assert out.shape == (batch, seq_len, d_model)
        assert attn.shape == (batch, num_heads, seq_len, seq_len)

    def test_attention_weights_sum(self):
        batch, seq_len, d_model, num_heads = 2, 10, 64, 8
        mha = MultiHeadAttention(d_model, num_heads)
        x = torch.randn(batch, seq_len, d_model)
        out, attn = mha(x, x, x)
        # 每行 softmax 和应为 1
        sums = attn.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_different_qkv(self):
        batch, seq_len, d_model, num_heads = 2, 10, 64, 8
        mha = MultiHeadAttention(d_model, num_heads)
        q = torch.randn(batch, seq_len, d_model)
        k = torch.randn(batch, seq_len, d_model)
        v = torch.randn(batch, seq_len, d_model)
        out, attn = mha(q, k, v)
        assert out.shape == (batch, seq_len, d_model)


class TestMultiQueryAttention:
    """测试多查询注意力 (MQA)"""

    def test_output_shape(self):
        batch, seq_len, d_model, num_heads = 2, 10, 64, 8
        mqa = MultiQueryAttention(d_model, num_heads)
        x = torch.randn(batch, seq_len, d_model)
        out, attn = mqa(x, x, x)
        assert out.shape == (batch, seq_len, d_model)
        assert attn.shape == (batch, num_heads, seq_len, seq_len)

    def test_kv_heads_equal_one(self):
        batch, seq_len, d_model, num_heads = 2, 10, 64, 8
        mqa = MultiQueryAttention(d_model, num_heads)
        x = torch.randn(batch, seq_len, d_model)
        out, _ = mqa(x, x, x)
        # 验证 K/V 投影输出维度为 head_dim
        k = mqa.w_k(x)
        assert k.shape == (batch, seq_len, mqa.head_dim)


class TestGroupedQueryAttention:
    """测试分组查询注意力 (GQA)"""

    def test_output_shape(self):
        batch, seq_len, d_model, num_heads, num_kv_heads = 2, 10, 64, 8, 4
        gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
        x = torch.randn(batch, seq_len, d_model)
        out, attn = gqa(x, x, x)
        assert out.shape == (batch, seq_len, d_model)
        assert attn.shape == (batch, num_heads, seq_len, seq_len)

    def test_kv_head_count(self):
        d_model, num_heads, num_kv_heads = 64, 8, 4
        gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
        assert gqa.num_kv_heads == num_kv_heads
        assert gqa.head_dim == d_model // num_heads


class TestMultiHeadLatentAttention:
    """测试多头潜在注意力 (MLA)"""

    def test_output_shape(self):
        batch, seq_len, d_model, num_heads = 2, 10, 64, 8
        mla = MultiHeadLatentAttention(d_model, num_heads)
        x = torch.randn(batch, seq_len, d_model)
        out = mla(x)
        assert out.shape == (batch, seq_len, d_model)

    def test_latent_dim(self):
        d_model, num_heads = 64, 8
        mla = MultiHeadLatentAttention(d_model, num_heads)
        assert hasattr(mla, 'latent_dim')
        assert mla.latent_dim > 0


class TestFlashAttention:
    """测试 Flash Attention"""

    def test_output_shape(self):
        batch, seq_len, d_model, num_heads = 2, 16, 64, 8
        fa = FlashAttention(d_model, num_heads)
        x = torch.randn(batch, seq_len, d_model)
        out = fa(x)
        assert out.shape == (batch, seq_len, d_model)

    def test_chunk_size(self):
        d_model, num_heads = 64, 8
        fa = FlashAttention(d_model, num_heads)
        assert fa.chunk_size > 0


class TestSelfAttention:
    """测试自注意力"""

    def test_output_shape(self):
        batch, seq_len, d_model = 2, 10, 64
        sa = SelfAttention(d_model)
        x = torch.randn(batch, seq_len, d_model)
        out = sa(x)
        assert out.shape == (batch, seq_len, d_model)

    def test_parameter_exists(self):
        d_model = 64
        sa = SelfAttention(d_model)
        assert hasattr(sa, 'wq')
        assert hasattr(sa, 'wk')
        assert hasattr(sa, 'wv')
        assert hasattr(sa, 'wo')


class TestCrossAttention:
    """测试交叉注意力"""

    def test_output_shape(self):
        batch, q_len, kv_len, d_model = 2, 10, 15, 64
        ca = CrossAttention(d_model)
        q = torch.randn(batch, q_len, d_model)
        kv = torch.randn(batch, kv_len, d_model)
        out = ca(q, kv)
        assert out.shape == (batch, q_len, d_model)

    def test_different_lengths(self):
        batch, q_len, kv_len, d_model = 2, 5, 20, 64
        ca = CrossAttention(d_model)
        q = torch.randn(batch, q_len, d_model)
        kv = torch.randn(batch, kv_len, d_model)
        out = ca(q, kv)
        assert out.shape == (batch, q_len, d_model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
