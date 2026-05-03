"""
推理优化模块单元测试
测试 KV Cache、PageAttention、Quantization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from inference_optimization.kv_cache import KVCache, CachedMultiHeadAttention
from inference_optimization.page_attention import (
    BlockAllocator,
    PagedKVCache,
    PagedAttentionLayer,
)
from inference_optimization.quantization import (
    symmetric_quantize,
    symmetric_dequantize,
    asymmetric_quantize,
    asymmetric_dequantize,
    QuantizedLinear,
    compute_quantization_error,
)


class TestKVCache:
    """测试 KV Cache"""

    def test_init(self):
        batch_size = 2
        max_seq_len = 32
        num_kv_heads = 4
        head_dim = 64
        cache = KVCache(batch_size, max_seq_len, num_kv_heads, head_dim)
        assert cache.current_len == 0
        assert cache.cache_k.shape == (batch_size, num_kv_heads, max_seq_len, head_dim)
        assert cache.cache_v.shape == (batch_size, num_kv_heads, max_seq_len, head_dim)

    def test_update(self):
        batch_size = 2
        max_seq_len = 32
        num_kv_heads = 4
        head_dim = 64
        cache = KVCache(batch_size, max_seq_len, num_kv_heads, head_dim)
        k_new = torch.randn(batch_size, num_kv_heads, 5, head_dim)
        v_new = torch.randn(batch_size, num_kv_heads, 5, head_dim)
        k_full, v_full = cache.update(k_new, v_new)
        assert k_full.shape == (batch_size, num_kv_heads, 5, head_dim)
        assert v_full.shape == (batch_size, num_kv_heads, 5, head_dim)
        assert cache.current_len == 5

    def test_overflow(self):
        batch_size = 2
        max_seq_len = 10
        num_kv_heads = 4
        head_dim = 64
        cache = KVCache(batch_size, max_seq_len, num_kv_heads, head_dim)
        k_new = torch.randn(batch_size, num_kv_heads, 15, head_dim)
        v_new = torch.randn(batch_size, num_kv_heads, 15, head_dim)
        with pytest.raises(AssertionError):
            cache.update(k_new, v_new)

    def test_reset(self):
        batch_size = 2
        max_seq_len = 32
        num_kv_heads = 4
        head_dim = 64
        cache = KVCache(batch_size, max_seq_len, num_kv_heads, head_dim)
        k_new = torch.randn(batch_size, num_kv_heads, 5, head_dim)
        v_new = torch.randn(batch_size, num_kv_heads, 5, head_dim)
        cache.update(k_new, v_new)
        cache.reset()
        assert cache.current_len == 0
        assert torch.all(cache.cache_k == 0)
        assert torch.all(cache.cache_v == 0)


class TestCachedMultiHeadAttention:
    """测试带 KV Cache 的多头注意力"""

    def test_forward_no_cache(self):
        embed_dim = 64
        num_heads = 8
        mha = CachedMultiHeadAttention(embed_dim, num_heads)
        x = torch.randn(2, 10, embed_dim)
        out, past_kv = mha(x, use_cache=False)
        assert out.shape == (2, 10, embed_dim)
        assert past_kv is None

    def test_forward_with_cache(self):
        embed_dim = 64
        num_heads = 8
        mha = CachedMultiHeadAttention(embed_dim, num_heads)
        x = torch.randn(2, 5, embed_dim)
        out, past_kv = mha(x, use_cache=True)
        assert out.shape == (2, 5, embed_dim)
        assert past_kv is not None
        assert len(past_kv) == 2

    def test_cache_consistency(self):
        embed_dim = 64
        num_heads = 8
        mha = CachedMultiHeadAttention(embed_dim, num_heads)
        mha.eval()
        prompt = torch.randn(2, 5, embed_dim)
        with torch.no_grad():
            out_full, _ = mha(prompt, use_cache=False)
            mha.reset_cache()
            out_cache, _ = mha(prompt, use_cache=True)
        assert torch.allclose(out_full, out_cache, atol=1e-5)


class TestBlockAllocator:
    """测试物理块分配器"""

    def test_allocate(self):
        allocator = BlockAllocator(num_blocks=16, block_size=4)
        blocks = allocator.allocate(3)
        assert len(blocks) == 3
        assert len(set(blocks)) == 3

    def test_free(self):
        allocator = BlockAllocator(num_blocks=16, block_size=4)
        blocks = allocator.allocate(3)
        allocator.free(blocks)
        assert len(allocator.free_blocks) == 16

    def test_ref_count(self):
        allocator = BlockAllocator(num_blocks=16, block_size=4)
        blocks = allocator.allocate(2)
        allocator.incr_ref(blocks)
        for b in blocks:
            assert allocator.ref_count[b] == 2


class TestPagedKVCache:
    """测试分页 KV Cache"""

    def test_add_sequence(self):
        cache = PagedKVCache(num_blocks=16, block_size=4, num_kv_heads=4, head_dim=64)
        seq_id = cache.add_sequence(init_len=0)
        assert seq_id in cache.page_tables
        assert cache.seq_lengths[seq_id] == 0

    def test_append(self):
        cache = PagedKVCache(num_blocks=16, block_size=4, num_kv_heads=4, head_dim=64)
        seq_id = cache.add_sequence(init_len=0)
        k_new = torch.randn(4, 64)
        v_new = torch.randn(4, 64)
        cache.append(seq_id, k_new, v_new)
        assert cache.seq_lengths[seq_id] == 1

    def test_get_kv(self):
        cache = PagedKVCache(num_blocks=16, block_size=4, num_kv_heads=4, head_dim=64)
        seq_id = cache.add_sequence(init_len=0)
        k_new = torch.randn(2, 4, 64)
        v_new = torch.randn(2, 4, 64)
        cache.append(seq_id, k_new, v_new)
        k_out, v_out = cache.get_kv(seq_id)
        assert k_out.shape == (2, 4, 64)
        assert v_out.shape == (2, 4, 64)

    def test_fork_sequence(self):
        cache = PagedKVCache(num_blocks=16, block_size=4, num_kv_heads=4, head_dim=64)
        parent = cache.add_sequence(init_len=0)
        k_new = torch.randn(2, 4, 64)
        v_new = torch.randn(2, 4, 64)
        cache.append(parent, k_new, v_new)
        child = cache.fork_sequence(parent)
        assert cache.seq_lengths[child] == cache.seq_lengths[parent]
        assert cache.page_tables[child] == cache.page_tables[parent]


class TestPagedAttentionLayer:
    """测试分页注意力层"""

    def test_forward(self):
        embed_dim = 64
        num_heads = 8
        num_kv_heads = 4
        head_dim = embed_dim // num_heads
        cache = PagedKVCache(num_blocks=16, block_size=4, num_kv_heads=num_kv_heads, head_dim=head_dim)
        layer = PagedAttentionLayer(embed_dim, num_heads, num_kv_heads, head_dim)
        seq_id = cache.add_sequence(init_len=0)
        x = torch.randn(1, 1, embed_dim)
        out = layer(x, cache, seq_id)
        assert out.shape == (1, 1, embed_dim)


class TestQuantization:
    """测试量化"""

    def test_symmetric_quantize_shape(self):
        x = torch.randn(10, 20)
        x_q, scale = symmetric_quantize(x, bits=8)
        assert x_q.shape == x.shape
        assert scale.dim() == 0 or scale.shape[0] == 1

    def test_symmetric_roundtrip(self):
        x = torch.randn(10, 20)
        x_q, scale = symmetric_quantize(x, bits=8)
        x_dq = symmetric_dequantize(x_q, scale)
        # 反量化后应接近原始值
        mse = ((x - x_dq) ** 2).mean().item()
        assert mse < 0.1

    def test_asymmetric_quantize_shape(self):
        x = torch.randn(10, 20)
        x_q, scale, zp = asymmetric_quantize(x, bits=8)
        assert x_q.shape == x.shape

    def test_asymmetric_roundtrip(self):
        x = torch.randn(10, 20)
        x_q, scale, zp = asymmetric_quantize(x, bits=8)
        x_dq = asymmetric_dequantize(x_q, scale, zp)
        mse = ((x - x_dq) ** 2).mean().item()
        assert mse < 0.1

    def test_quantized_linear(self):
        lin = QuantizedLinear(64, 32, bits=8, symmetric=True, per_channel=True)
        lin.quantize_weight()
        x = torch.randn(4, 64)
        out = lin(x)
        assert out.shape == (4, 32)

    def test_quantization_error(self):
        x = torch.randn(10, 20)
        mse_8 = compute_quantization_error(x, bits=8)
        mse_4 = compute_quantization_error(x, bits=4)
        # INT8 精度应高于 INT4
        assert mse_8 < mse_4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
