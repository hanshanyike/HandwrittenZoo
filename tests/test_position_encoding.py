"""
位置编码模块单元测试
测试 Sinusoidal、RoPE、ALiBi、Learnable
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from position_encoding.sinusoidal.sinusoidal import SinusoidalPositionEncoding
from position_encoding.rope.rope import RotaryPositionEmbedding, apply_rotary_pos_emb
from position_encoding.alibi.alibi import ALiBiPositionBias
from position_encoding.learnable.learnable import LearnablePositionEncoding


class TestSinusoidal:
    """测试正弦位置编码"""

    def test_output_shape(self):
        batch, seq_len, d_model = 2, 10, 64
        pe = SinusoidalPositionEncoding(d_model)
        x = torch.randn(batch, seq_len, d_model)
        out = pe(x)
        assert out.shape == (batch, seq_len, d_model)

    def test_position_independence(self):
        d_model = 64
        pe = SinusoidalPositionEncoding(d_model)
        x1 = torch.randn(1, 1, d_model)
        x2 = torch.randn(1, 1, d_model)
        out1 = pe(x1)
        out2 = pe(x2)
        # 相同位置编码应加到不同输入上
        assert out1.shape == x1.shape
        assert out2.shape == x2.shape

    def test_different_positions(self):
        d_model = 64
        pe = SinusoidalPositionEncoding(d_model)
        x = torch.zeros(1, 5, d_model)
        out = pe(x)
        # 不同位置应有不同编码值
        assert not torch.allclose(out[0, 0], out[0, 1], atol=1e-6)


class TestRoPE:
    """测试旋转位置编码 (RoPE)"""

    def test_apply_rotary_shape(self):
        batch, num_heads, seq_len, head_dim = 2, 8, 10, 64
        q = torch.randn(batch, num_heads, seq_len, head_dim)
        k = torch.randn(batch, num_heads, seq_len, head_dim)
        q_rot, k_rot = apply_rotary_pos_emb(q, k)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_rope_module(self):
        batch, seq_len, d_model, num_heads = 2, 10, 64, 8
        rope = RotaryPositionEmbedding(d_model, num_heads)
        x = torch.randn(batch, seq_len, d_model)
        out = rope(x)
        assert out.shape == (batch, seq_len, d_model)

    def test_relative_position_effect(self):
        batch, num_heads, seq_len, head_dim = 1, 1, 2, 64
        q = torch.randn(batch, num_heads, seq_len, head_dim)
        k = torch.randn(batch, num_heads, seq_len, head_dim)
        q_rot, k_rot = apply_rotary_pos_emb(q, k)
        # 旋转后不应全等
        assert not torch.allclose(q_rot, q, atol=1e-6)


class TestALiBi:
    """测试 ALiBi 位置偏置"""

    def test_output_shape(self):
        batch, num_heads, q_len, k_len = 2, 8, 10, 15
        alibi = ALiBiPositionBias(num_heads)
        q = torch.randn(batch, num_heads, q_len, 64)
        k = torch.randn(batch, num_heads, k_len, 64)
        out = alibi(q, k)
        assert out.shape == (batch, num_heads, q_len, k_len)

    def test_bias_is_negative(self):
        batch, num_heads, q_len, k_len = 1, 4, 5, 5
        alibi = ALiBiPositionBias(num_heads)
        q = torch.randn(batch, num_heads, q_len, 64)
        k = torch.randn(batch, num_heads, k_len, 64)
        bias = alibi(q, k)
        # ALiBi 偏置应为非正数（对远距离施加惩罚）
        assert (bias <= 0).all()

    def test_heads_have_different_slopes(self):
        num_heads = 4
        alibi = ALiBiPositionBias(num_heads)
        slopes = alibi.slopes
        # 不同头应有不同斜率
        assert slopes.shape == (num_heads, 1, 1)
        assert not torch.allclose(slopes[0], slopes[1], atol=1e-6)


class TestLearnable:
    """测试可学习位置编码"""

    def test_output_shape(self):
        batch, seq_len, d_model, max_len = 2, 10, 64, 100
        pe = LearnablePositionEncoding(d_model, max_len)
        x = torch.randn(batch, seq_len, d_model)
        out = pe(x)
        assert out.shape == (batch, seq_len, d_model)

    def test_parameter_exists(self):
        d_model, max_len = 64, 100
        pe = LearnablePositionEncoding(d_model, max_len)
        assert hasattr(pe, 'pos_emb')
        assert pe.pos_emb.shape == (max_len, d_model)

    def test_learnable(self):
        d_model, max_len = 64, 100
        pe = LearnablePositionEncoding(d_model, max_len)
        assert pe.pos_emb.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
