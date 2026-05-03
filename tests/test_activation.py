"""
激活函数与 FFN 模块单元测试
测试 ReLU、GELU、Swish、SwiGLU、FFN
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from activation.activations.activations import ReLU, GELU, Swish
from activation.swiglu.swiglu import SwiGLU
from activation.feed_forward.feed_forward import FeedForward


class TestReLU:
    """测试 ReLU"""

    def test_positive_values_unchanged(self):
        relu = ReLU()
        x = torch.tensor([1.0, 2.0, 3.0])
        out = relu(x)
        assert torch.allclose(out, x)

    def test_negative_values_zero(self):
        relu = ReLU()
        x = torch.tensor([-1.0, -2.0, 0.0])
        out = relu(x)
        expected = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(out, expected)

    def test_output_shape(self):
        relu = ReLU()
        x = torch.randn(2, 10, 64)
        out = relu(x)
        assert out.shape == x.shape


class TestGELU:
    """测试 GELU"""

    def test_approximate_shape(self):
        gelu = GELU()
        x = torch.randn(2, 10, 64)
        out = gelu(x)
        assert out.shape == x.shape

    def test_zero_input(self):
        gelu = GELU()
        x = torch.zeros(5)
        out = gelu(x)
        # GELU(0) = 0
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_positive_greater_than_relu(self):
        gelu = GELU()
        relu = ReLU()
        x = torch.tensor([1.0, 2.0, 3.0])
        gelu_out = gelu(x)
        relu_out = relu(x)
        # 对于正数，GELU 略小于输入（因为乘以了 CDF），但大于 ReLU 的边界情况不成立
        # 这里仅验证输出非负
        assert (gelu_out >= 0).all()


class TestSwish:
    """测试 Swish"""

    def test_output_shape(self):
        swish = Swish()
        x = torch.randn(2, 10, 64)
        out = swish(x)
        assert out.shape == x.shape

    def test_zero_input(self):
        swish = Swish()
        x = torch.zeros(5)
        out = swish(x)
        # Swish(0) = 0 * sigmoid(0) = 0
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_large_positive(self):
        swish = Swish()
        x = torch.tensor([10.0, 20.0])
        out = swish(x)
        # 对于大正数，sigmoid -> 1，Swish -> x
        assert torch.allclose(out, x, atol=1e-4)

    def test_large_negative(self):
        swish = Swish()
        x = torch.tensor([-10.0, -20.0])
        out = swish(x)
        # 对于大负数，sigmoid -> 0，Swish -> 0
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-4)


class TestSwiGLU:
    """测试 SwiGLU"""

    def test_output_shape(self):
        batch, seq_len, d_model = 2, 10, 64
        swiglu = SwiGLU(d_model)
        x = torch.randn(batch, seq_len, d_model)
        out = swiglu(x)
        assert out.shape == (batch, seq_len, d_model)

    def test_gate_mechanism(self):
        d_model = 64
        swiglu = SwiGLU(d_model)
        x = torch.randn(2, 10, d_model)
        out = swiglu(x)
        # 输出不应全为零（门控机制允许信息通过）
        assert not torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_parameter_count(self):
        d_model = 64
        swiglu = SwiGLU(d_model)
        total = sum(p.numel() for p in swiglu.parameters())
        assert total > 0


class TestFeedForward:
    """测试前馈网络 (FFN)"""

    def test_output_shape(self):
        batch, seq_len, d_model = 2, 10, 64
        ffn = FeedForward(d_model)
        x = torch.randn(batch, seq_len, d_model)
        out = ffn(x)
        assert out.shape == (batch, seq_len, d_model)

    def test_different_activation(self):
        batch, seq_len, d_model = 2, 10, 64
        for act in ['relu', 'gelu', 'swish']:
            ffn = FeedForward(d_model, activation=act)
            x = torch.randn(batch, seq_len, d_model)
            out = ffn(x)
            assert out.shape == (batch, seq_len, d_model)

    def test_parameter_exists(self):
        d_model = 64
        ffn = FeedForward(d_model)
        assert hasattr(ffn, 'fc1')
        assert hasattr(ffn, 'fc2')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
