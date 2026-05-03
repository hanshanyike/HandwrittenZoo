"""
归一化模块单元测试
测试 LayerNorm、RMSNorm、BatchNorm
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from normalization.layer_norm.layer_norm import LayerNorm
from normalization.rms_norm.rms_norm import RMSNorm
from normalization.batch_norm.batch_norm import BatchNorm


class TestLayerNorm:
    """测试 LayerNorm"""

    def test_output_shape(self):
        batch, seq_len, d_model = 2, 10, 64
        ln = LayerNorm(d_model)
        x = torch.randn(batch, seq_len, d_model)
        out = ln(x)
        assert out.shape == (batch, seq_len, d_model)

    def test_mean_zero(self):
        d_model = 64
        ln = LayerNorm(d_model)
        x = torch.randn(10, d_model)
        out = ln(x)
        mean = out.mean(dim=-1)
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)

    def test_std_one(self):
        d_model = 64
        ln = LayerNorm(d_model)
        x = torch.randn(10, d_model)
        out = ln(x)
        std = out.std(dim=-1, unbiased=False)
        assert torch.allclose(std, torch.ones_like(std), atol=1e-4)

    def test_parameter_exists(self):
        d_model = 64
        ln = LayerNorm(d_model)
        assert hasattr(ln, 'gamma')
        assert hasattr(ln, 'beta')
        assert ln.gamma.shape == (d_model,)
        assert ln.beta.shape == (d_model,)


class TestRMSNorm:
    """测试 RMSNorm"""

    def test_output_shape(self):
        batch, seq_len, d_model = 2, 10, 64
        rms = RMSNorm(d_model)
        x = torch.randn(batch, seq_len, d_model)
        out = rms(x)
        assert out.shape == (batch, seq_len, d_model)

    def test_rms_equals_one(self):
        d_model = 64
        rms = RMSNorm(d_model)
        x = torch.randn(10, d_model)
        out = rms(x)
        # RMSNorm 输出均方根应接近 1
        rms_val = torch.sqrt((out ** 2).mean(dim=-1))
        assert torch.allclose(rms_val, torch.ones_like(rms_val), atol=1e-4)

    def test_parameter_exists(self):
        d_model = 64
        rms = RMSNorm(d_model)
        assert hasattr(rms, 'gamma')
        assert rms.gamma.shape == (d_model,)

    def test_no_beta(self):
        d_model = 64
        rms = RMSNorm(d_model)
        assert not hasattr(rms, 'beta')


class TestBatchNorm:
    """测试 BatchNorm"""

    def test_output_shape(self):
        batch, channels, height, width = 4, 16, 32, 32
        bn = BatchNorm(channels)
        x = torch.randn(batch, channels, height, width)
        out = bn(x)
        assert out.shape == (batch, channels, height, width)

    def test_running_stats_updated_in_train(self):
        channels = 16
        bn = BatchNorm(channels)
        bn.train()
        x = torch.randn(4, channels, 8, 8)
        initial_mean = bn.running_mean.clone()
        out = bn(x)
        # 训练模式下 running_mean 应被更新
        assert not torch.allclose(bn.running_mean, initial_mean, atol=1e-6)

    def test_running_stats_frozen_in_eval(self):
        channels = 16
        bn = BatchNorm(channels)
        bn.train()
        x = torch.randn(4, channels, 8, 8)
        bn(x)
        frozen_mean = bn.running_mean.clone()
        bn.eval()
        x2 = torch.randn(4, channels, 8, 8)
        bn(x2)
        # 评估模式下 running_mean 不应变化
        assert torch.allclose(bn.running_mean, frozen_mean, atol=1e-6)

    def test_parameter_exists(self):
        channels = 16
        bn = BatchNorm(channels)
        assert hasattr(bn, 'gamma')
        assert hasattr(bn, 'beta')
        assert bn.gamma.shape == (channels,)
        assert bn.beta.shape == (channels,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
