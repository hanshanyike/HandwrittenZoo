"""
微调模块单元测试
测试 LoRA、QLoRA、Prefix Tuning
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from fine_tuning.lora import LoRALayer, LinearWithLoRA
from fine_tuning.qlora import QLoRALayer, LinearWithQLoRA
from fine_tuning.prefix_tuning import (
    PrefixEmbedding,
    PrefixTuningMLP,
    PrefixTuningAttention,
    PrefixTuningTransformer,
)


class TestLoRA:
    """测试 LoRA"""

    def test_lora_output_shape(self):
        in_features = 64
        out_features = 128
        rank = 8
        lora = LoRALayer(in_features, out_features, rank)
        x = torch.randn(2, in_features)
        out = lora(x)
        assert out.shape == (2, out_features)

    def test_lora_rank(self):
        in_features = 64
        out_features = 128
        rank = 8
        lora = LoRALayer(in_features, out_features, rank)
        assert lora.lora_A.shape == (in_features, rank)
        assert lora.lora_B.shape == (rank, out_features)

    def test_linear_with_lora(self):
        in_features = 64
        out_features = 128
        rank = 8
        linear = torch.nn.Linear(in_features, out_features)
        lora_linear = LinearWithLoRA(linear, rank)
        x = torch.randn(2, in_features)
        out = lora_linear(x)
        assert out.shape == (2, out_features)

    def test_lora_freeze_base(self):
        in_features = 64
        out_features = 128
        rank = 8
        linear = torch.nn.Linear(in_features, out_features)
        lora_linear = LinearWithLoRA(linear, rank)
        # 基础线性层应被冻结
        for param in linear.parameters():
            assert not param.requires_grad

    def test_lora_trainable(self):
        in_features = 64
        out_features = 128
        rank = 8
        lora = LoRALayer(in_features, out_features, rank)
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad


class TestQLoRA:
    """测试 QLoRA"""

    def test_qlora_output_shape(self):
        in_features = 64
        out_features = 128
        rank = 8
        qlora = QLoRALayer(in_features, out_features, rank)
        x = torch.randn(2, in_features)
        out = qlora(x)
        assert out.shape == (2, out_features)

    def test_linear_with_qlora(self):
        in_features = 64
        out_features = 128
        rank = 8
        linear = torch.nn.Linear(in_features, out_features)
        qlora_linear = LinearWithQLoRA(linear, rank)
        x = torch.randn(2, in_features)
        out = qlora_linear(x)
        assert out.shape == (2, out_features)

    def test_quantization(self):
        in_features = 64
        out_features = 128
        rank = 8
        qlora = QLoRALayer(in_features, out_features, rank)
        # 量化后的权重应为低精度表示
        assert hasattr(qlora, 'quantized_weight')


class TestPrefixTuning:
    """测试 Prefix Tuning"""

    def test_prefix_embedding_shape(self):
        num_layers = 4
        num_heads = 8
        head_dim = 64
        prefix_len = 20
        pe = PrefixEmbedding(num_layers, num_heads, head_dim, prefix_len)
        prefix_k, prefix_v = pe(0)
        assert prefix_k.shape == (num_heads, prefix_len, head_dim)
        assert prefix_v.shape == (num_heads, prefix_len, head_dim)

    def test_prefix_tuning_mlp_shape(self):
        prefix_len = 20
        num_layers = 4
        num_heads = 8
        head_dim = 64
        mlp = PrefixTuningMLP(prefix_len, num_layers, num_heads, head_dim)
        out = mlp()
        assert out.shape == (num_layers, 2, num_heads, prefix_len, head_dim)

    def test_prefix_attention_shape(self):
        d_model = 64
        num_heads = 8
        prefix_len = 10
        layer_idx = 0
        attn = PrefixTuningAttention(d_model, num_heads, prefix_len, layer_idx)
        x = torch.randn(2, 5, d_model)
        prefix_k = torch.randn(num_heads, prefix_len, d_model // num_heads)
        prefix_v = torch.randn(num_heads, prefix_len, d_model // num_heads)
        out = attn(x, prefix_k, prefix_v)
        assert out.shape == (2, 5, d_model)

    def test_prefix_transformer_forward(self):
        vocab_size = 100
        d_model = 64
        n_layers = 2
        num_heads = 8
        prefix_len = 10
        model = PrefixTuningTransformer(
            vocab_size, d_model, 50, n_layers, num_heads, prefix_len=prefix_len
        )
        x = torch.randint(0, vocab_size, (2, 5))
        out = model(x)
        assert out.shape == (2, 5, vocab_size)

    def test_prefix_transformer_frozen(self):
        vocab_size = 100
        d_model = 64
        n_layers = 2
        num_heads = 8
        prefix_len = 10
        model = PrefixTuningTransformer(
            vocab_size, d_model, 50, n_layers, num_heads, prefix_len=prefix_len
        )
        # 非前缀参数应被冻结
        for name, param in model.named_parameters():
            if "prefix" not in name:
                assert not param.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
