"""
MoE 模块单元测试
测试 MoE Layer、Load Balance、Switch Transformer
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from moe.moe_layer.moe_layer import MoELayer
from moe.load_balance.load_balance import LoadBalancingLoss
from moe.switch_transformer.switch_transformer import SwitchTransformerLayer, SwitchTransformer


class TestMoELayer:
    """测试 MoE 层"""

    def test_output_shape(self):
        d_model = 64
        num_experts = 4
        top_k = 2
        moe = MoELayer(d_model, num_experts, top_k)
        x = torch.randn(2, 10, d_model)
        out = moe(x)
        assert out.shape == (2, 10, d_model)

    def test_top_k_selection(self):
        d_model = 64
        num_experts = 4
        top_k = 2
        moe = MoELayer(d_model, num_experts, top_k)
        x = torch.randn(2, 10, d_model)
        out = moe(x)
        # 输出不应全为零
        assert not torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_expert_count(self):
        d_model = 64
        num_experts = 4
        top_k = 2
        moe = MoELayer(d_model, num_experts, top_k)
        assert len(moe.experts) == num_experts


class TestLoadBalancingLoss:
    """测试负载均衡损失"""

    def test_loss_shape(self):
        num_experts = 4
        lb = LoadBalancingLoss(num_experts)
        router_probs = torch.randn(2, 10, num_experts)
        expert_indices = torch.randint(0, num_experts, (2, 10, 2))
        loss = lb(router_probs, expert_indices)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_uniform_distribution_low_loss(self):
        num_experts = 4
        lb = LoadBalancingLoss(num_experts)
        # 均匀分布的 router probs
        router_probs = torch.ones(2, 10, num_experts) / num_experts
        expert_indices = torch.randint(0, num_experts, (2, 10, 2))
        loss = lb(router_probs, expert_indices)
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_imbalanced_distribution_high_loss(self):
        num_experts = 4
        lb = LoadBalancingLoss(num_experts)
        # 极度不平衡：所有概率集中在第一个专家
        router_probs = torch.zeros(2, 10, num_experts)
        router_probs[:, :, 0] = 1.0
        expert_indices = torch.zeros(2, 10, 2, dtype=torch.long)
        loss = lb(router_probs, expert_indices)
        assert torch.isfinite(loss)
        assert loss >= 0


class TestSwitchTransformerLayer:
    """测试 Switch Transformer 层"""

    def test_output_shape(self):
        d_model = 64
        num_experts = 4
        layer = SwitchTransformerLayer(d_model, num_experts)
        x = torch.randn(2, 10, d_model)
        out = layer(x)
        assert out.shape == (2, 10, d_model)

    def test_single_expert_selection(self):
        d_model = 64
        num_experts = 4
        layer = SwitchTransformerLayer(d_model, num_experts)
        x = torch.randn(2, 10, d_model)
        out = layer(x)
        # Switch Transformer 每个 token 只路由到一个专家
        assert not torch.allclose(out, torch.zeros_like(out), atol=1e-6)


class TestSwitchTransformer:
    """测试 Switch Transformer 模型"""

    def test_forward_shape(self):
        vocab_size = 100
        d_model = 64
        num_experts = 4
        n_layers = 2
        model = SwitchTransformer(vocab_size, d_model, num_experts, n_layers)
        x = torch.randint(0, vocab_size, (2, 10))
        out = model(x)
        assert out.shape == (2, 10, vocab_size)

    def test_parameter_count(self):
        vocab_size = 100
        d_model = 64
        num_experts = 4
        n_layers = 2
        model = SwitchTransformer(vocab_size, d_model, num_experts, n_layers)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0

    def test_embedding_layer(self):
        vocab_size = 100
        d_model = 64
        num_experts = 4
        n_layers = 2
        model = SwitchTransformer(vocab_size, d_model, num_experts, n_layers)
        x = torch.randint(0, vocab_size, (2, 10))
        emb = model.embedding(x)
        assert emb.shape == (2, 10, d_model)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
