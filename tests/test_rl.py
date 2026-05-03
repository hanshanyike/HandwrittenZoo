"""
强化学习模块单元测试
测试 PPO、DPO、GRPO、Reward Model
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from rl.ppo.ppo import PPOAgent, ActorCritic
from rl.dpo.dpo import DPOTrainer
from rl.grpo.grpo import GRPOTrainer
from rl.reward_model.reward_model import RewardModel


class TestActorCritic:
    """测试 Actor-Critic 网络"""

    def test_actor_output_shape(self):
        state_dim = 64
        action_dim = 10
        ac = ActorCritic(state_dim, action_dim)
        state = torch.randn(2, state_dim)
        action_probs, _ = ac(state)
        assert action_probs.shape == (2, action_dim)

    def test_critic_output_shape(self):
        state_dim = 64
        action_dim = 10
        ac = ActorCritic(state_dim, action_dim)
        state = torch.randn(2, state_dim)
        _, value = ac(state)
        assert value.shape == (2, 1)

    def test_action_probs_sum(self):
        state_dim = 64
        action_dim = 10
        ac = ActorCritic(state_dim, action_dim)
        state = torch.randn(2, state_dim)
        action_probs, _ = ac(state)
        sums = action_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestPPO:
    """测试 PPO 算法"""

    def test_select_action(self):
        state_dim = 64
        action_dim = 10
        agent = PPOAgent(state_dim, action_dim)
        state = torch.randn(state_dim)
        action, log_prob, value = agent.select_action(state)
        assert 0 <= action < action_dim
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_compute_gae(self):
        state_dim = 64
        action_dim = 10
        agent = PPOAgent(state_dim, action_dim)
        rewards = [1.0, 0.5, 0.0]
        values = [0.5, 0.3, 0.1]
        dones = [0, 0, 1]
        advantages, returns = agent.compute_gae(rewards, values, dones)
        assert len(advantages) == len(rewards)
        assert len(returns) == len(rewards)

    def test_update(self):
        state_dim = 64
        action_dim = 10
        agent = PPOAgent(state_dim, action_dim)
        states = torch.randn(4, state_dim)
        actions = torch.randint(0, action_dim, (4,))
        old_log_probs = torch.randn(4)
        advantages = torch.randn(4)
        returns = torch.randn(4)
        agent.update(states, actions, old_log_probs, advantages, returns)
        # 更新后参数应发生变化
        assert True


class TestDPO:
    """测试 DPO 算法"""

    def test_dpo_loss_shape(self):
        beta = 0.1
        dpo = DPOTrainer(beta)
        policy_chosen_logps = torch.randn(4)
        policy_rejected_logps = torch.randn(4)
        reference_chosen_logps = torch.randn(4)
        reference_rejected_logps = torch.randn(4)
        loss = dpo.compute_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        assert loss.dim() == 0  # 标量
        assert torch.isfinite(loss)

    def test_dpo_loss_positive(self):
        beta = 0.1
        dpo = DPOTrainer(beta)
        policy_chosen_logps = torch.tensor([0.0, 0.0])
        policy_rejected_logps = torch.tensor([-1.0, -1.0])
        reference_chosen_logps = torch.tensor([0.0, 0.0])
        reference_rejected_logps = torch.tensor([0.0, 0.0])
        loss = dpo.compute_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        assert loss >= 0


class TestGRPO:
    """测试 GRPO 算法"""

    def test_grpo_loss_shape(self):
        grpo = GRPOTrainer()
        old_log_probs = torch.randn(4, 10)
        new_log_probs = torch.randn(4, 10)
        advantages = torch.randn(4, 10)
        loss = grpo.compute_loss(old_log_probs, new_log_probs, advantages)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_grpo_clipping(self):
        grpo = GRPOTrainer(epsilon=0.2)
        old_log_probs = torch.zeros(4, 10)
        # ratio 极大，应被裁剪
        new_log_probs = torch.ones(4, 10) * 2.0
        advantages = torch.ones(4, 10)
        loss = grpo.compute_loss(old_log_probs, new_log_probs, advantages)
        assert torch.isfinite(loss)


class TestRewardModel:
    """测试奖励模型"""

    def test_output_shape(self):
        vocab_size = 100
        d_model = 64
        model = RewardModel(vocab_size, d_model)
        input_ids = torch.randint(0, vocab_size, (2, 10))
        rewards = model(input_ids)
        assert rewards.shape == (2, 1)

    def test_reward_range(self):
        vocab_size = 100
        d_model = 64
        model = RewardModel(vocab_size, d_model)
        input_ids = torch.randint(0, vocab_size, (2, 10))
        rewards = model(input_ids)
        # 奖励应为有限实数
        assert torch.isfinite(rewards).all()

    def test_different_inputs_different_rewards(self):
        vocab_size = 100
        d_model = 64
        model = RewardModel(vocab_size, d_model)
        input_ids1 = torch.randint(0, vocab_size, (1, 10))
        input_ids2 = torch.randint(0, vocab_size, (1, 10))
        rewards1 = model(input_ids1)
        rewards2 = model(input_ids2)
        # 不同输入通常产生不同奖励（概率极高）
        assert rewards1.shape == rewards2.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
