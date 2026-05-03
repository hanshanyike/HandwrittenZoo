"""
Group Relative Policy Optimization (GRPO) — DeepSeek-R1 核心算法

算法简介:
    GRPO 是 DeepSeek 团队于 2024 年在 DeepSeekMath 论文中提出的强化学习算法，
    被 DeepSeek-R1 采用为核心训练方法。它通过"组内相对奖励"替代 PPO 中的价值网络（Critic），
    大幅降低显存占用并提升训练稳定性。

核心思想:
    1. 去价值网络：PPO 需要维护一个与策略模型同等规模的价值模型，GRPO 完全去掉 Critic，
       改为对同一 prompt 采样 G 个回答，用组内奖励的均值和标准差做优势归一化。
    2. 组内相对优势：同一组内的回答共享 baseline，优势仅取决于该回答在组内的相对表现，
       天然消除不同 prompt 之间的奖励尺度差异。
    3. 逐 token 惩罚：对每个生成 token 计算 KL 散度惩罚，防止策略偏离参考模型。

时间复杂度: O(G * B * L * d^2)  per step, G=group size
空间复杂度: O(G * B * L)  for storing group rollouts（无 Critic，省一半显存）
面试频率: ★★★★★ (2025 年最热门考点，DeepSeek-R1 带火)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GRPOTrainer:
    """
    Group Relative Policy Optimization 训练器。

    关键差异（对比 PPO）：
    - 无需 Value Model（Critic）
    - 对同一 prompt 采样 G 个输出，用组统计量计算优势
    - 优势对所有 token 相同（per-output advantage，非 per-token）
    """

    def __init__(
        self,
        policy: nn.Module,
        reference_model: nn.Module,
        group_size: int = 4,
        clip_epsilon: float = 0.2,
        kl_coef: float = 0.02,
        max_grad_norm: float = 1.0,
    ):
        """
        Args:
            policy: 当前策略模型（待优化）
            reference_model: 参考模型（固定，通常用 SFT 模型初始化）
            group_size: 每组采样输出数量 G，典型值 4~16
            clip_epsilon: PPO 风格裁剪阈值 ε
            kl_coef: KL 散度惩罚系数 β
            max_grad_norm: 梯度裁剪范数上限
        """
        self.policy = policy
        self.reference_model = reference_model
        self.group_size = group_size
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm

        # 参考模型不参与训练
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def compute_group_advantages(
        self,
        rewards: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        计算组相对优势（Group Relative Advantage）。

        公式:
            A_i = (r_i - mean(r)) / (std(r) + eps)

        Args:
            rewards: [batch_size, group_size] 每组 G 个输出的奖励
            eps: 防止除零的小常数

        Returns:
            advantages: [batch_size, group_size] 归一化后的优势
        """
        # 沿 group 维度计算均值和标准差
        mean_rewards = rewards.mean(dim=1, keepdim=True)  # [B, 1]
        std_rewards = rewards.std(dim=1, keepdim=True)    # [B, 1]
        # 组内归一化：突出相对好坏，消除不同 prompt 的奖励尺度差异
        advantages = (rewards - mean_rewards) / (std_rewards + eps)
        return advantages

    def compute_kl_penalty_per_token(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算逐 token 的 KL 散度: KL(π_θ || π_ref)。

        Args:
            policy_logits: [B*G, L, V] 策略模型 logits
            ref_logits: [B*G, L, V] 参考模型 logits

        Returns:
            kl: [B*G, L] 每个 token 的 KL 值
        """
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        # KL = Σ p_θ * (log p_θ - log p_ref)
        policy_probs = policy_log_probs.exp()
        kl = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)
        return kl

    def grpo_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        kl_penalty: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算 GRPO 损失。

        公式（per token）:
            L = - (1 / (G * |o|)) * Σ_i Σ_t [
                    min(r_t * A_i, clip(r_t) * A_i) - β * KL_t
                ]
            其中 r_t = π_θ(o_{i,t}) / π_old(o_{i,t})

        Args:
            old_log_probs: [B*G, L] 旧策略对数概率
            new_log_probs: [B*G, L] 新策略对数概率
            advantages: [B*G] 每个输出的优势（该输出内所有 token 共享）
            kl_penalty: [B*G, L] 逐 token KL 惩罚
            attention_mask: [B*G, L] 有效 token mask

        Returns:
            loss: 标量损失
            metrics: 包含 clip fraction、KL mean 等指标
        """
        mask = attention_mask.float()

        # 重要性采样比率
        ratio = torch.exp(new_log_probs - old_log_probs)  # [B*G, L]

        # 将 advantages 广播到 token 维度 [B*G, 1] -> [B*G, L]
        advantages_expanded = advantages.unsqueeze(-1)  # [B*G, 1]

        # Clipped surrogate
        surr1 = ratio * advantages_expanded
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_expanded
        # 取 min：与 PPO 一致，限制更新幅度
        surrogate = torch.min(surr1, surr2)

        # 总目标：surrogate - β * KL
        per_token_loss = -(surrogate - self.kl_coef * kl_penalty)

        # 仅对有效 token 求平均
        loss = (per_token_loss * mask).sum() / mask.sum()

        # 辅助指标
        with torch.no_grad():
            clip_fraction = (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean().item()
            kl_mean = (kl_penalty * mask).sum() / mask.sum()

        metrics = {
            "loss": loss.item(),
            "clip_fraction": clip_fraction,
            "kl_mean": kl_mean.item(),
        }
        return loss, metrics

    def update(
        self,
        rollout_data: dict,
        policy_optimizer: torch.optim.Optimizer,
    ) -> dict:
        """
        使用组采样数据执行 GRPO 更新。

        Args:
            rollout_data: 字典，包含:
                - "input_ids": [B*G, L] 生成序列的 token ids
                - "attention_mask": [B*G, L]
                - "old_log_probs": [B*G, L] 旧策略对数概率
                - "rewards": [B, G] 每组 G 个输出的奖励（未展开）
            policy_optimizer: 策略模型优化器

        Returns:
            metrics: 训练指标
        """
        input_ids = rollout_data["input_ids"]
        attention_mask = rollout_data["attention_mask"]
        old_log_probs = rollout_data["old_log_probs"]
        rewards = rollout_data["rewards"]  # [B, G]

        batch_size = rewards.size(0)
        group_size = rewards.size(1)
        seq_len = input_ids.size(1)

        # 计算组相对优势 [B, G] -> 展开为 [B*G]
        advantages = self.compute_group_advantages(rewards)  # [B, G]
        advantages = advantages.view(-1)  # [B*G]

        # 重新前向获取新策略 logits
        policy_logits = self.policy(input_ids, attention_mask=attention_mask).logits

        # 参考模型前向（无梯度）
        with torch.no_grad():
            ref_logits = self.reference_model(input_ids, attention_mask=attention_mask).logits

        # 计算新策略下每个 token 的 log prob
        # 假设 input_ids 是动作序列，取对应位置的 log_prob
        shift_logits = policy_logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        log_probs_all = F.log_softmax(shift_logits, dim=-1)
        new_log_probs = torch.gather(
            log_probs_all, dim=2, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # 对齐 old_log_probs 维度（假设 old_log_probs 也已移位）
        old_log_probs_aligned = old_log_probs[:, 1:].contiguous() if old_log_probs.size(1) == seq_len else old_log_probs

        # 计算 KL 惩罚（逐 token）
        shift_ref_logits = ref_logits[:, :-1, :].contiguous()
        kl_penalty = self.compute_kl_penalty_per_token(shift_logits, shift_ref_logits)

        # 对齐 attention_mask
        mask_aligned = shift_mask

        # 计算 GRPO 损失
        loss, metrics = self.grpo_loss(
            old_log_probs=old_log_probs_aligned,
            new_log_probs=new_log_probs,
            advantages=advantages,
            kl_penalty=kl_penalty,
            attention_mask=mask_aligned,
        )

        policy_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        policy_optimizer.step()

        return metrics


class DummyPolicy(nn.Module):
    """仅供自测的极简策略网络。"""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.logit_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        out, _ = self.lstm(x)
        logits = self.logit_head(out)
        return type("PolicyOutput", (), {"logits": logits})()


if __name__ == "__main__":
    # ===================== 自测模块 =====================
    torch.manual_seed(42)

    vocab_size = 1000
    hidden_size = 128
    batch_size = 2
    group_size = 4
    seq_len = 8

    policy = DummyPolicy(vocab_size, hidden_size)
    ref_model = DummyPolicy(vocab_size, hidden_size)

    trainer = GRPOTrainer(
        policy=policy,
        reference_model=ref_model,
        group_size=group_size,
        clip_epsilon=0.2,
        kl_coef=0.02,
    )

    # 构造 dummy rollout 数据
    total_samples = batch_size * group_size
    input_ids = torch.randint(0, vocab_size, (total_samples, seq_len))
    attention_mask = torch.ones(total_samples, seq_len, dtype=torch.long)
    old_log_probs = torch.randn(total_samples, seq_len) * -1.0
    # 模拟奖励：让组内有一定差异
    rewards = torch.randn(batch_size, group_size)

    rollout = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "old_log_probs": old_log_probs,
        "rewards": rewards,
    }

    policy_opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    metrics = trainer.update(rollout, policy_opt)
    print("[GRPO Self-Test]")
    print(f"Metrics: {metrics}")
    print("GRPO 前向/反向传播检查通过。")
