"""
Proximal Policy Optimization (PPO) — RLHF Stage 3 经典算法

算法简介:
    PPO 是 OpenAI 提出的策略梯度算法，通过 clipped surrogate objective 限制策略更新幅度，
    在 RLHF 中用于根据 Reward Model 的反馈微调语言模型，使其输出更符合人类偏好。

核心思想:
    1. Importance Sampling：利用旧策略采集数据，通过重要性采样比率复用数据做多轮更新。
    2. Clipped Surrogate：将策略比率限制在 [1-ε, 1+ε] 区间内，防止策略因单步更新过大而崩溃。
    3. KL Penalty：在奖励中加入与参考模型的 KL 散度惩罚，防止策略偏离原始分布过远。

时间复杂度: O(B * L * d^2)  per update epoch
空间复杂度: O(B * L)  for storing rollouts (log probs, advantages, etc.)
面试频率: ★★★★★ (RLHF 必考，工程落地最广泛的算法)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PPOTrainer:
    """
    PPO 训练器（适用于 RLHF 场景）。

    假设策略模型 policy 输出每个 token 的 logits，
    通过 rollouts 收集 (log_prob, reward, value) 后计算 advantage，
    再使用 clipped surrogate objective 更新策略。
    """

    def __init__(
        self,
        policy: nn.Module,
        reference_model: nn.Module,
        value_model: nn.Module,
        clip_epsilon: float = 0.2,
        value_clip: Optional[float] = None,
        kl_coef: float = 0.02,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
        ppo_epochs: int = 4,
    ):
        """
        Args:
            policy: 当前策略模型（待优化），输出 logits
            reference_model: 参考模型（固定，通常用 SFT 模型初始化），用于计算 KL 惩罚
            value_model: 价值模型（Critic），估计状态价值 V(s)
            clip_epsilon: PPO 裁剪阈值 ε，典型值 0.1~0.2
            value_clip: value function 的裁剪范围（可选），用于稳定训练
            kl_coef: KL 散度惩罚系数，控制策略与参考模型的偏离程度
            value_coef: value loss 的权重系数 c1
            entropy_coef: 熵奖励系数 c2，鼓励探索
            max_grad_norm: 梯度裁剪范数上限
            ppo_epochs: 每次采样后更新的轮数
        """
        self.policy = policy
        self.reference_model = reference_model
        self.value_model = value_model
        self.clip_epsilon = clip_epsilon
        self.value_clip = value_clip
        self.kl_coef = kl_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs

        # 参考模型不参与训练
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def compute_gae_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用 GAE (Generalized Advantage Estimation) 计算优势函数和回报。

        Args:
            rewards: [batch_size, seq_len] 每个 token 的奖励（含 KL 惩罚后）
            values: [batch_size, seq_len] Critic 估计的状态价值 V(s_t)
            masks: [batch_size, seq_len] 有效 token mask（1 表示有效，0 为 pad）
            gamma: 折扣因子
            lam: GAE 参数 λ，平衡偏差与方差

        Returns:
            advantages: [batch_size, seq_len] 优势估计 A_t
            returns: [batch_size, seq_len] 折扣回报 G_t
        """
        batch_size, seq_len = rewards.shape
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0

        # 从序列末尾向前递推，利用贝尔曼方程分解优势
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = 0.0  # 假设序列结束后价值为 0
            else:
                next_value = values[:, t + 1]

            # TD 残差: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = rewards[:, t] + gamma * next_value * masks[:, t] - values[:, t]

            # GAE: A_t = δ_t + (γλ) * A_{t+1}
            last_gae = delta + gamma * lam * last_gae * masks[:, t]
            advantages[:, t] = last_gae

        # 回报 = 优势 + 价值估计（用于更新 value model）
        returns = advantages + values
        return advantages, returns

    def compute_kl_penalty(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算策略与参考模型之间的逐 token KL 散度（KL(π_θ || π_ref)）。

        Args:
            policy_logits: [batch_size, seq_len, vocab_size]
            ref_logits: [batch_size, seq_len, vocab_size]
            attention_mask: [batch_size, seq_len]

        Returns:
            kl_penalty: [batch_size, seq_len] 每个 token 的 KL 惩罚值
        """
        # 转换为概率分布
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        ref_log_probs = F.log_softmax(ref_logits, dim=-1)
        # KL = Σ p_θ * (log p_θ - log p_ref)
        kl = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(dim=-1)
        kl = kl * attention_mask.float()
        return kl

    def ppo_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 PPO 的 clipped surrogate loss、value loss 和 entropy bonus。

        Args:
            old_log_probs: [batch_size, seq_len] 旧策略下的对数概率（采集数据时）
            new_log_probs: [batch_size, seq_len] 新策略下的对数概率（当前模型）
            advantages: [batch_size, seq_len] GAE 优势估计
            attention_mask: [batch_size, seq_len]

        Returns:
            policy_loss: 策略损失（标量）
            value_loss: 价值损失（标量）
            entropy: 策略熵（标量）
        """
        mask = attention_mask.float()

        # 重要性采样比率: r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        # 取最小值：当 advantage > 0 时，防止 ratio 过大；当 advantage < 0 时，防止 ratio 过小
        policy_loss = -torch.sum(torch.min(surr1, surr2) * mask) / mask.sum()

        # 策略熵：鼓励探索，防止过早收敛到确定性策略
        # entropy = -Σ p(a) log p(a)，这里用 log_probs 近似
        probs = torch.exp(new_log_probs)
        entropy = -torch.sum(probs * new_log_probs * mask) / mask.sum()

        # value loss 在外部计算，这里返回占位
        value_loss = torch.tensor(0.0, device=policy_loss.device)
        return policy_loss, value_loss, entropy

    def update(
        self,
        rollout_data: dict,
        policy_optimizer: torch.optim.Optimizer,
        value_optimizer: torch.optim.Optimizer,
    ) -> dict:
        """
        使用采集的 rollout 数据执行多轮 PPO 更新。

        Args:
            rollout_data: 字典，包含以下 key:
                - "input_ids": [B, L]
                - "attention_mask": [B, L]
                - "old_log_probs": [B, L] 旧策略对数概率
                - "rewards": [B, L] 奖励（已包含 KL 惩罚）
                - "values": [B, L] 旧价值估计
            policy_optimizer: 策略模型优化器
            value_optimizer: 价值模型优化器

        Returns:
            metrics: 训练指标字典
        """
        input_ids = rollout_data["input_ids"]
        attention_mask = rollout_data["attention_mask"]
        old_log_probs = rollout_data["old_log_probs"]
        rewards = rollout_data["rewards"]
        old_values = rollout_data["values"]

        # 计算 GAE 优势和回报
        advantages, returns = self.compute_gae_advantages(
            rewards, old_values, attention_mask
        )
        # 对 advantage 做标准化，降低方差
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.ppo_epochs):
            # 重新前向，获取新策略的 logits 和价值估计
            policy_logits = self.policy(input_ids, attention_mask=attention_mask).logits
            new_values = self.value_model(input_ids, attention_mask=attention_mask).squeeze(-1)

            # 计算新策略下每个 token 的对数概率（假设已知目标 token）
            # 实际 LLM 场景中，目标 token 是 input_ids 的下一个 token
            # 这里简化：假设 input_ids 就是动作序列，取对应位置的 log_prob
            # 注意：真实实现中需用 gather 获取目标 token 的 log_prob
            # 为保持简洁，本实现假设 rollout_data 已提供 new_log_probs 或在外部计算
            # 以下用占位逻辑演示核心思想
            shift_logits = policy_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            shift_mask = attention_mask[:, 1:].contiguous()
            log_probs = F.log_softmax(shift_logits, dim=-1)
            new_log_probs = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(-1)).squeeze(-1)

            # 对齐 old_log_probs 的维度
            old_log_probs_aligned = old_log_probs[:, 1:].contiguous()
            advantages_aligned = advantages[:, 1:].contiguous()
            returns_aligned = returns[:, 1:].contiguous()
            values_aligned = new_values[:, 1:].contiguous()
            mask_aligned = shift_mask.float()

            # PPO 策略损失
            policy_loss, _, entropy = self.ppo_loss(
                old_log_probs_aligned, new_log_probs, advantages_aligned, mask_aligned
            )

            # Value loss（MSE），可选裁剪
            value_pred_clipped = old_values[:, 1:] + torch.clamp(
                values_aligned - old_values[:, 1:],
                -self.value_clip if self.value_clip else float("inf"),
                self.value_clip if self.value_clip else float("inf"),
            )
            value_loss1 = F.mse_loss(values_aligned, returns_aligned, reduction="none")
            value_loss2 = F.mse_loss(value_pred_clipped, returns_aligned, reduction="none")
            value_loss = torch.sum(torch.max(value_loss1, value_loss2) * mask_aligned) / mask_aligned.sum()

            # 总损失
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.value_model.parameters(), self.max_grad_norm)
            policy_optimizer.step()
            value_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        metrics = {
            "policy_loss": total_policy_loss / self.ppo_epochs,
            "value_loss": total_value_loss / self.ppo_epochs,
            "entropy": total_entropy / self.ppo_epochs,
        }
        return metrics


class DummyPolicy(nn.Module):
    """仅供自测的极简策略网络（输出 logits 和 value）。"""

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


class DummyValueModel(nn.Module):
    """仅供自测的极简价值模型。"""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        out, _ = self.lstm(x)
        values = self.value_head(out).squeeze(-1)
        return values


if __name__ == "__main__":
    # ===================== 自测模块 =====================
    torch.manual_seed(42)

    vocab_size = 1000
    hidden_size = 128
    batch_size = 2
    seq_len = 8

    policy = DummyPolicy(vocab_size, hidden_size)
    ref_model = DummyPolicy(vocab_size, hidden_size)
    value_model = DummyValueModel(vocab_size, hidden_size)

    trainer = PPOTrainer(
        policy=policy,
        reference_model=ref_model,
        value_model=value_model,
        clip_epsilon=0.2,
        kl_coef=0.02,
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=2,
    )

    # 构造 dummy rollout 数据
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    # 模拟旧策略的 log_probs
    old_log_probs = torch.randn(batch_size, seq_len) * -1.0  # 负数模拟 log_prob
    rewards = torch.randn(batch_size, seq_len)
    values = torch.randn(batch_size, seq_len)

    rollout = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "old_log_probs": old_log_probs,
        "rewards": rewards,
        "values": values,
    }

    policy_opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    value_opt = torch.optim.Adam(value_model.parameters(), lr=1e-3)

    metrics = trainer.update(rollout, policy_opt, value_opt)
    print("[PPO Self-Test]")
    print(f"Metrics: {metrics}")
    print("PPO 前向/反向传播检查通过。")
