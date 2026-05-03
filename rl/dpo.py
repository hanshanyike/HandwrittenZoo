"""
Direct Preference Optimization (DPO) — 无需奖励模型的 RLHF 对齐算法

算法简介:
    DPO 是 Rafailov 等人于 2023 年提出的偏好优化算法，绕过传统 RLHF 中独立的奖励模型和 PPO，
    直接在偏好数据上优化语言模型，证明"语言模型本身就是奖励模型"。

核心思想:
    1. 从 Bradley-Terry 模型出发，推导出最优策略的闭式解：
       π*(y|x) ∝ π_ref(y|x) * exp(r(x,y)/β)。
    2. 将奖励函数 r(x,y) 用策略和参考模型的对数概率比显式表示，消去奖励模型。
    3. 最终转化为一个二分类交叉熵问题，直接对偏好数据做监督学习。

时间复杂度: O(B * L * d^2)  per step（与一次前向传播同阶）
空间复杂度: O(B * L)  for storing log probs
面试频率: ★★★★★ (2024-2025 年 LLM 对齐面试最热门考点)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization 损失模块。

    核心公式:
        L_DPO = -E[ log σ( β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)) ) ]
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        """
        Args:
            beta: 温度系数，控制偏好强度。β 越小，模型越倾向于严格区分 chosen/rejected；
                  β 越大，策略越接近参考模型。典型值 0.1~0.5。
            label_smoothing: 标签平滑系数，缓解过拟合和偏好噪声。
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def _get_batch_log_probs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算每个序列在模型下的对数似然（per-token log prob 求和）。

        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len] 目标 token ids（与 logits 对齐）
            attention_mask: [batch_size, seq_len] 有效 token mask

        Returns:
            log_probs: [batch_size] 每个序列的总对数概率
        """
        # 将 logits 转换为 log 概率
        log_probs_all = F.log_softmax(logits, dim=-1)
        # 取出目标 token 对应的 log prob
        # gather: [B, L, 1] -> squeeze -> [B, L]
        log_probs = torch.gather(log_probs_all, dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
        # 仅对有效 token 求和，pad 位置不计入
        log_probs = (log_probs * attention_mask.float()).sum(dim=1)
        return log_probs

    def forward(
        self,
        policy_chosen_logits: torch.Tensor,
        policy_rejected_logits: torch.Tensor,
        reference_chosen_logits: torch.Tensor,
        reference_rejected_logits: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_labels: torch.Tensor,
        chosen_mask: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算 DPO 损失及辅助指标。

        Args:
            policy_chosen_logits: [B, L, V] 策略模型对 chosen 序列的 logits
            policy_rejected_logits: [B, L, V] 策略模型对 rejected 序列的 logits
            reference_chosen_logits: [B, L, V] 参考模型对 chosen 序列的 logits（无梯度）
            reference_rejected_logits: [B, L, V] 参考模型对 rejected 序列的 logits（无梯度）
            chosen_labels: [B, L] chosen 序列的目标 labels
            rejected_labels: [B, L] rejected 序列的目标 labels
            chosen_mask: [B, L] chosen 序列的 attention mask
            rejected_mask: [B, L] rejected 序列的 attention mask

        Returns:
            loss: 标量 DPO 损失
            metrics: 包含 accuracy、chosen_reward、rejected_reward 等指标的字典
        """
        # 计算策略模型和参考模型下的序列对数概率
        policy_chosen_logps = self._get_batch_log_probs(
            policy_chosen_logits, chosen_labels, chosen_mask
        )
        policy_rejected_logps = self._get_batch_log_probs(
            policy_rejected_logits, rejected_labels, rejected_mask
        )
        reference_chosen_logps = self._get_batch_log_probs(
            reference_chosen_logits, chosen_labels, chosen_mask
        )
        reference_rejected_logps = self._get_batch_log_probs(
            reference_rejected_logits, rejected_labels, rejected_mask
        )

        # 隐式奖励: r(x,y) = β * log(π_θ(y|x) / π_ref(y|x))
        # 等价于 β * (log π_θ - log π_ref)
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)

        # DPO 损失核心: -log σ(chosen_reward - rejected_reward)
        logits_diff = chosen_rewards - rejected_rewards  # [B]

        if self.label_smoothing > 0:
            # 标签平滑：将目标从 1 略微拉向 0，提升鲁棒性
            loss = (
                -F.logsigmoid(logits_diff) * (1.0 - self.label_smoothing)
                - F.logsigmoid(-logits_diff) * self.label_smoothing
            ).mean()
        else:
            loss = -F.logsigmoid(logits_diff).mean()

        # 辅助指标（不计算梯度）
        with torch.no_grad():
            accuracy = (logits_diff > 0).float().mean().item()
            chosen_reward_avg = chosen_rewards.mean().item()
            rejected_reward_avg = rejected_rewards.mean().item()
            margin = (chosen_rewards - rejected_rewards).mean().item()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy,
            "chosen_reward": chosen_reward_avg,
            "rejected_reward": rejected_reward_avg,
            "margin": margin,
        }
        return loss, metrics


class DPOTrainer:
    """
    简化的 DPO 训练器封装。

    将策略模型和参考模型的前向传播与 DPO 损失计算封装在一起，
    方便在训练循环中直接调用。
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        """
        Args:
            policy_model: 待优化的策略模型（如 GPT/Llama）
            reference_model: 固定参考模型（通常用 SFT 模型初始化）
            beta: DPO 温度系数
            label_smoothing: 标签平滑系数
        """
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.dpo_loss = DPOLoss(beta=beta, label_smoothing=label_smoothing)

        # 参考模型冻结
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def compute_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        给定一对 chosen/rejected 序列，计算 DPO 损失。

        Args:
            chosen_input_ids: [B, L]
            chosen_attention_mask: [B, L]
            rejected_input_ids: [B, L]
            rejected_attention_mask: [B, L]

        Returns:
            loss, metrics
        """
        # 策略模型前向（带梯度）
        policy_chosen_logits = self.policy_model(
            chosen_input_ids, attention_mask=chosen_attention_mask
        ).logits
        policy_rejected_logits = self.policy_model(
            rejected_input_ids, attention_mask=rejected_attention_mask
        ).logits

        # 参考模型前向（无梯度）
        with torch.no_grad():
            reference_chosen_logits = self.reference_model(
                chosen_input_ids, attention_mask=chosen_attention_mask
            ).logits
            reference_rejected_logits = self.reference_model(
                rejected_input_ids, attention_mask=rejected_attention_mask
            ).logits

        # labels 取 input_ids 的移位版本（因果语言模型标准做法）
        # 为简化，这里假设 logits 和 labels 已经对齐（如 logits[:, :-1, :] 对应 labels[:, 1:]）
        # 实际实现中需根据模型结构调整
        # 本实现假设传入的 logits 已经过移位处理，labels 与 logits 对齐
        loss, metrics = self.dpo_loss(
            policy_chosen_logits=policy_chosen_logits,
            policy_rejected_logits=policy_rejected_logits,
            reference_chosen_logits=reference_chosen_logits,
            reference_rejected_logits=reference_rejected_logits,
            chosen_labels=chosen_input_ids,
            rejected_labels=rejected_input_ids,
            chosen_mask=chosen_attention_mask,
            rejected_mask=rejected_attention_mask,
        )
        return loss, metrics


class DummyLM(nn.Module):
    """仅供自测的极简语言模型。"""

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=hidden_size * 4, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        x = self.embedding(input_ids)
        # 构造因果 mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(input_ids.size(1)).to(input_ids.device)
        out = self.transformer(x, x, tgt_mask=tgt_mask)
        logits = self.lm_head(out)
        return type("LMOutput", (), {"logits": logits})()


if __name__ == "__main__":
    # ===================== 自测模块 =====================
    torch.manual_seed(42)

    vocab_size = 1000
    hidden_size = 128
    batch_size = 2
    seq_len = 8

    policy = DummyLM(vocab_size, hidden_size)
    reference = DummyLM(vocab_size, hidden_size)

    trainer = DPOTrainer(policy, reference, beta=0.1, label_smoothing=0.0)

    # 构造 dummy 数据：chosen 用较大 token id，rejected 用较小 token id
    chosen_ids = torch.randint(500, vocab_size, (batch_size, seq_len))
    rejected_ids = torch.randint(0, 500, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    loss, metrics = trainer.compute_loss(chosen_ids, mask, rejected_ids, mask)

    print("[DPO Self-Test]")
    print(f"Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    # 反向传播检查
    loss.backward()
    assert policy.embedding.weight.grad is not None, "梯度未回传！"
    print("梯度回传检查通过。")
