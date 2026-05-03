"""
Reward Model — Bradley-Terry Preference Model (RLHF Stage 2)

算法简介:
    基于 Bradley-Terry 模型，将人类偏好（chosen vs rejected）建模为标量奖励的对比概率，
    通过最大似然估计训练一个奖励模型 r(x, y)，为后续 PPO/GRPO 提供奖励信号。

核心思想:
    1. 假设存在隐式奖励函数 r(x, y)，使得人类偏好的概率满足 logistic 形式。
    2. 将偏好学习转化为二分类交叉熵问题，无需显式建模人类打分分布。
    3. 训练完成后，模型对单个 response 输出标量 reward，用于强化学习阶段。

时间复杂度: O(B * L * d^2)  per step, B=batch, L=seq_len, d=hidden_dim
空间复杂度: O(d^2)  for the reward head
面试频率: ★★★★★ (RLHF 流程必考点)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    """
    Bradley-Terry 风格的奖励模型。

    结构: 一个 transformer encoder（或任意文本编码器）+ 线性头，输出标量 reward。
    训练目标: 对于偏好对 (chosen, rejected)，最大化 chosen 被偏好的概率。
    """

    def __init__(self, encoder: nn.Module, hidden_size: int, dropout: float = 0.1):
        """
        Args:
            encoder: 文本编码器（如 Transformer Encoder 或预训练语言模型），输出 last_hidden_state
            hidden_size: 编码器隐藏层维度
            dropout: dropout 概率
        """
        super().__init__()
        self.encoder = encoder
        # 奖励头：将 [CLS] 或 pooled 表示映射为标量 reward
        self.reward_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算每个样本的标量奖励。

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            rewards: [batch_size] 标量奖励值
        """
        # 编码器输出最后一层隐藏状态
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # 取最后一个有效 token 的 hidden state 作为句子表示（比 CLS 更适合生成任务）
        # 通过 attention_mask 找到每个序列最后一个非 pad token 的位置
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        # 计算每个序列实际长度
        seq_lengths = attention_mask.sum(dim=1) - 1  # [B]
        # 取出最后一个有效 token 的向量
        batch_size = input_ids.size(0)
        pooled = last_hidden[torch.arange(batch_size), seq_lengths]  # [B, H]
        # 映射为标量 reward
        reward = self.reward_head(pooled).squeeze(-1)  # [B]
        return reward


def bradley_terry_loss(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
    label_smoothing: float = 0.0
) -> torch.Tensor:
    """
    Bradley-Terry 偏好损失（二元交叉熵形式）。

    数学推导:
        P(chosen > rejected | x) = sigmoid(r(x, chosen) - r(x, rejected))
        Loss = -E[log sigmoid(r_chosen - r_rejected)]

    Args:
        reward_chosen: [batch_size] chosen 样本的奖励
        reward_rejected: [batch_size] rejected 样本的奖励
        label_smoothing: 标签平滑系数，缓解过拟合

    Returns:
        loss: 标量损失
    """
    # 奖励差值，反映相对偏好强度
    diff = reward_chosen - reward_rejected  # [B]
    # 当 diff 很大时，模型对偏好对非常自信；diff 接近 0 时，模型难以区分
    # label_smoothing 将目标从 1 略微拉低，防止过度自信
    if label_smoothing > 0:
        # 平滑后的目标: (1 - smoothing) * 1 + smoothing * 0 = 1 - smoothing
        # 等价于二元交叉熵中调整 target
        loss = -F.logsigmoid(diff) * (1.0 - label_smoothing) - F.logsigmoid(-diff) * label_smoothing
    else:
        loss = -F.logsigmoid(diff)
    return loss.mean()


def pairwise_accuracy(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> float:
    """
    评估指标：计算模型能正确区分 chosen/rejected 的比例。

    Args:
        reward_chosen: [batch_size]
        reward_rejected: [batch_size]

    Returns:
        accuracy: 标量准确率
    """
    correct = (reward_chosen > reward_rejected).float().sum().item()
    total = reward_chosen.size(0)
    return correct / total


class DummyEncoder(nn.Module):
    """
    仅供自测使用的极简编码器（模拟 BERT/GPT 输出结构）。
    """

    def __init__(self, vocab_size: int = 1000, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        x = self.embedding(input_ids)
        # 简单处理 mask：Transformer 需要 src_key_padding_mask (True 表示 pad)
        if attention_mask is not None:
            src_mask = ~attention_mask.bool()  # [B, L]
        else:
            src_mask = None
        out = self.transformer(x, src_key_padding_mask=src_mask)
        # 模拟 HuggingFace 输出格式
        return type("EncoderOutput", (), {"last_hidden_state": out})()


if __name__ == "__main__":
    # ===================== 自测模块 =====================
    torch.manual_seed(42)

    batch_size = 4
    seq_len = 16
    vocab_size = 1000
    hidden_size = 128

    # 构造 dummy 数据：chosen 的 token id 整体偏大，让模型能学到规律
    chosen_ids = torch.randint(500, vocab_size, (batch_size, seq_len))
    rejected_ids = torch.randint(0, 500, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # 构建模型
    encoder = DummyEncoder(vocab_size, hidden_size)
    rm = RewardModel(encoder, hidden_size)

    # 前向
    r_chosen = rm(chosen_ids, attention_mask)
    r_rejected = rm(rejected_ids, attention_mask)

    # 计算损失
    loss = bradley_terry_loss(r_chosen, r_rejected)
    acc = pairwise_accuracy(r_chosen, r_rejected)

    print("[RewardModel Self-Test]")
    print(f"Chosen rewards:   {r_chosen.detach().numpy()}")
    print(f"Rejected rewards: {r_rejected.detach().numpy()}")
    print(f"BT Loss: {loss.item():.4f}")
    print(f"Pairwise Acc: {acc:.2%}")

    # 反向传播检查
    loss.backward()
    assert encoder.embedding.weight.grad is not None, "梯度未回传，请检查实现！"
    print("梯度回传检查通过。")
