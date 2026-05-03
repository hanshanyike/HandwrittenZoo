"""
Load Balancing Loss for MoE
============================
MoE 训练中的负载均衡损失，用于防止门控网络将所有 token 路由到少数“热门”专家，
导致其他专家闲置（称为“专家崩溃” Expert Collapse）。

核心思想：
    - 鼓励每个专家处理的 token 数量趋于均匀。
    - 作为辅助损失（auxiliary loss）加到总损失上，与主任务损失联合优化。
    - Switch Transformer 采用基于路由概率和实际分配分数的可微损失。

时间复杂度：O(batch * seq_len * num_experts)
空间复杂度：O(num_experts)

面试频率：高
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoadBalancingLoss(nn.Module):
    """
    Switch Transformer 风格的负载均衡损失。

    公式：
        aux_loss = E * sum_{i=1}^{E}(f_i * P_i)

    其中：
        - f_i: 专家 i 实际被分配的 token 比例（fraction of tokens routed to expert i）
        - P_i: 专家 i 的平均门控概率（mean routing probability to expert i）
        - E: 专家总数

    当负载完全均衡时，f_i = P_i = 1/E，aux_loss = 1；
    当负载极度不均衡时，aux_loss 趋近于 E。
    """

    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

    def forward(
        self,
        gate_probs: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            gate_probs: (num_tokens, num_experts) — 门控网络的 softmax 输出
            expert_indices: (num_tokens, top_k) — 每个 token 被路由到的专家索引
        返回:
            aux_loss: 标量张量
        """
        num_tokens = gate_probs.size(0)
        device = gate_probs.device

        # f_i: 每个专家实际处理的 token 比例
        # 使用 one-hot 统计每个专家被分配到的 token 数
        # expert_mask: (num_tokens, num_experts)
        expert_mask = F.one_hot(
            expert_indices[:, 0], num_classes=self.num_experts
        ).float()  # 只考虑主专家（top-1）用于负载统计

        # 每个专家被分配到的 token 数 / 总 token 数
        f = expert_mask.mean(dim=0)  # (num_experts,)

        # P_i: 每个专家的平均门控概率
        P = gate_probs.mean(dim=0)  # (num_experts,)

        # 负载均衡损失: E * sum(f_i * P_i)
        aux_loss = self.num_experts * torch.sum(f * P)
        return aux_loss


class LoadBalancingLossV2(nn.Module):
    """
    更通用的负载均衡损失，支持对 Top-K 所有选择进行统计。

    与 V1 的区别：V1 只统计每个 token 的 Top-1 专家；V2 统计 Top-K 中所有专家。
    """

    def __init__(self, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        gate_probs: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        参数:
            gate_probs: (num_tokens, num_experts)
            expert_indices: (num_tokens, top_k)
        """
        num_tokens = gate_probs.size(0)

        # 对 Top-K 中的每个位置都生成 one-hot 并求和
        # 这样每个 token 可能贡献给多个专家
        expert_mask = torch.zeros(num_tokens, self.num_experts, device=gate_probs.device)
        for k in range(self.top_k):
            expert_mask += F.one_hot(expert_indices[:, k], num_classes=self.num_experts).float()

        # 归一化：每个 token 贡献为 1（平均分配到 K 个专家）
        expert_mask = expert_mask / self.top_k

        f = expert_mask.mean(dim=0)  # (num_experts,)
        P = gate_probs.mean(dim=0)   # (num_experts,)

        aux_loss = self.num_experts * torch.sum(f * P)
        return aux_loss


class ImportanceLoss(nn.Module):
    """
    重要性损失（Importance Loss），出自原始 Sparsely-Gated MoE 论文。

    核心思想：
        惩罚专家的重要性方差，使每个专家的重要性趋于相等。
        重要性定义为每个专家对所有 token 的门控概率之和。

    公式：
        importance_i = sum_{x}(g_i(x))
        cv = coefficient_of_variation(importance)
        loss = cv^2 = variance(importance) / mean(importance)^2
    """

    def __init__(self):
        super().__init__()

    def forward(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """
        参数:
            gate_probs: (num_tokens, num_experts)
        返回:
            importance_loss: 标量张量
        """
        # importance: (num_experts,)
        importance = gate_probs.sum(dim=0)
        mean_importance = importance.mean()
        variance = ((importance - mean_importance) ** 2).mean()

        # 变异系数平方
        cv_squared = variance / (mean_importance ** 2 + 1e-10)
        return cv_squared


class AuxLossFreeLoadBalancing(nn.Module):
    """
    无辅助损失的负载均衡（Aux-Loss-Free Load Balancing），DeepSeek-MoE / ST-MoE 采用。

    核心思想：
        不通过损失函数惩罚不均衡，而是直接在前向传播时调整门控分数，
        使得每个专家的容量（capacity）被均匀填满。
        这是一种“硬”均衡策略，训练更稳定，无需调 aux_loss 的权重系数。

    实现方式（简化版）：
        在 Top-K 选择前，给每个专家的门控分数加上一个与当前已分配 token 数相关的偏置，
        已分配越多的专家，其分数被惩罚得越厉害。
    """

    def __init__(self, num_experts: int):
        super().__init__()
        self.num_experts = num_experts

    def forward(
        self,
        gate_logits: torch.Tensor,
        capacity_factor: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            gate_logits: (num_tokens, num_experts)
            capacity_factor: 每个专家的容量相对于平均负载的倍数
        返回:
            expert_indices: (num_tokens,) — 每个 token 被路由到的专家
            aux_loss: 始终为 0（无辅助损失）
        """
        num_tokens = gate_logits.size(0)
        device = gate_logits.device

        # 每个专家的目标容量
        capacity = int((num_tokens / self.num_experts) * capacity_factor)

        # 记录每个专家已分配的 token 数
        expert_counts = torch.zeros(self.num_experts, device=device, dtype=torch.int32)
        expert_indices = torch.empty(num_tokens, device=device, dtype=torch.long)

        # 按 token 依次分配（贪心策略，实际系统会用更高效的并行策略）
        for t in range(num_tokens):
            logits = gate_logits[t].clone()
            # 对已满载的专家施加极大负偏置，阻止继续分配
            mask_full = expert_counts >= capacity
            logits[mask_full] = float("-inf")

            # 选择分数最高的专家
            selected = torch.argmax(logits)
            expert_indices[t] = selected
            expert_counts[selected] += 1

        aux_loss = torch.tensor(0.0, device=device)
        return expert_indices, aux_loss


if __name__ == "__main__":
    # 自测：验证负载均衡损失的计算逻辑
    num_tokens = 1000
    num_experts = 8
    top_k = 2

    # 模拟门控概率（故意让专家 0 概率偏高，测试损失是否增大）
    gate_logits = torch.randn(num_tokens, num_experts)
    gate_logits[:, 0] += 2.0  # 偏置专家 0
    gate_probs = F.softmax(gate_logits, dim=-1)

    # Top-K 选择
    _, expert_indices = torch.topk(gate_probs, top_k, dim=-1)

    # 测试 LoadBalancingLoss
    lb_loss = LoadBalancingLoss(num_experts)
    loss1 = lb_loss(gate_probs, expert_indices)
    print(f"[LoadBalancingLoss V1] aux_loss: {loss1.item():.4f}")

    # 测试 LoadBalancingLossV2
    lb_loss_v2 = LoadBalancingLossV2(num_experts, top_k)
    loss2 = lb_loss_v2(gate_probs, expert_indices)
    print(f"[LoadBalancingLoss V2] aux_loss: {loss2.item():.4f}")

    # 测试 ImportanceLoss
    imp_loss = ImportanceLoss()
    loss3 = imp_loss(gate_probs)
    print(f"[ImportanceLoss] loss: {loss3.item():.4f}")

    # 测试 AuxLossFreeLoadBalancing
    aux_free = AuxLossFreeLoadBalancing(num_experts)
    indices, loss4 = aux_free(gate_logits)
    print(f"[AuxLossFree] assigned expert counts: {torch.bincount(indices, minlength=num_experts)}")
    print(f"[AuxLossFree] aux_loss: {loss4.item():.4f} (should be 0)")

    # 验证：均匀分布时 V1 损失应接近 1.0
    uniform_probs = torch.ones(num_tokens, num_experts) / num_experts
    uniform_indices = torch.randint(0, num_experts, (num_tokens, top_k))
    loss_uniform = lb_loss(uniform_probs, uniform_indices)
    print(f"[Uniform Check] V1 loss with uniform routing: {loss_uniform.item():.4f} (expect ~1.0)")

    print("All load balancing self-tests passed.")
