"""
Mixture of Experts (MoE) Layer
==============================
稀疏门控混合专家层，通过门控网络将输入动态路由到 Top-K 个专家，实现模型容量的
稀疏扩展。被广泛应用于 Mixtral、GShard、Switch Transformer 等大模型中。

核心思想：
    - 用门控网络（Gating Network）计算输入对每个专家的亲和度（routing score）。
    - 只激活 Top-K 个专家，其余专家不参与计算，实现稀疏性。
    - 每个专家通常是独立的 FFN，输出按门控权重加权求和。

时间复杂度：O(batch * seq_len * d_model * k)  （仅 k 个专家参与计算）
空间复杂度：O(batch * seq_len * num_experts)  （门控分数矩阵）

面试频率：极高
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """
    单个专家网络，通常为标准 FFN（Feed-Forward Network）。
    这里采用两层 Linear + GELU 的经典结构，也可替换为 SwiGLU。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (tokens_in_expert, d_model)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MoELayer(nn.Module):
    """
    稀疏门控混合专家层（Sparse Gated MoE）。

    前向流程：
        1. 将输入展平为 (num_tokens, d_model)。
        2. 门控网络输出每个 token 对每个专家的分数。
        3. Top-K 选择：每个 token 只路由到分数最高的 K 个专家。
        4. 按专家聚合 token，并行计算各专家输出。
        5. 按门控权重加权求和，恢复原始形状。
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        # 若未指定专家中间维度，默认放大 4 倍
        self.d_ff = d_ff if d_ff is not None else 4 * d_model

        # 门控网络：将输入映射为 num_experts 维的 logits
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # 专家池：每个专家独立初始化，参数不共享
        self.experts = nn.ModuleList(
            [Expert(d_model, self.d_ff, dropout) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            x: (batch_size, seq_len, d_model)
        返回:
            output: (batch_size, seq_len, d_model)
            aux_loss: 标量张量，用于负载均衡的辅助损失（此处返回 0，由外部 LoadBalance 模块计算）
        """
        batch_size, seq_len, d_model = x.shape
        # 展平为 (num_tokens, d_model)，方便按 token 路由
        x_flat = x.view(-1, d_model)  # (batch * seq_len, d_model)
        num_tokens = x_flat.size(0)

        # 1) 门控分数: (num_tokens, num_experts)
        gate_logits = self.gate(x_flat)
        # softmax 归一化，得到每个 token 对各专家的概率分布
        gate_probs = F.softmax(gate_logits, dim=-1)

        # 2) Top-K 选择：对每个 token 选 K 个专家
        # topk_values: (num_tokens, top_k), topk_indices: (num_tokens, top_k)
        topk_probs, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        # 对 Top-K 概率重新归一化，使得每个 token 的权重和为 1
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # 3) 初始化输出缓冲区
        output = torch.zeros_like(x_flat)  # (num_tokens, d_model)

        # 4) 按专家聚合 token 并计算
        # 遍历每个专家，收集所有路由到该专家的 token
        for expert_id in range(self.num_experts):
            # 找到哪些 token 的 Top-K 列表中包含当前 expert_id
            # mask: (num_tokens, top_k) 的布尔张量
            mask = topk_indices == expert_id  # (num_tokens, top_k)
            # token 是否被当前专家处理: (num_tokens,)
            token_mask = mask.any(dim=-1)
            if token_mask.sum() == 0:
                continue  # 没有 token 路由到该专家，跳过

            # 收集输入 token
            expert_input = x_flat[token_mask]  # (tokens_for_expert, d_model)
            # 计算该专家的输出
            expert_output = self.experts[expert_id](expert_input)  # (tokens_for_expert, d_model)

            # 获取对应门控权重: (tokens_for_expert,)
            # 使用 mask 索引出每个 token 对应当前专家的权重
            token_weights = topk_probs[token_mask][mask[token_mask]]  # (tokens_for_expert,)
            # 加权后累加到输出缓冲区
            output[token_mask] += token_weights.unsqueeze(-1) * expert_output

        # 5) 恢复形状
        output = output.view(batch_size, seq_len, d_model)

        # 辅助损失占位，实际负载均衡损失由外部模块计算
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return output, aux_loss


class MoELayerEfficient(nn.Module):
    """
    更高效的 MoE 实现（向量化版本）。
    利用 scatter/add 操作减少 Python 循环开销，适合大规模训练。
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_ff = d_ff if d_ff is not None else 4 * d_model

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(d_model, self.d_ff, dropout) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        num_tokens = x_flat.size(0)

        # 门控 + Top-K
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-9)

        output = torch.zeros_like(x_flat)

        # 向量化：将每个 token 的 top_k 选择展开为长向量
        # token_indices: (num_tokens * top_k,)
        token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(self.top_k)
        # expert_indices: (num_tokens * top_k,)
        expert_indices = topk_indices.view(-1)
        # weights: (num_tokens * top_k,)
        weights = topk_probs.view(-1)

        # 按专家分组处理（仍需要循环专家，但内部操作高度向量化）
        for expert_id in range(self.num_experts):
            mask = expert_indices == expert_id
            if mask.sum() == 0:
                continue
            # 属于该专家的 token 索引
            selected_tokens = token_indices[mask]
            selected_weights = weights[mask]
            expert_input = x_flat[selected_tokens]
            expert_out = self.experts[expert_id](expert_input)
            # 加权写回
            weighted_out = selected_weights.unsqueeze(-1) * expert_out
            output.index_add_(0, selected_tokens, weighted_out)

        output = output.view(batch_size, seq_len, d_model)
        aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        return output, aux_loss


if __name__ == "__main__":
    # 自测：验证 MoE 层输出形状及稀疏激活特性
    batch_size, seq_len, d_model = 2, 8, 64
    num_experts, top_k = 8, 2

    moe = MoELayer(d_model, num_experts, top_k)
    x = torch.randn(batch_size, seq_len, d_model)
    out, aux = moe(x)

    assert out.shape == (batch_size, seq_len, d_model), f"输出形状错误: {out.shape}"
    print(f"[MoE Layer] 输入形状: {x.shape}, 输出形状: {out.shape}")
    print(f"[MoE Layer] 专家数: {num_experts}, Top-K: {top_k}")
    print(f"[MoE Layer] 门控输出示例 (第一个 token): {F.softmax(moe.gate(x.view(-1, d_model))[0], dim=-1)}")

    # 测试高效版本
    moe_eff = MoELayerEfficient(d_model, num_experts, top_k)
    out_eff, _ = moe_eff(x)
    assert out_eff.shape == (batch_size, seq_len, d_model)
    print("[MoE Layer Efficient] 自测通过")

    print("All MoE layer self-tests passed.")
