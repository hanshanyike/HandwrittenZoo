"""
Switch Transformer
==================
Google 提出的极简 MoE 架构，核心创新是**每个 token 只路由到 1 个专家（Top-1）**，
并通过容量因子（Capacity Factor）和负载均衡损失解决训练不稳定性问题。

核心思想：
    - 极致稀疏：K=1，计算量最小，通信开销最低。
    - 容量限制：每个专家有最大处理 token 数（capacity），超出部分被丢弃（dropped tokens）。
    - 可微负载均衡：通过 aux_loss 鼓励均匀路由。

时间复杂度：O(batch * seq_len * d_model * d_ff)  （仅 1 个专家激活）
空间复杂度：O(batch * seq_len * num_experts)  （路由表）

面试频率：极高
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwitchRouter(nn.Module):
    """
    Switch Transformer 的路由器（门控网络）。
    对每个 token 输出 num_experts 维 logits，然后选 Top-1。
    """

    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数:
            x: (num_tokens, d_model)
        返回:
            expert_indices: (num_tokens,) — 每个 token 选中的专家索引
            gate_probs: (num_tokens, num_experts) — 完整的 softmax 概率
            expert_gate: (num_tokens,) — 每个 token 对选中专家的门控权重
        """
        gate_logits = self.gate(x)  # (num_tokens, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-1 选择
        expert_gate, expert_indices = torch.max(gate_probs, dim=-1)  # (num_tokens,)
        return expert_indices, gate_probs, expert_gate


class SwitchExpert(nn.Module):
    """
    Switch Transformer 中的单个专家，标准 FFN。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SwitchMoELayer(nn.Module):
    """
    Switch Transformer 的 MoE 层。

    关键机制：
        1. 每个 token 只路由到 1 个专家。
        2. 容量限制：每个专家最多处理 capacity 个 token，超出的被丢弃。
        3. 被丢弃的 token 直接走残差连接（即不经过专家处理）。
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        d_ff: int = None,
        capacity_factor: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.d_ff = d_ff if d_ff is not None else 4 * d_model

        self.router = SwitchRouter(d_model, num_experts)
        self.experts = nn.ModuleList(
            [SwitchExpert(d_model, self.d_ff, dropout) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            x: (batch_size, seq_len, d_model)
        返回:
            output: (batch_size, seq_len, d_model)
            aux_loss: 标量 — 负载均衡辅助损失
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (num_tokens, d_model)
        num_tokens = x_flat.size(0)

        # 1) 路由
        expert_indices, gate_probs, expert_gate = self.router(x_flat)

        # 2) 计算容量
        capacity = int((num_tokens / self.num_experts) * self.capacity_factor)

        # 3) 初始化输出和丢弃标记
        output = torch.zeros_like(x_flat)
        dropped = torch.zeros(num_tokens, dtype=torch.bool, device=x.device)

        # 4) 按专家处理 token
        for expert_id in range(self.num_experts):
            # 找到路由到该专家的所有 token
            token_mask = expert_indices == expert_id  # (num_tokens,)
            selected_indices = torch.where(token_mask)[0]

            if selected_indices.numel() == 0:
                continue

            # 容量截断：超出容量的 token 被丢弃
            if selected_indices.numel() > capacity:
                # 按门控分数排序，保留分数最高的 capacity 个
                selected_gate = expert_gate[selected_indices]
                _, sorted_idx = torch.sort(selected_gate, descending=True)
                keep_idx = selected_indices[sorted_idx[:capacity]]
                drop_idx = selected_indices[sorted_idx[capacity:]]
                dropped[drop_idx] = True
            else:
                keep_idx = selected_indices

            if keep_idx.numel() == 0:
                continue

            # 专家前向计算
            expert_input = x_flat[keep_idx]
            expert_output = self.experts[expert_id](expert_input)

            # 按门控权重加权
            gate_weights = expert_gate[keep_idx].unsqueeze(-1)
            output[keep_idx] = gate_weights * expert_output

        # 5) 被丢弃的 token 直接复制输入（残差连接会在外部加）
        output[dropped] = x_flat[dropped]

        # 6) 恢复形状
        output = output.view(batch_size, seq_len, d_model)

        # 7) 计算负载均衡损失
        # f_i: 每个专家实际处理的 token 比例
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()
        f = expert_mask.mean(dim=0)
        P = gate_probs.mean(dim=0)
        aux_loss = self.num_experts * torch.sum(f * P)

        return output, aux_loss


class SwitchTransformerLayer(nn.Module):
    """
    Switch Transformer 的完整层：Pre-Norm + Self-Attention + Switch MoE。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_experts: int,
        d_ff: int = None,
        capacity_factor: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.moe = SwitchMoELayer(d_model, num_experts, d_ff, capacity_factor, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        参数:
            x: (batch, seq_len, d_model)
            mask: 可选的 attention mask
        返回:
            output: (batch, seq_len, d_model)
            aux_loss: 标量
        """
        # Self-Attention with Pre-Norm
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)

        # Switch MoE with Pre-Norm
        normed = self.norm2(x)
        moe_out, aux_loss = self.moe(normed)
        x = x + moe_out

        return x, aux_loss


class SwitchTransformer(nn.Module):
    """
    简化的 Switch Transformer 模型，用于自回归语言建模。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        max_len: int = 512,
        n_layers: int = 6,
        num_heads: int = 12,
        num_experts: int = 8,
        d_ff: int = None,
        capacity_factor: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            SwitchTransformerLayer(
                d_model, num_heads, num_experts, d_ff, capacity_factor, dropout
            )
            for _ in range(n_layers)
        ])

        self.norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask  # (seq_len, seq_len)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        causal_mask = self._make_causal_mask(seq_len, input_ids.device)

        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss = layer(x, causal_mask)
            total_aux_loss += aux_loss

        x = self.norm_final(x)
        logits = self.lm_head(x)

        # 平均各层的辅助损失
        total_aux_loss = total_aux_loss / len(self.layers)
        return logits, total_aux_loss


if __name__ == "__main__":
    # 自测：验证 Switch Transformer 的前向传播和辅助损失
    batch_size, seq_len = 2, 16
    vocab_size = 1000
    d_model = 256
    num_experts = 4
    capacity_factor = 1.0

    model = SwitchTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=2,
        num_heads=8,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits, aux_loss = model(input_ids)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    print(f"[Switch Transformer] Logits shape: {logits.shape}")
    print(f"[Switch Transformer] Aux loss: {aux_loss.item():.4f}")

    # 验证参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Switch Transformer] Total parameters: {total_params / 1e6:.2f}M")

    # 验证 dropped tokens 机制：将 capacity_factor 设很小，观察 aux_loss 变化
    model_small_cap = SwitchTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=1,
        num_heads=8,
        num_experts=num_experts,
        capacity_factor=0.25,  # 很小的容量，强制丢弃 token
    )
    _, aux_loss_small = model_small_cap(input_ids)
    print(f"[Switch Transformer] Aux loss (capacity_factor=0.25): {aux_loss_small.item():.4f}")

    print("All Switch Transformer self-tests passed.")
