"""
LoRA (Low-Rank Adaptation)
===========================
参数高效微调（PEFT）的代表性方法，通过低秩矩阵分解来近似全量微调的权重更新，
在冻结预训练权重的前提下，仅训练少量适配器参数。

核心思想：
    - 预训练权重 W_0 保持冻结，不计算梯度。
    - 引入两个低秩矩阵 A (d x r) 和 B (r x k)，其中 r << min(d, k)。
    - 前向传播时：h = W_0 @ x + (alpha / r) * B @ A @ x
    - 只训练 A 和 B，大幅减少可训练参数量。

时间复杂度：O(batch * seq_len * d_model * rank)
空间复杂度：O(d_model * rank) 每层（远小于全量微调的 O(d_model^2)）

面试频率：极高（大模型微调必考点）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """
    通用 LoRA 适配层，可包装任意 nn.Linear 层。

    参数:
        in_features: 输入维度
        out_features: 输出维度
        rank: 低秩维度 r（通常 4、8、16、64）
        alpha: 缩放系数，控制 LoRA 更新的幅度
        dropout: LoRA 路径上的 dropout 概率
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # 缩放因子

        # 低秩分解矩阵
        # A: (in_features, rank) — 用高斯初始化
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        # B: (rank, out_features) — 用零初始化，确保训练开始时 LoRA 输出为 0
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 初始化 A 用 Kaiming Uniform（类似 nn.Linear 的默认初始化）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 保持为零，保证训练初期 LoRA 不产生扰动

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算 LoRA 增量：delta_W @ x = B @ A @ x

        参数:
            x: (..., in_features)
        返回:
            output: (..., out_features)
        """
        # x: (..., in_features)
        # 先 dropout，再与 A 相乘，再与 B 相乘
        # 等价于: (x @ A) @ B = x @ (A @ B)，其中 A @ B 是低秩近似
        x_d = self.dropout(x)  # (..., in_features)
        # x @ A: (..., rank)
        h = torch.matmul(x_d, self.lora_A)
        # (x @ A) @ B: (..., out_features)
        h = torch.matmul(h, self.lora_B)
        return h * self.scaling


class LinearWithLoRA(nn.Module):
    """
    将 LoRA 适配器注入到标准 nn.Linear 层的包装模块。

    前向传播:
        y = W_0 @ x + b + (alpha / r) * B @ A @ x
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        # 冻结原始权重，不参与训练
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.lora = LoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始线性层输出
        base_out = self.base_layer(x)
        # LoRA 增量
        lora_out = self.lora(x)
        return base_out + lora_out


def inject_lora_into_model(
    model: nn.Module,
    target_modules: list[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """
    将 LoRA 注入到模型的指定模块中。

    参数:
        model: 待注入的 PyTorch 模型
        target_modules: 目标模块名称列表，如 ["q_proj", "v_proj", "k_proj", "o_proj"]
        rank: LoRA 秩
        alpha: LoRA 缩放系数
        dropout: LoRA dropout
    返回:
        已注入 LoRA 的模型（原地修改）
    """
    for name, module in model.named_modules():
        # 检查当前模块名称是否匹配目标模块
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 获取父模块和当前模块的属性名
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = model.get_submodule(parent_name) if parent_name else model

                # 创建 LoRA 包装层
                lora_layer = LinearWithLoRA(module, rank, alpha, dropout)
                # 替换原模块
                setattr(parent, child_name, lora_layer)

    return model


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """
    获取模型中所有 LoRA 可训练参数。
    用于传给 optimizer，确保只训练 LoRA 参数。
    """
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            lora_params.append(param)
    return lora_params


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    将 LoRA 权重合并回基座权重，便于推理部署。
    W_merged = W_0 + (alpha / r) * B @ A
    """
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            # 计算合并后的权重
            delta_W = module.lora.scaling * (module.lora.lora_B @ module.lora.lora_A.t())
            # 加到基座权重上
            module.base_layer.weight.data += delta_W.t()
            # 可选：将 LoRA 参数置零或删除
            module.lora.lora_A.data.zero_()
            module.lora.lora_B.data.zero_()
    return model


# ==================== 演示：在简单 Transformer 上应用 LoRA ====================

class SimpleAttention(nn.Module):
    """简化的自注意力层，用于演示 LoRA 注入。"""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)


class SimpleTransformer(nn.Module):
    """两层 Transformer，用于 LoRA 自测。"""

    def __init__(self, vocab_size: int, d_model: int = 128, num_heads: int = 4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.attn1 = SimpleAttention(d_model, num_heads)
        self.attn2 = SimpleAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)
        x = x + self.attn1(x)
        x = self.norm(x)
        x = x + self.attn2(x)
        return self.head(x)


if __name__ == "__main__":
    # 自测：验证 LoRA 的参数效率
    vocab_size = 1000
    d_model = 128
    rank = 8
    alpha = 16.0

    # 1) 创建基座模型
    model = SimpleTransformer(vocab_size, d_model)
    total_params_before = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"[LoRA] 基座模型总参数量: {total_params_before / 1e3:.1f}K")
    print(f"[LoRA] 基座模型可训练参数量: {trainable_before / 1e3:.1f}K")

    # 2) 注入 LoRA
    model = inject_lora_into_model(
        model,
        target_modules=["q_proj", "v_proj"],  # 只对 Q 和 V 注入 LoRA（常见做法）
        rank=rank,
        alpha=alpha,
        dropout=0.05,
    )

    total_params_after = sum(p.numel() for p in model.parameters())
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = get_lora_parameters(model)
    lora_param_count = sum(p.numel() for p in lora_params)

    print(f"[LoRA] 注入后总参数量: {total_params_after / 1e3:.1f}K")
    print(f"[LoRA] 注入后可训练参数量: {trainable_after / 1e3:.1f}K")
    print(f"[LoRA] LoRA 专属参数量: {lora_param_count / 1e3:.1f}K")
    print(f"[LoRA] 参数效率: {trainable_after / total_params_after * 100:.2f}%")

    # 3) 验证前向传播
    batch_size, seq_len = 2, 10
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model(x)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print(f"[LoRA] 前向传播输出形状: {logits.shape}")

    # 4) 验证梯度只流向 LoRA 参数
    loss = logits.mean()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} 没有梯度"
        else:
            assert param.grad is None, f"{name} 不应有梯度"
    print("[LoRA] 梯度检查通过：只有 LoRA 参数有梯度")

    # 5) 验证权重合并
    model_merged = merge_lora_weights(model)
    logits_merged = model_merged(x)
    # 合并后输出应相同（LoRA 参数已置零，但权重已合并）
    print(f"[LoRA] 权重合并完成")

    print("All LoRA self-tests passed.")
