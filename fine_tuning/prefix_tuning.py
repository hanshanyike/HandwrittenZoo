"""
Prefix Tuning
=============
在 Transformer 的每一层前面添加可学习的连续前缀向量（Prefix Embeddings），
冻结预训练模型全部参数，只训练这些前缀，实现参数高效微调。

核心思想：
    - 不修改模型权重，也不在输入层加 prompt，而是在 Transformer 每层的 K/V 前添加
      可学习的前缀向量。
    - 这些前缀作为额外的 key 和 value 参与注意力计算，影响模型的注意力分布。
    - 通过 MLP 重参数化（reparameterization）稳定训练，训练完成后可丢弃 MLP。

时间复杂度：O(batch * seq_len * prefix_len * d_model)
空间复杂度：O(num_layers * prefix_len * d_model * 2)  （每层 K 和 V 各一组前缀）

面试频率：高
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrefixEmbedding(nn.Module):
    """
    前缀嵌入层：存储可学习的前缀向量。

    参数:
        num_layers: Transformer 层数
        num_heads: 注意力头数
        head_dim: 每个头的维度
        prefix_len: 前缀长度（每个层、每个头）
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        prefix_len: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.prefix_len = prefix_len

        # 每层有 K 和 V 两组前缀
        # 形状: (num_layers, 2, num_heads, prefix_len, head_dim)
        # 其中 dim=1 的 2 分别对应 key 和 value 的前缀
        self.prefix_tokens = nn.Parameter(
            torch.randn(num_layers, 2, num_heads, prefix_len, head_dim)
        )

    def forward(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定层的前缀 K 和 V。

        返回:
            prefix_k: (num_heads, prefix_len, head_dim)
            prefix_v: (num_heads, prefix_len, head_dim)
        """
        prefix_k = self.prefix_tokens[layer_idx, 0]  # (num_heads, prefix_len, head_dim)
        prefix_v = self.prefix_tokens[layer_idx, 1]  # (num_heads, prefix_len, head_dim)
        return prefix_k, prefix_v


class PrefixTuningMLP(nn.Module):
    """
    前缀向量的 MLP 重参数化层。

    原始 Prefix Tuning 论文发现，直接优化前缀向量训练不稳定、性能较差。
    通过一个小的 MLP 生成前缀向量，可以显著提升训练稳定性。
    训练完成后，可将 MLP 输出缓存，丢弃 MLP 以节省显存。
    """

    def __init__(
        self,
        prefix_len: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        mlp_dim: int = 512,
    ):
        super().__init__()
        self.prefix_len = prefix_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim

        # 可学习的输入嵌入（作为 MLP 的输入）
        self.prefix_embed = nn.Parameter(torch.randn(prefix_len, mlp_dim))

        # MLP: 将低维嵌入映射到前缀向量
        self.mlp = nn.Sequential(
            nn.Linear(mlp_dim, mlp_dim),
            nn.Tanh(),
            nn.Linear(mlp_dim, num_layers * 2 * num_heads * head_dim),
        )

    def forward(self) -> torch.Tensor:
        """
        生成所有层的前缀向量。

        返回:
            prefix: (num_layers, 2, num_heads, prefix_len, head_dim)
        """
        # prefix_embed: (prefix_len, mlp_dim)
        out = self.mlp(self.prefix_embed)  # (prefix_len, num_layers * 2 * num_heads * head_dim)
        out = out.view(
            self.prefix_len,
            self.num_layers,
            2,
            self.num_heads,
            self.head_dim,
        )
        # 调整维度顺序为 (num_layers, 2, num_heads, prefix_len, head_dim)
        out = out.permute(1, 2, 3, 0, 4)
        return out


class PrefixTuningAttention(nn.Module):
    """
    带 Prefix Tuning 的自注意力层。

    与标准自注意力的区别：在 K 和 V 前面拼接可学习的前缀向量。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        prefix_len: int,
        layer_idx: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.prefix_len = prefix_len
        self.layer_idx = layer_idx

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        参数:
            x: (batch, seq_len, d_model)
            prefix_k: (num_heads, prefix_len, head_dim)
            prefix_v: (num_heads, prefix_len, head_dim)
            mask: 可选的 attention mask
        返回:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 标准 Q/K/V 投影
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: (batch, num_heads, seq_len, head_dim)

        # 在 K 和 V 前面拼接前缀
        # prefix_k/v: (num_heads, prefix_len, head_dim) -> 扩展 batch 维度
        prefix_k = prefix_k.unsqueeze(0).expand(batch_size, -1, -1, -1)
        prefix_v = prefix_v.unsqueeze(0).expand(batch_size, -1, -1, -1)

        K = torch.cat([prefix_k, K], dim=2)  # (batch, num_heads, prefix_len + seq_len, head_dim)
        V = torch.cat([prefix_v, V], dim=2)

        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: (batch, num_heads, seq_len, prefix_len + seq_len)

        if mask is not None:
            # mask 需要扩展以覆盖前缀部分（前缀部分不 mask）
            prefix_mask = torch.ones(batch_size, 1, seq_len, self.prefix_len, device=x.device)
            full_mask = torch.cat([prefix_mask, mask], dim=-1)
            scores = scores.masked_fill(full_mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)  # (batch, num_heads, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(context)
        return output


class PrefixTuningTransformerLayer(nn.Module):
    """
    带 Prefix Tuning 的 Transformer 层。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        prefix_len: int,
        layer_idx: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = PrefixTuningAttention(d_model, num_heads, prefix_len, layer_idx, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), prefix_k, prefix_v, mask)
        x = x + self.ffn(self.norm2(x))
        return x


class PrefixTuningTransformer(nn.Module):
    """
    带 Prefix Tuning 的完整 Transformer 模型。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        max_len: int = 512,
        n_layers: int = 6,
        num_heads: int = 12,
        d_ff: int = 2048,
        prefix_len: int = 50,
        use_reparam: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.prefix_len = prefix_len
        self.use_reparam = use_reparam

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            PrefixTuningTransformerLayer(d_model, num_heads, d_ff, prefix_len, i, dropout)
            for i in range(n_layers)
        ])

        self.norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 前缀参数
        if use_reparam:
            self.prefix_generator = PrefixTuningMLP(
                prefix_len=prefix_len,
                num_layers=n_layers,
                num_heads=num_heads,
                head_dim=d_model // num_heads,
                mlp_dim=512,
            )
        else:
            self.prefix_embedding = PrefixEmbedding(
                num_layers=n_layers,
                num_heads=num_heads,
                head_dim=d_model // num_heads,
                prefix_len=prefix_len,
            )

        # 冻结预训练参数（除前缀外）
        self._freeze_pretrained()
        self.apply(self._init_weights)

    def _freeze_pretrained(self):
        """冻结所有非前缀参数。"""
        for name, param in self.named_parameters():
            if "prefix" not in name:
                param.requires_grad = False

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        return mask  # (1, 1, seq_len, seq_len)

    def get_prefixes(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """获取每层的前缀 K 和 V。"""
        if self.use_reparam:
            all_prefixes = self.prefix_generator()  # (n_layers, 2, num_heads, prefix_len, head_dim)
            prefixes = []
            for i in range(self.n_layers):
                prefixes.append((all_prefixes[i, 0], all_prefixes[i, 1]))
            return prefixes
        else:
            return [self.prefix_embedding(i) for i in range(self.n_layers)]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        causal_mask = self._make_causal_mask(seq_len, input_ids.device)
        prefixes = self.get_prefixes()

        for i, layer in enumerate(self.layers):
            prefix_k, prefix_v = prefixes[i]
            x = layer(x, prefix_k, prefix_v, causal_mask)

        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits


if __name__ == "__main__":
    # 自测：验证 Prefix Tuning 的参数效率和前向传播
    vocab_size = 1000
    d_model = 256
    n_layers = 4
    num_heads = 8
    prefix_len = 20

    # 1) 使用重参数化版本
    model_reparam = PrefixTuningTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        num_heads=num_heads,
        prefix_len=prefix_len,
        use_reparam=True,
    )

    total_params = sum(p.numel() for p in model_reparam.parameters())
    trainable_params = sum(p.numel() for p in model_reparam.parameters() if p.requires_grad)
    print(f"[Prefix Tuning] 总参数量: {total_params / 1e6:.2f}M")
    print(f"[Prefix Tuning] 可训练参数量: {trainable_params / 1e3:.1f}K")
    print(f"[Prefix Tuning] 参数效率: {trainable_params / total_params * 100:.4f}%")

    # 2) 前向传播测试
    batch_size, seq_len = 2, 16
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    logits = model_reparam(x)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    print(f"[Prefix Tuning] 输出形状: {logits.shape}")

    # 3) 验证梯度只流向前缀参数
    loss = logits.mean()
    loss.backward()
    for name, param in model_reparam.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} 应该有梯度"
        else:
            assert param.grad is None or param.grad.abs().sum().item() == 0, f"{name} 不应有梯度"
    print("[Prefix Tuning] 梯度检查通过")

    # 4) 不使用重参数化版本对比
    model_direct = PrefixTuningTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        num_heads=num_heads,
        prefix_len=prefix_len,
        use_reparam=False,
    )
    trainable_direct = sum(p.numel() for p in model_direct.parameters() if p.requires_grad)
    print(f"[Prefix Tuning] 直接优化前缀的可训练参数量: {trainable_direct / 1e3:.1f}K")

    # 5) 验证前缀长度对显存的影响
    for pl in [10, 50, 100]:
        m = PrefixTuningTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            num_heads=num_heads,
            prefix_len=pl,
            use_reparam=True,
        )
        tp = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"[Prefix Tuning] prefix_len={pl} -> 可训练参数: {tp / 1e3:.1f}K")

    print("All Prefix Tuning self-tests passed.")
