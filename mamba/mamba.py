"""
Mamba: Selective State Space Model
===================================

Mamba 是 2023-2024 年兴起的新型序列建模架构，旨在挑战 Transformer 的统治地位。
核心创新：引入选择性机制（Selectivity），使 SSM 参数成为输入的函数，从而实现线性时间复杂度的同时保持与 Transformer 相媲美的建模能力。

核心思想：
    传统 SSM（如 S4）的 A、B、C 参数是固定的，无法根据输入内容选择性关注/忽略信息；
    Mamba 通过让这些参数由输入动态生成，实现了类似注意力机制的"选择性"，但计算复杂度仅为 O(n)。

时间复杂度：O(n * d)（线性，n 为序列长度，d 为状态维度）
空间复杂度：O(n * d)（需存储隐状态）
面试频率：极高（2024-2025 大模型面试新热点）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    """SiLU / Swish 激活函数：x * sigmoid(x)，Mamba 中用于门控。"""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SelectiveSSM(nn.Module):
    """
    选择性状态空间模型（Selective State Space Model）。

    核心创新：传统的 S4 模型中，状态矩阵 A、B、C 是静态的；
    Mamba 将这些参数变为输入的函数，实现内容感知的动态选择性。

    状态空间离散化：
        原始连续系统: x'(t) = A x(t) + B u(t)
                     y(t)   = C x(t)
        离散化后（Zero-Order Hold）:
            x_{t} = A_{t} x_{t-1} + B_{t} u_{t}
            y_{t} = C_{t} x_{t-1}
        其中 A_{t}, B_{t}, C_{t} 由输入动态生成。

    Args:
        d_model: 输入维度
        d_state: SSM 状态维度（通常设为 d_model 的倍数或分数）
        d_conv: 局部卷积核大小
        expand: 中间维度扩展因子
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.randn(self.d_inner, d_state)
        A = torch.linalg.solve(
            torch.eye(d_state) - torch.exp(torch.tensor(math.pi / 4)) * A,
            -A,
        )
        self.A_log = nn.Parameter(torch.log(torch.abs(A) + 1e-5))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.act = SiLU()

    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        """
        硬件感知的并行前缀扫描（Parallel Prefix Scan / Recurrent Scan）。

        由于 SSM 的递归特性，朴素实现是 O(n) 串行的。
        Mamba 通过 Scan 分解，利用并行扫描算法实现硬件友好的计算。

        Args:
            x: (batch, seq_len, d_inner) 输入
            dt: (batch, seq_len, d_inner) 动态时间步
            A: (d_inner, d_state) 状态转移矩阵
            B: (batch, seq_len, d_state) 输入映射
            C: (batch, seq_len, d_state) 输出映射
            D: (d_inner,) 跳跃连接

        Returns:
            y: (batch, seq_len, d_inner)
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        delta = F.softplus(dt)

        dA = torch.exp(delta.unsqueeze(-1) * A)
        dB_u = delta.unsqueeze(-1) * x.unsqueeze(-1) * B.unsqueeze(2)

        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(seq_len):
            h = dA[:, i] * h + dB_u[:, i]
            y = torch.einsum("bdn,bn->bd", h, C[:, i])
            ys.append(y)

        y = torch.stack(ys, dim=1)
        y = y + x * D

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_conv = self.conv1d(x_inner.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)[:, :seq_len, :]

        x_ssm = self.act(x_conv)

        x_dbl = self.x_proj(x_ssm)
        dt, B_input, C_input = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        dt = self.dt_proj(dt)

        A = -torch.exp(self.A_log.float())

        y = self._selective_scan(x_ssm, dt, A, B_input, C_input, self.D.float())

        y = y * F.silu(z)

        output = self.out_proj(y)
        return output


class MambaBlock(nn.Module):
    """
    Mamba 块。

    结构：
        输入 -> 线性投影 -> 因果卷积 -> SiLU -> 选择性 SSM -> 残差连接
                                                        |
        输入 -------------------------------------------+
                                                        |
    输出 = 输入 + 残差

    残差设计：类似 ResNet，允许梯度直接流过主路径，提升深层模型训练稳定性。
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(x)
        x = self.norm(x)
        return x


class Mamba(nn.Module):
    """
    Mamba 语言模型主干。

    由多层 MambaBlock 堆叠而成，输出隐状态（而非 logits）。
    用于下游任务时可接语言模型头（lm_head）。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                MambaBlock(d_model, d_state, d_conv, expand)
                for _ in range(n_layers)
            ]
        )
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits


if __name__ == "__main__":
    VOCAB_SIZE = 1000
    D_MODEL = 64
    N_LAYERS = 2
    D_STATE = 8
    D_CONV = 4
    EXPAND = 2
    BATCH_SIZE = 2
    SEQ_LEN = 16

    model = Mamba(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        d_state=D_STATE,
        d_conv=D_CONV,
        expand=EXPAND,
    )

    input_ids = torch.randint(1, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    logits = model(input_ids)

    assert logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    print(f"[Mamba] Logits shape: {logits.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Mamba] Total parameters: {total_params / 1e6:.2f}M")

    print("Self-test passed.")
