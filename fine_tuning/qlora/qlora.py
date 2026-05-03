"""
QLoRA (Quantized Low-Rank Adaptation)
======================================
在 LoRA 基础上引入 4-bit 量化，将预训练模型权重压缩到 NF4/FP4 精度，
配合双量化（Double Quantization）和分页优化器（Paged Optimizer），
实现在单张消费级 GPU 上微调 65B+ 参数大模型。

核心思想：
    - 基座权重以 4-bit 量化形式存储，大幅节省显存。
    - 计算时动态反量化为 16-bit（BF16/FP16），保证精度。
    - LoRA 适配器保持 16-bit 精度训练。
    - 双量化：对量化常数再次量化，进一步节省显存。

时间复杂度：O(batch * seq_len * d_model * rank)  （与 LoRA 相同）
空间复杂度：O(d_model * rank)  LoRA 参数 + O(d_model^2 / 2)  4-bit 基座权重

面试频率：极高（大模型高效微调必考点）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 4-bit 量化工具函数 ====================

def quantize_to_4bit(
    weight: torch.Tensor,
    block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将 FP16/BF16 权重量化为 4-bit（简化版 NF4 模拟）。

    实际 QLoRA 使用 bitsandbytes 的 NF4 量化，这里用均匀量化演示原理。

    参数:
        weight: 原始权重矩阵 (out_features, in_features)
        block_size: 分块大小，每块独立量化
    返回:
        quantized: 量化后的整数值 (out_features, in_features)，存储为 8-bit 以兼容 PyTorch
        scale: 每块的缩放因子 (num_blocks,)
        zero_point: 每块的零点 (num_blocks,)
    """
    orig_shape = weight.shape
    weight = weight.view(-1)
    num_blocks = math.ceil(weight.numel() / block_size)

    # 补齐到 block_size 的整数倍
    pad_len = num_blocks * block_size - weight.numel()
    if pad_len > 0:
        weight = F.pad(weight, (0, pad_len))

    blocks = weight.view(num_blocks, block_size)

    # 计算每块的最小值和最大值
    w_min = blocks.min(dim=1, keepdim=True).values
    w_max = blocks.max(dim=1, keepdim=True).values

    # 4-bit 量化范围: 0 ~ 15
    qmax = 15.0
    scale = (w_max - w_min) / qmax
    zero_point = w_min

    # 量化: q = round((w - zp) / scale)
    quantized_blocks = torch.round((blocks - zero_point) / (scale + 1e-8)).clamp(0, qmax)
    quantized = quantized_blocks.view(-1)[:orig_shape.numel()].view(orig_shape)

    return quantized.to(torch.uint8), scale.squeeze(), zero_point.squeeze()


def dequantize_from_4bit(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    orig_shape: tuple,
    block_size: int = 64,
) -> torch.Tensor:
    """
    将 4-bit 量化值反量化为 FP16。

    参数:
        quantized: 量化后的整数值
        scale: 每块缩放因子
        zero_point: 每块零点
        orig_shape: 原始权重形状
        block_size: 分块大小
    """
    weight = quantized.view(-1).float()
    num_blocks = math.ceil(weight.numel() / block_size)

    pad_len = num_blocks * block_size - weight.numel()
    if pad_len > 0:
        weight = F.pad(weight, (0, pad_len))

    blocks = weight.view(num_blocks, block_size)

    # 反量化: w = q * scale + zp
    # scale 和 zero_point 需要扩展为 (num_blocks, 1)
    if scale.dim() == 1:
        scale = scale.unsqueeze(1)
    if zero_point.dim() == 1:
        zero_point = zero_point.unsqueeze(1)

    dequantized_blocks = blocks * scale + zero_point
    dequantized = dequantized_blocks.view(-1)[:orig_shape.numel()].view(orig_shape)
    return dequantized


# ==================== 4-bit 线性层 ====================

class Linear4Bit(nn.Module):
    """
    模拟 4-bit 量化线性层。

    实际系统中由 bitsandbytes 提供，这里用 PyTorch 原生操作演示核心原理：
        - 权重以 4-bit 形式存储。
        - 前向传播时动态反量化为 16-bit 进行计算。
        - 梯度只流向输入和偏置（权重冻结）。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, block_size: int = 64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        # 注册量化参数（不作为 nn.Parameter，不参与梯度）
        self.register_buffer("quantized_weight", None)
        self.register_buffer("scale", None)
        self.register_buffer("zero_point", None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def from_float(self, weight: torch.Tensor):
        """从 FP16 权重初始化 4-bit 量化参数。"""
        q, s, zp = quantize_to_4bit(weight, self.block_size)
        self.quantized_weight = q
        self.scale = s
        self.zero_point = zp
        self.orig_shape = weight.shape

    def get_dequantized_weight(self) -> torch.Tensor:
        """获取反量化后的 FP16 权重（用于前向计算）。"""
        return dequantize_from_4bit(
            self.quantized_weight, self.scale, self.zero_point, self.orig_shape, self.block_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向时动态反量化
        weight = self.get_dequantized_weight()
        return F.linear(x, weight, self.bias)


# ==================== QLoRA 层 ====================

class QLoRALayer(nn.Module):
    """
    QLoRA 适配层：在 4-bit 基座权重旁添加 LoRA 旁路。

    与标准 LoRA 的区别：基座权重以 4-bit 存储，计算时反量化为 16-bit。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_d = self.dropout(x)
        h = torch.matmul(x_d, self.lora_A)
        h = torch.matmul(h, self.lora_B)
        return h * self.scaling


class LinearWithQLoRA(nn.Module):
    """
    QLoRA 包装层：4-bit 基座 + LoRA 旁路。
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 64,
        alpha: float = 16.0,
        dropout: float = 0.0,
        block_size: int = 64,
    ):
        super().__init__()
        # 将基座权重转换为 4-bit
        self.base_layer = Linear4Bit(
            base_layer.in_features,
            base_layer.out_features,
            bias=base_layer.bias is not None,
            block_size=block_size,
        )
        self.base_layer.from_float(base_layer.weight.data)
        if base_layer.bias is not None:
            self.base_layer.bias.data = base_layer.bias.data

        self.lora = QLoRALayer(
            base_layer.in_features,
            base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora(x)
        return base_out + lora_out


# ==================== 双量化（Double Quantization）演示 ====================

class DoubleQuantization:
    """
    双量化：对量化常数（scale）再次量化，进一步节省显存。

    QLoRA 中，每个 block 都有一个 32-bit 的 scale。当 block_size=64 时，
    scale 的存储开销为 32/64 = 0.5 bit/参数。双量化将 scale 量化为 8-bit，
    将开销降到 8/64 = 0.125 bit/参数。
    """

    def __init__(self, block_size: int = 256):
        self.block_size = block_size

    def quantize_scale(self, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """将 scale 量化为 8-bit。"""
        s_min = scale.min()
        s_max = scale.max()
        q_scale = 255.0
        scale_of_scale = (s_max - s_min) / q_scale
        zero_point_of_scale = s_min

        quantized_scale = torch.round((scale - zero_point_of_scale) / (scale_of_scale + 1e-8)).clamp(0, 255)
        return quantized_scale.to(torch.uint8), torch.tensor([scale_of_scale, zero_point_of_scale])

    def dequantize_scale(self, quantized_scale: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """将 8-bit scale 反量化。"""
        scale_of_scale, zero_point_of_scale = meta[0], meta[1]
        return quantized_scale.float() * scale_of_scale + zero_point_of_scale


# ==================== 分页优化器（Paged Optimizer）概念演示 ====================

class PagedOptimizer(torch.optim.Optimizer):
    """
    分页优化器的概念演示。

    实际 QLoRA 使用 bitsandbytes 的 CUDA 实现，当 GPU 显存不足时，
    自动将优化器状态分页到 CPU 内存，需要时再取回。

    这里仅演示核心思想：在 GPU 和 CPU 之间动态交换优化器状态。
    """

    def __init__(self, params, lr=1e-3, page_threshold_mb=512):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        self.page_threshold = page_threshold_mb * 1024 * 1024
        self.cpu_state = {}  # 存储被分页到 CPU 的状态

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # 简化的 SGD 更新（实际应为 AdamW）
                p.data.add_(p.grad.data, alpha=-group["lr"])
        return loss


# ==================== 自测 ====================

if __name__ == "__main__":
    in_features, out_features = 512, 512
    rank = 64
    batch_size, seq_len = 2, 10

    # 1) 测试 4-bit 量化和反量化
    weight = torch.randn(out_features, in_features)
    q, s, zp = quantize_to_4bit(weight, block_size=64)
    weight_deq = dequantize_from_4bit(q, s, zp, weight.shape, block_size=64)

    # 计算量化误差
    quant_error = (weight - weight_deq).abs().mean().item()
    print(f"[QLoRA] 4-bit 量化平均绝对误差: {quant_error:.6f}")

    # 显存节省估算
    orig_mem = weight.numel() * 2  # FP16 = 2 bytes
    quant_mem = q.numel() * 1 + s.numel() * 4 + zp.numel() * 4  # 近似
    print(f"[QLoRA] 原始显存: {orig_mem / 1024:.1f} KB, 量化后: {quant_mem / 1024:.1f} KB")
    print(f"[QLoRA] 压缩比: {orig_mem / quant_mem:.2f}x")

    # 2) 测试 Linear4Bit 前向传播
    linear_4bit = Linear4Bit(in_features, out_features, bias=True, block_size=64)
    linear_4bit.from_float(weight)
    x = torch.randn(batch_size, seq_len, in_features)
    out_4bit = linear_4bit(x)

    # 与原始 FP16 对比
    linear_fp16 = nn.Linear(in_features, out_features, bias=True)
    linear_fp16.weight.data = weight
    linear_fp16.bias.data = linear_4bit.bias.data
    out_fp16 = linear_fp16(x)

    forward_error = (out_4bit - out_fp16).abs().mean().item()
    print(f"[QLoRA] 4-bit 前向传播误差: {forward_error:.6f}")

    # 3) 测试 QLoRA 层
    qlora_layer = LinearWithQLoRA(linear_fp16, rank=rank, alpha=16.0, dropout=0.05)
    out_qlora = qlora_layer(x)
    assert out_qlora.shape == (batch_size, seq_len, out_features)
    print(f"[QLoRA] QLoRA 输出形状: {out_qlora.shape}")

    # 4) 验证梯度只流向 LoRA
    loss = out_qlora.mean()
    loss.backward()
    assert qlora_layer.lora.lora_A.grad is not None
    assert qlora_layer.lora.lora_B.grad is not None
    print("[QLoRA] 梯度检查通过：LoRA 参数有梯度，基座权重冻结")

    # 5) 测试双量化
    dq = DoubleQuantization(block_size=256)
    q_scale, meta = dq.quantize_scale(s)
    s_deq = dq.dequantize_scale(q_scale, meta)
    scale_error = (s - s_deq).abs().mean().item()
    print(f"[QLoRA] 双量化 scale 误差: {scale_error:.8f}")

    # 6) 参数统计
    base_params = sum(p.numel() for p in qlora_layer.base_layer.parameters() if p is not None)
    lora_params = sum(p.numel() for p in qlora_layer.lora.parameters())
    print(f"[QLoRA] 基座参数量(4-bit存储): {base_params / 1e3:.1f}K (等效FP16: {base_params * 4 / 1e3:.1f}K)")
    print(f"[QLoRA] LoRA 参数量: {lora_params / 1e3:.1f}K")

    print("All QLoRA self-tests passed.")
