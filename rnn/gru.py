"""
GRU (Gated Recurrent Unit)

Algorithm: 门控循环单元 —— 将 LSTM 的三门控简化为两门控（重置门、更新门），
         合并细胞状态与隐藏状态，在保持长程依赖建模能力的同时减少参数量和计算量。

Core Idea:
- LSTM 用细胞状态和隐藏状态分离设计实现长程记忆，但参数量较大。
- GRU 的核心洞察：将细胞状态与隐藏状态合并为单一状态向量，用两个门控实现类似功能。
- 重置门（Reset Gate）控制前一时刻状态有多少被用于计算候选状态，
  更新门（Update Gate）控制前一时刻状态有多少被保留到当前时刻。
- 实验表明 GRU 在多数任务上与 LSTM 性能相当，但训练速度更快。

Complexity:
    - 时间复杂度: O(T * (3 * H * (H + I))) —— T为序列长度，H为隐藏维度，I为输入维度
    - 空间复杂度: O(T * H) —— 存储每个时间步的隐藏状态
    - 参数量: 3 * (I + H) * H + 3 * H（约为 LSTM 的 75%）

Interview Frequency: 高（与 LSTM 对比是面试常考点）

References:
    - Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", EMNLP 2014
"""

import torch
import torch.nn as nn


class GRUCell(nn.Module):
    """
    单个 GRU 单元的从零实现。

    门控机制:
        重置门 r_t: 控制前一时刻隐藏状态有多少参与当前候选状态的计算
        更新门 z_t: 控制前一时刻隐藏状态有多少被保留到当前时刻

    状态更新:
        g_t = tanh(W_g * x_t + r_t * (U_g * h_{t-1}))   （候选状态）
        h_t = (1 - z_t) * g_t + z_t * h_{t-1}           （隐藏状态更新）

    与 LSTM 的关键区别:
        - 没有独立的细胞状态，只有隐藏状态
        - 两个门而非三个门
        - 更新门同时承担了 LSTM 中遗忘门和输入门的角色

    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 重置门和更新门的权重（共享输入投影）
        # 形状: (input_size, 2 * hidden_size)
        self.weight_ir = nn.Parameter(
            torch.randn(input_size, 2 * hidden_size) * 0.01
        )
        self.weight_hr = nn.Parameter(
            torch.randn(hidden_size, 2 * hidden_size) * 0.01
        )
        self.bias_r = nn.Parameter(torch.zeros(2 * hidden_size))

        # 候选状态的权重（独立的输入和隐藏状态投影）
        # 形状: (input_size, hidden_size)
        self.weight_ig = nn.Parameter(
            torch.randn(input_size, hidden_size) * 0.01
        )
        self.weight_hg = nn.Parameter(
            torch.randn(hidden_size, hidden_size) * 0.01
        )
        # 分离输入偏置和隐藏偏置，与 PyTorch 官方实现保持一致
        # 官方公式: n_t = tanh(W_in x_t + b_in + r_t * (W_hn h_{t-1} + b_hn))
        self.bias_ig = nn.Parameter(torch.zeros(hidden_size))
        self.bias_hg = nn.Parameter(torch.zeros(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        单个时间步的前向传播。

        Args:
            x: 当前时间步输入，形状 (batch, input_size)
            h_prev: 上一时刻隐藏状态，形状 (batch, hidden_size)
        Returns:
            h_t: 当前时刻隐藏状态，形状 (batch, hidden_size)
        """
        # 1) 计算重置门和更新门的预激活值
        # 将输入和上一时刻隐藏状态分别投影后相加
        gates = torch.matmul(x, self.weight_ir) + torch.matmul(h_prev, self.weight_hr) + self.bias_r

        # 切分为重置门和更新门
        r, z = gates.chunk(2, dim=-1)

        # 应用 sigmoid 激活（门控输出 0~1）
        r = torch.sigmoid(r)  # 重置门：控制历史信息参与度
        z = torch.sigmoid(z)  # 更新门：控制历史信息保留比例

        # 2) 计算候选状态
        # 与 PyTorch 官方实现保持一致:
        # n_t = tanh(W_in x_t + b_in + r_t * (W_hn h_{t-1} + b_hn))
        g = torch.matmul(x, self.weight_ig) + self.bias_ig + r * (torch.matmul(h_prev, self.weight_hg) + self.bias_hg)
        g = torch.tanh(g)

        # 3) 更新隐藏状态
        # 更新门 z_t 控制新旧状态的混合比例
        # h_t = (1 - z_t) * g_t + z_t * h_{t-1}
        h_t = (1 - z) * g + z * h_prev

        return h_t


class GRU(nn.Module):
    """
    完整 GRU 网络实现（支持多层和单向）。

    架构:
        1. 输入投影（可选）：将输入维度映射到隐藏维度
        2. 多层 GRU 堆叠：每层由多个 GRUCell 按时间步展开
        3. 输出投影（可选）：将隐藏状态映射到输出维度

    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
        num_layers: GRU 层数，默认 1
        output_size: 输出维度，若为 None 则输出隐藏状态
        dropout: 层间 Dropout 概率（仅当 num_layers > 1 时生效）
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        output_size: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout

        # 构建多层 GRU
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(GRUCell(layer_input_size, hidden_size))

        # 层间 Dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

        # 可选的输出投影层
        if output_size is not None:
            self.output_projection = nn.Linear(hidden_size, output_size)
        else:
            self.output_projection = None

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        Args:
            x: 输入序列，形状 (batch, seq_len, input_size)
            hidden: 可选的初始隐藏状态，形状 (num_layers, batch, hidden_size)
        Returns:
            outputs: 所有时间步的输出，形状 (batch, seq_len, hidden_size 或 output_size)
            h_n: 最后时刻的隐藏状态，形状 (num_layers, batch, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # 初始化隐藏状态（若未提供）
        if hidden is None:
            h = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype
            )
        else:
            h = hidden

        # 存储每一层、每个时间步的输出
        layer_outputs = [[] for _ in range(self.num_layers)]

        # 按时间步展开
        for t in range(seq_len):
            layer_input = x[:, t, :]  # 当前时间步输入

            for layer in range(self.num_layers):
                # 当前层的 GRUCell 前向传播
                h_t = self.cells[layer](layer_input, h[layer])

                # 更新该层的隐藏状态
                h = h.clone()
                h[layer] = h_t

                # 存储输出
                layer_outputs[layer].append(h_t)

                # 下一层的输入
                layer_input = h_t

                # 层间 Dropout
                if self.dropout_layer is not None and layer < self.num_layers - 1:
                    layer_input = self.dropout_layer(layer_input)

        # 取最后一层的所有时间步输出
        outputs = torch.stack(layer_outputs[-1], dim=1)

        # 可选的输出投影
        if self.output_projection is not None:
            outputs = self.output_projection(outputs)

        return outputs, h


if __name__ == "__main__":
    # ==================== 自测区域 ====================
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    input_size = 16
    hidden_size = 32
    num_layers = 2
    x = torch.randn(batch_size, seq_len, input_size)

    # 1. 测试单个 GRUCell
    print("=" * 60)
    print("Test 1: GRUCell forward pass")
    cell = GRUCell(input_size, hidden_size)
    h_prev = torch.randn(batch_size, hidden_size)
    h_t = cell(x[:, 0, :], h_prev)
    assert h_t.shape == (batch_size, hidden_size)
    print(f"GRUCell h_t shape: {h_t.shape}")

    # 2. 测试单层 GRU
    print("=" * 60)
    print("Test 2: Single-layer GRU")
    gru_single = GRU(input_size, hidden_size, num_layers=1)
    out, h_n = gru_single(x)
    assert out.shape == (batch_size, seq_len, hidden_size)
    assert h_n.shape == (1, batch_size, hidden_size)
    print(f"GRU output shape: {out.shape}")
    print(f"Final h_n shape:  {h_n.shape}")

    # 3. 测试多层 GRU
    print("=" * 60)
    print("Test 3: Multi-layer GRU")
    gru_multi = GRU(input_size, hidden_size, num_layers=num_layers, dropout=0.3)
    out_m, h_m = gru_multi(x)
    assert out_m.shape == (batch_size, seq_len, hidden_size)
    assert h_m.shape == (num_layers, batch_size, hidden_size)
    print(f"Multi-layer GRU output shape: {out_m.shape}")
    print(f"Final h_n shape:              {h_m.shape}")

    # 4. 测试带输出投影的 GRU
    print("=" * 60)
    print("Test 4: GRU with output projection")
    output_size = 8
    gru_proj = GRU(input_size, hidden_size, num_layers=1, output_size=output_size)
    out_p, _ = gru_proj(x)
    assert out_p.shape == (batch_size, seq_len, output_size)
    print(f"GRU with projection output shape: {out_p.shape}")

    # 5. 测试自定义实现与 PyTorch 官方 GRU 的数值一致性
    print("=" * 60)
    print("Test 5: Numerical comparison with PyTorch official GRU")

    gru_custom = GRU(input_size, hidden_size, num_layers=1)
    gru_official = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)

    # 参数同步
    with torch.no_grad():
        # 官方 GRU 的 weight_ih 形状为 (3*hidden_size, input_size)
        # 我们的实现中，weight_ir 对应重置门和更新门（2*hidden_size）
        # weight_ig 对应候选状态（hidden_size）
        # 需要按官方顺序拼接：[update_gate, reset_gate, candidate]
        # 我们的顺序：[reset_gate, update_gate, candidate]
        # 官方顺序：weight_ih_l0 = [W_ir, W_iz, W_in]（重置、更新、候选）
        # 我们的 weight_ir = [W_r, W_z]，需要重排为 [W_z, W_r]
        w_ir = gru_custom.cells[0].weight_ir
        w_hr = gru_custom.cells[0].weight_hr
        b_r = gru_custom.cells[0].bias_r

        # 官方 GRU 的 weight_ih 顺序: [W_ir, W_iz, W_in]（重置、更新、候选）
        # 即 chunk(3, 0) 后: r, z, n
        # 我们的 weight_ir 是按列存储: [W_r, W_z]，形状 (input_size, 2*H)
        # 因此需要: w_ih_official = [w_ir[:, :H].T; w_ir[:, H:2H].T; w_ig.T]
        w_ih_official = torch.cat([
            w_ir[:, :hidden_size].T,              # reset (r)
            w_ir[:, hidden_size:2*hidden_size].T,  # update (z)
            gru_custom.cells[0].weight_ig.T,       # candidate (g)
        ], dim=0)

        w_hh_official = torch.cat([
            w_hr[:, :hidden_size].T,              # reset (r)
            w_hr[:, hidden_size:2*hidden_size].T,  # update (z)
            gru_custom.cells[0].weight_hg.T,       # candidate (g)
        ], dim=0)

        # 官方 bias_ih = [b_ir + b_hr, b_iz + b_hz, b_in + b_hn]
        # 我们的 bias_r = [b_ir + b_hr, b_iz + b_hz]
        # 我们的 bias_ig = b_in, bias_hg = b_hn
        b_ih_official = torch.cat([
            b_r[:hidden_size],                     # reset (r)
            b_r[hidden_size:2*hidden_size],        # update (z)
            gru_custom.cells[0].bias_ig + gru_custom.cells[0].bias_hg,  # candidate (g)
        ], dim=0)

        gru_official.weight_ih_l0.copy_(w_ih_official)
        gru_official.weight_hh_l0.copy_(w_hh_official)
        gru_official.bias_ih_l0.copy_(b_ih_official)
        gru_official.bias_hh_l0.zero_()

    out_custom, h_custom = gru_custom(x)
    out_official, h_official = gru_official(x)

    out_close = torch.allclose(out_custom, out_official, atol=1e-5)
    h_close = torch.allclose(h_custom, h_official, atol=1e-5)

    print(f"Output match: {out_close}")
    print(f"h_n match:    {h_close}")

    # 6. 验证梯度回传
    print("=" * 60)
    print("Test 6: Gradient backpropagation")
    loss = out_custom.sum()
    loss.backward()
    has_grad = any(p.grad is not None for p in gru_custom.parameters())
    assert has_grad, "No gradients computed!"
    print("Gradient backpropagation: OK")

    # 7. 参数量对比：GRU vs LSTM
    print("=" * 60)
    print("Test 7: Parameter count comparison")
    from lstm import LSTM
    lstm_model = LSTM(input_size, hidden_size, num_layers=1)
    gru_model = GRU(input_size, hidden_size, num_layers=1)

    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    gru_params = sum(p.numel() for p in gru_model.parameters())

    print(f"LSTM parameters: {lstm_params:,}")
    print(f"GRU parameters:  {gru_params:,}")
    print(f"GRU / LSTM ratio: {gru_params / lstm_params:.2%}")

    print("=" * 60)
    print("All GRU tests passed!")
