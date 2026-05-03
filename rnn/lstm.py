"""
LSTM (Long Short-Term Memory)

Algorithm: 长短期记忆网络 —— 通过引入"细胞状态"（Cell State）和三门控机制
         （遗忘门、输入门、输出门）解决传统 RNN 的梯度消失/爆炸问题，
         实现对长距离依赖的有效建模。

Core Idea:
- 传统 RNN 的隐藏状态在每个时间步都被完全覆盖更新，导致长程信息难以保留。
- LSTM 引入一条"信息高速公路"——细胞状态 C_t，它贯穿整个时间序列，
  仅通过门控进行线性交互（加法和乘法），梯度可以相对完整地反向传播。
- 三个门控各司其职：遗忘门决定丢弃什么旧信息，输入门决定存储什么新信息，
  输出门决定隐藏状态输出什么。

Complexity:
    - 时间复杂度: O(T * (4 * H * (H + I))) —— T为序列长度，H为隐藏维度，I为输入维度
    - 空间复杂度: O(T * H) —— 存储每个时间步的隐藏状态和细胞状态
    - 参数量: 4 * (I + H) * H + 4 * H（四个门的权重和偏置）

Interview Frequency: 极高（RNN 面试必考，三门控机制是核心考点）

References:
    - Hochreiter & Schmidhuber, "Long Short-Term Memory", Neural Computation 1997
"""

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    """
    单个 LSTM 单元的从零实现。

    门控机制:
        遗忘门 f_t: 决定从细胞状态中丢弃哪些旧信息
        输入门 i_t: 决定哪些新候选信息将被存储到细胞状态
        候选状态 g_t: 当前时间步可能的新信息
        输出门 o_t: 决定细胞状态的哪些部分将输出到隐藏状态

    状态更新:
        C_t = f_t * C_{t-1} + i_t * g_t   （细胞状态：线性更新）
        h_t = o_t * tanh(C_t)              （隐藏状态：非线性输出）

    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 将输入 x_t 和上一时刻隐藏状态 h_{t-1} 拼接后，一次性计算四个门的线性变换
        # 这比分别定义四个线性层更高效，且便于并行计算
        # 权重矩阵形状: (input_size + hidden_size, 4 * hidden_size)
        self.weight_ih = nn.Parameter(
            torch.randn(input_size, 4 * hidden_size) * 0.01
        )
        self.weight_hh = nn.Parameter(
            torch.randn(hidden_size, 4 * hidden_size) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(4 * hidden_size))

        # 初始化：遗忘门偏置设为 1，防止初始阶段遗忘过多信息
        # 这是 LSTM 训练的一个常用技巧（Jozefowicz et al., 2015）
        with torch.no_grad():
            self.bias[hidden_size:2 * hidden_size].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        单个时间步的前向传播。

        Args:
            x: 当前时间步输入，形状 (batch, input_size)
            h_prev: 上一时刻隐藏状态，形状 (batch, hidden_size)
            c_prev: 上一时刻细胞状态，形状 (batch, hidden_size)
        Returns:
            h_t: 当前时刻隐藏状态，形状 (batch, hidden_size)
            c_t: 当前时刻细胞状态，形状 (batch, hidden_size)
        """
        # 1) 拼接输入和上一时刻隐藏状态
        # 形状: (batch, input_size + hidden_size)
        combined = torch.cat([x, h_prev], dim=-1)

        # 2) 计算四个门的预激活值（gates pre-activation）
        # 形状: (batch, 4 * hidden_size)
        gates = torch.matmul(combined, torch.cat([self.weight_ih, self.weight_hh], dim=0))
        gates = gates + self.bias

        # 3) 切分为四个门
        # chunk(4, dim=-1) 沿最后一维切分为 4 份，每份 hidden_size
        i, f, g, o = gates.chunk(4, dim=-1)

        # 4) 应用门控激活函数
        # 输入门：决定存储多少新信息（0~1）
        i = torch.sigmoid(i)
        # 遗忘门：决定保留多少旧信息（0~1）
        f = torch.sigmoid(f)
        # 候选状态：当前可能的新信息（-1~1）
        g = torch.tanh(g)
        # 输出门：决定输出多少细胞状态信息（0~1）
        o = torch.sigmoid(o)

        # 5) 更新细胞状态（核心：线性路径，梯度易传播）
        # C_t = f * C_{t-1} + i * g
        # 遗忘门控制旧信息的保留比例，输入门控制新信息的写入比例
        c_t = f * c_prev + i * g

        # 6) 计算隐藏状态（对细胞状态做非线性变换后输出）
        # h_t = o * tanh(C_t)
        # 输出门控制哪些信息被暴露给外部
        h_t = o * torch.tanh(c_t)

        return h_t, c_t


class LSTM(nn.Module):
    """
    完整 LSTM 网络实现（支持多层和单向）。

    架构:
        1. 输入投影（可选）：将输入维度映射到隐藏维度
        2. 多层 LSTM 堆叠：每层由多个 LSTMCell 按时间步展开
        3. 输出投影（可选）：将隐藏状态映射到输出维度

    Args:
        input_size: 输入特征维度
        hidden_size: 隐藏状态维度
        num_layers: LSTM 层数，默认 1
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

        # 构建多层 LSTM：每一层是一个 LSTMCell
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            self.cells.append(LSTMCell(layer_input_size, hidden_size))

        # 层间 Dropout（训练时对非最后一层的输出应用）
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

        # 可选的输出投影层
        if output_size is not None:
            self.output_projection = nn.Linear(hidden_size, output_size)
        else:
            self.output_projection = None

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播。

        Args:
            x: 输入序列，形状 (batch, seq_len, input_size)
            hidden: 可选的初始状态 (h_0, c_0)，每个形状 (num_layers, batch, hidden_size)
        Returns:
            outputs: 所有时间步的输出，形状 (batch, seq_len, hidden_size 或 output_size)
            (h_n, c_n): 最后时刻的隐藏状态和细胞状态，形状均为 (num_layers, batch, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # 初始化隐藏状态和细胞状态（若未提供）
        if hidden is None:
            h = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype
            )
            c = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                device=x.device, dtype=x.dtype
            )
        else:
            h, c = hidden

        # 存储每一层、每个时间步的输出
        # layer_outputs[layer][t] = (batch, hidden_size)
        layer_outputs = [[] for _ in range(self.num_layers)]

        # 按时间步展开
        for t in range(seq_len):
            layer_input = x[:, t, :]  # 当前时间步输入: (batch, input_size)

            for layer in range(self.num_layers):
                # 当前层的 LSTMCell 前向传播
                h_t, c_t = self.cells[layer](layer_input, h[layer], c[layer])

                # 更新该层的隐藏状态和细胞状态
                h = h.clone()
                c = c.clone()
                h[layer] = h_t
                c[layer] = c_t

                # 存储该层该时间步的输出
                layer_outputs[layer].append(h_t)

                # 下一层的输入是当前层的输出
                layer_input = h_t

                # 层间 Dropout（除最后一层外）
                if self.dropout_layer is not None and layer < self.num_layers - 1:
                    layer_input = self.dropout_layer(layer_input)

        # 取最后一层的所有时间步输出，堆叠为 (batch, seq_len, hidden_size)
        outputs = torch.stack(layer_outputs[-1], dim=1)

        # 可选的输出投影
        if self.output_projection is not None:
            outputs = self.output_projection(outputs)

        return outputs, (h, c)


if __name__ == "__main__":
    # ==================== 自测区域 ====================
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    input_size = 16
    hidden_size = 32
    num_layers = 2
    x = torch.randn(batch_size, seq_len, input_size)

    # 1. 测试单个 LSTMCell
    print("=" * 60)
    print("Test 1: LSTMCell forward pass")
    cell = LSTMCell(input_size, hidden_size)
    h_prev = torch.randn(batch_size, hidden_size)
    c_prev = torch.randn(batch_size, hidden_size)
    h_t, c_t = cell(x[:, 0, :], h_prev, c_prev)
    assert h_t.shape == (batch_size, hidden_size)
    assert c_t.shape == (batch_size, hidden_size)
    print(f"LSTMCell h_t shape: {h_t.shape}")
    print(f"LSTMCell c_t shape: {c_t.shape}")

    # 2. 测试单层 LSTM
    print("=" * 60)
    print("Test 2: Single-layer LSTM")
    lstm_single = LSTM(input_size, hidden_size, num_layers=1)
    out, (h_n, c_n) = lstm_single(x)
    assert out.shape == (batch_size, seq_len, hidden_size)
    assert h_n.shape == (1, batch_size, hidden_size)
    assert c_n.shape == (1, batch_size, hidden_size)
    print(f"LSTM output shape: {out.shape}")
    print(f"Final h_n shape:   {h_n.shape}")
    print(f"Final c_n shape:   {c_n.shape}")

    # 3. 测试多层 LSTM
    print("=" * 60)
    print("Test 3: Multi-layer LSTM")
    lstm_multi = LSTM(input_size, hidden_size, num_layers=num_layers, dropout=0.3)
    out_m, (h_m, c_m) = lstm_multi(x)
    assert out_m.shape == (batch_size, seq_len, hidden_size)
    assert h_m.shape == (num_layers, batch_size, hidden_size)
    assert c_m.shape == (num_layers, batch_size, hidden_size)
    print(f"Multi-layer LSTM output shape: {out_m.shape}")
    print(f"Final h_n shape:               {h_m.shape}")

    # 4. 测试带输出投影的 LSTM
    print("=" * 60)
    print("Test 4: LSTM with output projection")
    output_size = 8
    lstm_proj = LSTM(input_size, hidden_size, num_layers=1, output_size=output_size)
    out_p, _ = lstm_proj(x)
    assert out_p.shape == (batch_size, seq_len, output_size)
    print(f"LSTM with projection output shape: {out_p.shape}")

    # 5. 测试自定义实现与 PyTorch 官方 LSTM 的数值一致性
    print("=" * 60)
    print("Test 5: Numerical comparison with PyTorch official LSTM")

    # 构建一个单层、无投影的 LSTM 用于对比
    lstm_custom = LSTM(input_size, hidden_size, num_layers=1)
    lstm_official = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)

    # 将自定义 LSTM 的参数同步到官方 LSTM
    # 注意：官方 LSTM 的 weight_ih 形状为 (4*hidden_size, input_size)
    with torch.no_grad():
        lstm_official.weight_ih_l0.copy_(lstm_custom.cells[0].weight_ih.T)
        lstm_official.weight_hh_l0.copy_(lstm_custom.cells[0].weight_hh.T)
        lstm_official.bias_ih_l0.copy_(lstm_custom.cells[0].bias)
        # 官方实现将 bias 拆分为 bias_ih 和 bias_hh，我们将自定义的 bias 全部分配给 bias_ih
        lstm_official.bias_hh_l0.zero_()

    out_custom, (h_custom, c_custom) = lstm_custom(x)
    out_official, (h_official, c_official) = lstm_official(x)

    out_close = torch.allclose(out_custom, out_official, atol=1e-5)
    h_close = torch.allclose(h_custom, h_official, atol=1e-5)
    c_close = torch.allclose(c_custom, c_official, atol=1e-5)

    print(f"Output match: {out_close}")
    print(f"h_n match:    {h_close}")
    print(f"c_n match:    {c_close}")

    # 6. 验证梯度回传
    print("=" * 60)
    print("Test 6: Gradient backpropagation")
    loss = out_custom.sum()
    loss.backward()
    has_grad = any(p.grad is not None for p in lstm_custom.parameters())
    assert has_grad, "No gradients computed!"
    print("Gradient backpropagation: OK")

    print("=" * 60)
    print("All LSTM tests passed!")
