"""
BiLSTM (Bidirectional LSTM)

Algorithm: 双向长短期记忆网络 —— 同时训练两个独立的 LSTM：
         一个按正序处理序列（前向 LSTM），一个按逆序处理序列（后向 LSTM），
         将两者的隐藏状态拼接，使每个时间步的表示同时包含过去和未来的上下文信息。

Core Idea:
- 单向 LSTM 只能利用当前时间步之前的信息（左上下文），无法看到未来的信息（右上下文）。
- 在许多 NLP 任务中（如命名实体识别、情感分析、词性标注），当前词的理解需要同时依赖
  前后文信息。例如，"bank" 在"river bank"和"bank account"中的含义不同。
- BiLSTM 通过两个方向独立的 LSTM 分别编码左上下文和右上下文，然后拼接或求和，
  获得更完整的上下文表示。
- 两个方向的 LSTM 参数不共享，各自独立学习。

Complexity:
    - 时间复杂度: O(T * (8 * H * (H + I))) —— 约为单向 LSTM 的 2 倍
    - 空间复杂度: O(T * 2 * H) —— 存储两个方向的隐藏状态和细胞状态
    - 参数量: 2 * [4 * (I + H) * H + 4 * H]（两个独立 LSTM 的参数之和）

Interview Frequency: 极高（NLP 面试必考，双向机制是核心考点）

References:
    - Schuster & Paliwal, "Bidirectional Recurrent Neural Networks", IEEE 1997
    - Graves et al., "Speech Recognition with Deep Recurrent Neural Networks", ICASSP 2013
"""

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    """
    双向 LSTM 的从零实现。

    架构:
        1. 前向 LSTM（Forward LSTM）: 按时间步 0 -> T-1 顺序处理序列
        2. 后向 LSTM（Backward LSTM）: 按时间步 T-1 -> 0 逆序处理序列
        3. 输出融合: 将两个方向在每个时间步的隐藏状态拼接（concat）或求和（sum）

    输出形状:
        - concat 模式: (batch, seq_len, 2 * hidden_size)
        - sum 模式:   (batch, seq_len, hidden_size)

    Args:
        input_size: 输入特征维度
        hidden_size: 每个方向 LSTM 的隐藏状态维度
        num_layers: 每个方向的 LSTM 层数，默认 1
        output_size: 输出投影维度，若为 None 则输出融合后的双向状态
        dropout: 层间 Dropout 概率
        merge_mode: 双向输出融合方式，"concat" 或 "sum"，默认 "concat"
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        output_size: int | None = None,
        dropout: float = 0.0,
        merge_mode: str = "concat",
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        assert merge_mode in ("concat", "sum"), "merge_mode 必须为 'concat' 或 'sum'"
        self.merge_mode = merge_mode

        # 前向 LSTM：按正序处理
        # 使用 PyTorch 官方 LSTM 作为基础，确保数值稳定性
        self.forward_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # 后向 LSTM：按逆序处理（参数独立于前向 LSTM）
        self.backward_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # 可选的输出投影层
        # concat 模式下输入维度为 2*hidden_size，sum 模式下为 hidden_size
        proj_input_size = 2 * hidden_size if merge_mode == "concat" else hidden_size
        if output_size is not None:
            self.output_projection = nn.Linear(proj_input_size, output_size)
        else:
            self.output_projection = None

    def _init_hidden(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """初始化隐藏状态和细胞状态。"""
        h = torch.zeros(
            self.num_layers, batch_size, self.hidden_size,
            device=device, dtype=dtype
        )
        c = torch.zeros(
            self.num_layers, batch_size, self.hidden_size,
            device=device, dtype=dtype
        )
        return h, c

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播。

        Args:
            x: 输入序列，形状 (batch, seq_len, input_size)
            hidden: 可选的初始状态 ((h_f, c_f), (h_b, c_b))，
                    每个形状 (num_layers, batch, hidden_size)
        Returns:
            outputs: 融合后的双向输出
                - concat 模式: (batch, seq_len, 2 * hidden_size)
                - sum 模式:   (batch, seq_len, hidden_size)
            ((h_f, c_f), (h_b, c_b)): 两个方向最后时刻的隐藏状态和细胞状态
        """
        batch_size, seq_len, _ = x.shape

        # 初始化隐藏状态（若未提供）
        if hidden is None:
            h_f, c_f = self._init_hidden(batch_size, x.device, x.dtype)
            h_b, c_b = self._init_hidden(batch_size, x.device, x.dtype)
        else:
            (h_f, c_f), (h_b, c_b) = hidden

        # 1) 前向 LSTM：按正序处理
        # out_f: (batch, seq_len, hidden_size)
        out_f, (h_f_n, c_f_n) = self.forward_lstm(x, (h_f, c_f))

        # 2) 后向 LSTM：按逆序处理
        # 先将输入序列在时间维度上翻转
        x_rev = torch.flip(x, dims=[1])  # (batch, seq_len, input_size)
        out_b_rev, (h_b_n, c_b_n) = self.backward_lstm(x_rev, (h_b, c_b))

        # 将后向输出再翻转回正序，以便与前向输出按时间步对齐
        # out_b: (batch, seq_len, hidden_size)
        out_b = torch.flip(out_b_rev, dims=[1])

        # 3) 融合两个方向的输出
        if self.merge_mode == "concat":
            # 拼接：每个时间步的表示维度翻倍
            outputs = torch.cat([out_f, out_b], dim=-1)
        else:  # sum
            # 求和：保持维度不变，但融合双向信息
            outputs = out_f + out_b

        # 4) 可选的输出投影
        if self.output_projection is not None:
            outputs = self.output_projection(outputs)

        return outputs, ((h_f_n, c_f_n), (h_b_n, c_b_n))


class BiLSTMClassifier(nn.Module):
    """
    基于 BiLSTM 的序列分类器示例。

    结构:
        Embedding -> BiLSTM -> 全局池化 -> 全连接分类

    常用于文本分类、情感分析等任务。

    Args:
        vocab_size: 词表大小
        embed_dim: 词嵌入维度
        hidden_size: BiLSTM 隐藏维度
        num_classes: 分类类别数
        num_layers: BiLSTM 层数
        dropout: Dropout 概率
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = BiLSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            merge_mode="concat",
        )
        # concat 模式下 BiLSTM 输出维度为 2 * hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入 token IDs，形状 (batch, seq_len)
        Returns:
            分类 logits，形状 (batch, num_classes)
        """
        # 词嵌入: (batch, seq_len) -> (batch, seq_len, embed_dim)
        emb = self.embedding(x)

        # BiLSTM: (batch, seq_len, embed_dim) -> (batch, seq_len, 2 * hidden_size)
        lstm_out, _ = self.bilstm(emb)

        # 全局平均池化：将序列维度压缩
        # (batch, seq_len, 2 * hidden_size) -> (batch, 2 * hidden_size)
        pooled = lstm_out.mean(dim=1)

        # 分类
        logits = self.classifier(pooled)
        return logits


if __name__ == "__main__":
    # ==================== 自测区域 ====================
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 5
    input_size = 16
    hidden_size = 32
    num_layers = 2
    x = torch.randn(batch_size, seq_len, input_size)

    # 1. 测试 BiLSTM (concat 模式)
    print("=" * 60)
    print("Test 1: BiLSTM with concat merge mode")
    bilstm_concat = BiLSTM(
        input_size, hidden_size, num_layers=num_layers, merge_mode="concat"
    )
    out_c, ((h_f, c_f), (h_b, c_b)) = bilstm_concat(x)
    assert out_c.shape == (batch_size, seq_len, 2 * hidden_size)
    assert h_f.shape == (num_layers, batch_size, hidden_size)
    assert h_b.shape == (num_layers, batch_size, hidden_size)
    print(f"BiLSTM (concat) output shape: {out_c.shape}")
    print(f"Forward h_n shape:            {h_f.shape}")
    print(f"Backward h_n shape:           {h_b.shape}")

    # 2. 测试 BiLSTM (sum 模式)
    print("=" * 60)
    print("Test 2: BiLSTM with sum merge mode")
    bilstm_sum = BiLSTM(
        input_size, hidden_size, num_layers=num_layers, merge_mode="sum"
    )
    out_s, _ = bilstm_sum(x)
    assert out_s.shape == (batch_size, seq_len, hidden_size)
    print(f"BiLSTM (sum) output shape: {out_s.shape}")

    # 3. 测试带输出投影的 BiLSTM
    print("=" * 60)
    print("Test 3: BiLSTM with output projection")
    output_size = 8
    bilstm_proj = BiLSTM(
        input_size, hidden_size, num_layers=1,
        output_size=output_size, merge_mode="concat"
    )
    out_p, _ = bilstm_proj(x)
    assert out_p.shape == (batch_size, seq_len, output_size)
    print(f"BiLSTM (projection) output shape: {out_p.shape}")

    # 4. 测试自定义 BiLSTM 与 PyTorch 官方 nn.LSTM(bidirectional=True) 的数值一致性
    print("=" * 60)
    print("Test 4: Numerical comparison with PyTorch official BiLSTM")

    bilstm_custom = BiLSTM(input_size, hidden_size, num_layers=1, merge_mode="concat")
    bilstm_official = nn.LSTM(
        input_size, hidden_size, num_layers=1,
        batch_first=True, bidirectional=True
    )

    # 参数同步：将自定义两个单向 LSTM 的参数复制到官方双向 LSTM
    with torch.no_grad():
        # 官方 LSTM 的权重命名：weight_ih_l0（前向第一层）, weight_ih_l0_reverse（后向第一层）
        bilstm_official.weight_ih_l0.copy_(bilstm_custom.forward_lstm.weight_ih_l0)
        bilstm_official.weight_hh_l0.copy_(bilstm_custom.forward_lstm.weight_hh_l0)
        bilstm_official.bias_ih_l0.copy_(bilstm_custom.forward_lstm.bias_ih_l0)
        bilstm_official.bias_hh_l0.copy_(bilstm_custom.forward_lstm.bias_hh_l0)

        bilstm_official.weight_ih_l0_reverse.copy_(bilstm_custom.backward_lstm.weight_ih_l0)
        bilstm_official.weight_hh_l0_reverse.copy_(bilstm_custom.backward_lstm.weight_hh_l0)
        bilstm_official.bias_ih_l0_reverse.copy_(bilstm_custom.backward_lstm.bias_ih_l0)
        bilstm_official.bias_hh_l0_reverse.copy_(bilstm_custom.backward_lstm.bias_hh_l0)

    out_custom, _ = bilstm_custom(x)
    out_official, _ = bilstm_official(x)

    out_close = torch.allclose(out_custom, out_official, atol=1e-5)
    print(f"Output match: {out_close}")

    # 5. 测试 BiLSTM 分类器
    print("=" * 60)
    print("Test 5: BiLSTM Classifier")
    vocab_size = 100
    embed_dim = 16
    num_classes = 4
    seq_len_text = 10

    classifier = BiLSTMClassifier(
        vocab_size, embed_dim, hidden_size, num_classes, num_layers=1
    )
    x_text = torch.randint(0, vocab_size, (batch_size, seq_len_text))
    logits = classifier(x_text)
    assert logits.shape == (batch_size, num_classes)
    print(f"Classifier logits shape: {logits.shape}")

    # 6. 验证梯度回传
    print("=" * 60)
    print("Test 6: Gradient backpropagation")
    loss = out_custom.sum()
    loss.backward()
    has_grad = any(p.grad is not None for p in bilstm_custom.parameters())
    assert has_grad, "No gradients computed!"
    print("Gradient backpropagation: OK")

    # 7. 参数量统计
    print("=" * 60)
    print("Test 7: Parameter count")
    custom_params = sum(p.numel() for p in bilstm_custom.parameters())
    official_params = sum(p.numel() for p in bilstm_official.parameters())
    print(f"Custom BiLSTM parameters:  {custom_params:,}")
    print(f"Official BiLSTM parameters: {official_params:,}")

    print("=" * 60)
    print("All BiLSTM tests passed!")
