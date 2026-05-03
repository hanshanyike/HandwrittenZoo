"""
RNN 模块单元测试
测试 LSTM、GRU、BiLSTM
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest

from rnn.lstm import LSTMCell, LSTM
from rnn.gru import GRUCell, GRU
from rnn.bilstm import BiLSTM


class TestLSTMCell:
    """测试 LSTM 单元"""

    def test_output_shape(self):
        input_size = 64
        hidden_size = 128
        cell = LSTMCell(input_size, hidden_size)
        x = torch.randn(2, input_size)
        h = torch.randn(2, hidden_size)
        c = torch.randn(2, hidden_size)
        new_h, new_c = cell(x, h, c)
        assert new_h.shape == (2, hidden_size)
        assert new_c.shape == (2, hidden_size)

    def test_forget_gate(self):
        input_size = 64
        hidden_size = 128
        cell = LSTMCell(input_size, hidden_size)
        x = torch.zeros(2, input_size)
        h = torch.zeros(2, hidden_size)
        c = torch.ones(2, hidden_size)
        new_h, new_c = cell(x, h, c)
        # 零输入时 forget gate 应接近 0.5（因为 sigmoid(0)=0.5），cell state 会衰减
        assert not torch.allclose(new_c, c, atol=1e-2)


class TestLSTM:
    """测试 LSTM 模型"""

    def test_output_shape(self):
        input_size = 64
        hidden_size = 128
        num_layers = 2
        lstm = LSTM(input_size, hidden_size, num_layers)
        x = torch.randn(5, 10, input_size)
        out, (h_n, c_n) = lstm(x)
        assert out.shape == (5, 10, hidden_size)
        assert h_n.shape == (num_layers, 5, hidden_size)
        assert c_n.shape == (num_layers, 5, hidden_size)

    def test_single_layer(self):
        input_size = 64
        hidden_size = 128
        lstm = LSTM(input_size, hidden_size, num_layers=1)
        x = torch.randn(3, 5, input_size)
        out, (h_n, c_n) = lstm(x)
        assert out.shape == (3, 5, hidden_size)
        assert h_n.shape == (1, 3, hidden_size)

    def test_hidden_state_passthrough(self):
        input_size = 64
        hidden_size = 128
        lstm = LSTM(input_size, hidden_size, num_layers=1)
        x = torch.randn(3, 5, input_size)
        h0 = torch.randn(1, 3, hidden_size)
        c0 = torch.randn(1, 3, hidden_size)
        out, (h_n, c_n) = lstm(x, (h0, c0))
        assert h_n.shape == (1, 3, hidden_size)
        assert c_n.shape == (1, 3, hidden_size)


class TestGRUCell:
    """测试 GRU 单元"""

    def test_output_shape(self):
        input_size = 64
        hidden_size = 128
        cell = GRUCell(input_size, hidden_size)
        x = torch.randn(2, input_size)
        h = torch.randn(2, hidden_size)
        new_h = cell(x, h)
        assert new_h.shape == (2, hidden_size)

    def test_update_gate(self):
        input_size = 64
        hidden_size = 128
        cell = GRUCell(input_size, hidden_size)
        x = torch.zeros(2, input_size)
        h = torch.ones(2, hidden_size)
        new_h = cell(x, h)
        # 零输入时 update gate 为 0.5，输出应为 0.5*h + 0.5*候选状态
        assert not torch.allclose(new_h, h, atol=1e-2)


class TestGRU:
    """测试 GRU 模型"""

    def test_output_shape(self):
        input_size = 64
        hidden_size = 128
        num_layers = 2
        gru = GRU(input_size, hidden_size, num_layers)
        x = torch.randn(5, 10, input_size)
        out, h_n = gru(x)
        assert out.shape == (5, 10, hidden_size)
        assert h_n.shape == (num_layers, 5, hidden_size)

    def test_single_layer(self):
        input_size = 64
        hidden_size = 128
        gru = GRU(input_size, hidden_size, num_layers=1)
        x = torch.randn(3, 5, input_size)
        out, h_n = gru(x)
        assert out.shape == (3, 5, hidden_size)
        assert h_n.shape == (1, 3, hidden_size)

    def test_hidden_state_passthrough(self):
        input_size = 64
        hidden_size = 128
        gru = GRU(input_size, hidden_size, num_layers=1)
        x = torch.randn(3, 5, input_size)
        h0 = torch.randn(1, 3, hidden_size)
        out, h_n = gru(x, h0)
        assert h_n.shape == (1, 3, hidden_size)


class TestBiLSTM:
    """测试双向 LSTM"""

    def test_output_shape(self):
        input_size = 64
        hidden_size = 128
        num_layers = 2
        bilstm = BiLSTM(input_size, hidden_size, num_layers)
        x = torch.randn(5, 10, input_size)
        out, (h_n, c_n) = bilstm(x)
        # 双向输出维度为 2*hidden_size
        assert out.shape == (5, 10, 2 * hidden_size)
        assert h_n.shape == (2 * num_layers, 5, hidden_size)
        assert c_n.shape == (2 * num_layers, 5, hidden_size)

    def test_bidirectional_difference(self):
        input_size = 64
        hidden_size = 128
        bilstm = BiLSTM(input_size, hidden_size, num_layers=1)
        x = torch.randn(1, 5, input_size)
        out, _ = bilstm(x)
        # 双向输出应包含前向和后向信息
        assert out.shape[-1] == 2 * hidden_size

    def test_vs_unidirectional(self):
        input_size = 64
        hidden_size = 128
        # 双向 LSTM 输出维度应为单向的 2 倍
        bilstm = BiLSTM(input_size, hidden_size, num_layers=1)
        lstm = LSTM(input_size, hidden_size, num_layers=1)
        x = torch.randn(2, 5, input_size)
        bi_out, _ = bilstm(x)
        uni_out, _ = lstm(x)
        assert bi_out.shape[-1] == 2 * uni_out.shape[-1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
