# HandwrittenZoo

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-1.12%2B-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Algorithms-50%2B-orange.svg" alt="Algorithms">
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/hanshanyike/HandwrittenZoo?style=social" alt="Stars">
  <img src="https://img.shields.io/github/forks/hanshanyike/HandwrittenZoo?style=social" alt="Forks">
  <img src="https://img.shields.io/github/issues/hanshanyike/HandwrittenZoo" alt="Issues">
</p>

<p align="center">
  <strong>🔗 <a href="README.md">中文</a> | English (Coming Soon)</strong>
</p>

---


HandwrittenZoo 是一个面向**算法工程师**与**大模型工程师**面试的深度学习核心算法手撕代码仓库。本仓库系统覆盖从经典网络到前沿大模型组件的完整技术栈，每个算法均提供**从零实现的 Python 代码** + **配套 Markdown 文档**，帮助面试者深入理解原理、快速备战高频考点。

---

<p align="center">
  <a href="#目录结构">📁 目录结构</a> •
  <a href="#面试重点提示">📝 面试重点</a> •
  <a href="#如何运行">🚀 快速开始</a> •
  <a href="#文件配对规范">📖 使用指南</a> •
  <a href="#学习路线建议">🛤️ 学习路线</a>
</p>

---

## 目录结构

```
HandwrittenZoo/
├── transformer/              # Transformer 架构全系列（原始 Transformer、BERT、GPT、LLaMA 等）
├── attention/                 # 注意力机制及其变体（MHA、MQA、GQA、FlashAttention、MLA 等）
├── position_encoding/         # 位置编码（正弦/余弦、RoPE、ALiBi 等）
├── normalization/             # 归一化层（LayerNorm、RMSNorm、BatchNorm 等）
├── activation/                # 激活函数与前馈网络（SwiGLU、GELU、FFN 变体等）
├── cnn/                       # 经典卷积网络（ResNet、VGG 等）
├── rnn/                       # 循环网络（LSTM、GRU、BiLSTM 等）
├── generative_model/          # 生成模型（VAE、GAN、Diffusion Model 等）
├── rl/                        # 强化学习与对齐（PPO、DPO、GRPO、Reward Model 等）
├── moe/                       # 混合专家模型（MoE、门控机制、负载均衡等）
├── mamba/                     # 选择性状态空间模型（Mamba SSM，线性时间复杂度）
├── fine_tuning/               # 参数高效微调（LoRA、QLoRA、Prefix Tuning 等）
├── inference_optimization/    # 推理优化（KV Cache、PageAttention、量化、投机解码等）
├── tokenization/              # 分词算法（BPE、WordPiece、SentencePiece 等）
├── tests/                     # 单元测试与验证
├── requirements.txt          # Python 依赖
└── README.md                 # 本文件
```

---

## 面试重点提示

| 技术方向 | 面试频率 | 核心考点 |
|---------|---------|---------|
| Transformer 架构 | 极高 | 自注意力计算、Mask 机制、Encoder-Decoder 差异、参数量估算 |
| 注意力变体 | 极高 | MHA vs MQA vs GQA 的区别与动机、FlashAttention 分块思想、MLA 低秩压缩 |
| 位置编码 | 高 | RoPE 旋转矩阵推导、ALiBi 外推性、长度外推问题 |
| 归一化层 | 高 | LayerNorm vs RMSNorm、Pre-Norm vs Post-Norm、为什么大模型用 RMSNorm |
| 激活与 FFN | 高 | SwiGLU 参数量陷阱、GELU 平滑性、各种激活函数对比 |
| 推理优化 | 高 | KV Cache 原理、PageAttention 分页管理、量化对精度的影响、投机解码加速机制 |
| 强化学习对齐 | 高 | PPO 裁剪目标、DPO 直接偏好优化、GRPO 组相对策略优化 |
| Mamba / SSM | 极高 | 选择性 SSM、线性复杂度、选择性扫描、与 Transformer 的区别 |
| 混合专家模型 | 中高 | MoE 门控机制、Top-K 路由、负载均衡损失、Switch Transformer |
| 参数高效微调 | 中 | LoRA 低秩分解、QLoRA 量化 + LoRA、Prefix Tuning 原理 |
| 生成模型 | 中 | VAE 重参数化技巧、GAN 训练稳定性、Diffusion 前向加噪与反向去噪 |
| 经典 CNN/RNN | 中 | 残差连接解决梯度消失、LSTM 门控机制、GRU 简化设计 |
| 分词算法 | 中 | BPE 合并策略、WordPiece 似然目标、SentencePiece 无预分词优势 |

---

## 如何运行

### 环境要求

- Python >= 3.8
- PyTorch >= 1.12
- NumPy >= 1.21

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行单个算法示例

```bash
# 例如运行 Transformer 完整实现
python transformer/transformer_full.py

# 例如运行 RoPE 位置编码
python position_encoding/rope/rope.py
```

### 运行测试

```bash
python -m unittest discover tests/
```

---

## 文件配对规范

每个核心算法对应**一对文件**：

- `algorithm_name.py` — 纯代码实现（含详细中文注释、docstring、复杂度分析）
- `algorithm_name.md` — 配套文档（核心思想、数学公式、复杂度分析、面试高频考点、代码解析）

---

## 学习路线建议

1. **基础阶段**：先掌握 `cnn/`、`rnn/` 经典网络，理解深度学习基础组件。
2. **核心阶段**：深入 `transformer/`、`attention/`、`position_encoding/`、`normalization/`，这是大模型面试的绝对核心。
3. **进阶阶段**：学习 `rl/`（对齐算法）、`moe/`（混合专家）、`inference_optimization/`（推理优化），这些是 2024-2025 面试的新 heavyweight。
4. **实战阶段**：通过 `fine_tuning/` 和 `tests/` 进行代码实践与验证。

---

## 贡献与反馈

欢迎提交 Issue 或 PR，补充更多高频面试算法或优化现有实现。

---

*本仓库持续更新，聚焦算法/大模型工程师面试实战。*
