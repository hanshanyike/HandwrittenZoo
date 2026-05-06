# Mamba: 选择性状态空间模型

本目录收录 Mamba 架构的从零实现。Mamba 是 2023-2024 年兴起的新型序列建模架构，旨在挑战 Transformer 的统治地位，核心创新是引入**选择性机制（Selection Mechanism）**，使 SSM 参数成为输入的函数，从而实现 $O(n)$ 线性时间复杂度的同时保持与 Transformer 相媲美的建模能力。

## 文件清单

| 文件 | 内容 | 面试重点 |
|------|------|----------|
| `mamba.py` + `mamba.md` | Mamba 完整实现（选择性 SSM + 并行扫描） | 极高 — 2024-2025 大模型面试新热点 |

## 面试焦点速览

### 1. Mamba vs Transformer

| 特性 | Transformer | Mamba |
|------|------------|-------|
| 时间复杂度 | $O(n^2)$ | $O(n)$ |
| 空间复杂度 | $O(n^2)$ | $O(n \cdot d_{state})$ |
| 选择性 | Attention 天然支持 | 通过选择性 SSM 实现 |
| 序列外推 | 困难（位置编码限制） | 优秀（递归形式天然支持） |
| 显存占用 | 随序列长度二次增长 | 线性增长 |

### 2. 核心创新：选择性机制

传统 S4 的问题：**A、B、C 是静态参数，与输入无关**

Mamba 的解决方案：让参数由输入动态生成

```python
# 输入依赖的投影
[x, dt, B, C] = x_proj(x_ssm)  # 全部由输入计算
```

这使得模型可以像注意力机制一样，根据内容选择性关注/忽略信息。

### 3. 高频考点

- **为什么需要局部卷积（d_conv）？** SSM 递归难以捕捉局部模式，卷积帮助捕获 n-gram 特征
- **为什么叫"线性"时间？** 每个 token 只与前一状态和当前输入交互，状态维度固定
- **选择性扫描是什么？** 硬件感知的并行扫描算法，利用 GPU 并行能力加速递归计算

## 运行自测

```bash
python mamba/mamba.py
```

## 参考资料

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [Mamba-minimal 官方实现](https://github.com/state-spaces/mamba)
