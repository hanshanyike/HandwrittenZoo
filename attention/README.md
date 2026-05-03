# Attention Mechanisms

本目录收录了 Transformer 架构中各类注意力机制的核心实现，从标准 MHA 到现代推理优化变体（MQA/GQA/MLA），以及高效的 FlashAttention 实现。

## 文件清单

| 文件 | 说明 |
|------|------|
| `multi_head_attention.py` + `.md` | 标准多头注意力（MHA），Transformer 原始设计 |
| `multi_query_attention.py` + `.md` | 多查询注意力（MQA），共享 K/V 压缩 Cache |
| `grouped_query_attention.py` + `.md` | 分组查询注意力（GQA），MHA 与 MQA 的折中 |
| `multi_head_latent_attention.py` + `.md` | 多头潜在注意力（MLA），DeepSeek 核心创新 |
| `flash_attention.py` + `.md` | 简化版 FlashAttention（分块 + 在线 softmax）|
| `self_attention.py` + `.md` | 自注意力独立模块 |
| `cross_attention.py` + `.md` | 交叉注意力独立模块 |

## 演进路线

```
MHA (2017)
  │
  ├── 推理优化方向 ──┬── MQA (2019) ── 共享所有 K/V，Cache 最小，能力可能下降
  │                  ├── GQA (2023) ── 分组共享 K/V，平衡 Cache 与能力
  │                  └── MLA (2024) ── 低秩压缩 + 解耦 RoPE，Cache 极小且能力更强
  │
  └── 计算优化方向 ─── FlashAttention (2022) ── 分块 + 在线 softmax，显存 O(N)
```

## 面试焦点

### 1. 复杂度与 Cache 对比
| 机制 | 每 token KV Cache | 表达能力 |
|------|------------------|----------|
| MHA | $2 n_h d_h$ | 最强 |
| GQA | $2 n_g d_h$ | 中等 |
| MQA | $2 d_h$ | 较弱 |
| MLA | $d_c + d_h^R$ | 优于 MHA |

### 2. 高频考点速查
- **为什么除以 $\sqrt{d_k}$？**: 稳定 softmax 梯度，防止点积过大导致饱和。
- **MQA 为什么加速推理？**: 减少 K/V 头数，降低显存带宽瓶颈。
- **GQA 与 MQA 的区别？**: GQA 保留多组 K/V，避免所有 query head 共享同一 K/V 导致的表达能力下降。
- **MLA 的核心创新？**: 低秩联合压缩 KV + 解耦 RoPE，使 KV Cache 压缩数十倍且能力不降。
- **FlashAttention 为什么省显存？**: 分块计算避免存储 $N \times N$ 注意力矩阵，在线 softmax 支持逐块更新。
- **自注意力 vs 交叉注意力？**: 自注意力 Q/K/V 同源；交叉注意力 Q 来自 Decoder，K/V 来自 Encoder。

### 3. 现代大模型选型
- **GPT-4 / LLaMA-1**: 标准 MHA
- **PaLM / Falcon**: MQA
- **LLaMA-2 70B / Mistral / LLaMA-3**: GQA
- **DeepSeek-V2/V3/R1**: MLA
- **所有现代高效实现**: FlashAttention / FlashAttention-2 / FlashAttention-3

## 运行自测

每个 `.py` 文件底部都包含 `if __name__ == "__main__":` 自测块，可直接运行：

```bash
python attention/multi_head_attention.py
python attention/multi_query_attention.py
python attention/grouped_query_attention.py
python attention/multi_head_latent_attention.py
python attention/flash_attention.py
python attention/self_attention.py
python attention/cross_attention.py
```

## 参考资料汇总
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434)
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437)
