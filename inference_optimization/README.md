# Inference Optimization

本目录聚焦大语言模型（LLM）推理阶段的核心优化技术。与训练不同，推理面临的是 **低延迟、高并发、显存受限** 的严苛环境，因此催生了一系列以 **空间换时间、降低精度换带宽、精细显存管理** 为思想的工程算法。

## 目录结构

| 文件 | 内容 | 面试重点 |
|------|------|----------|
| [kv_cache.py](kv_cache.py) + [kv_cache.md](kv_cache.md) | KV Cache 机制：缓存历史 K/V 避免重复计算 | 高 — 所有 LLM 推理的基础 |
| [page_attention.py](page_attention.py) + [page_attention.md](page_attention.md) | PageAttention 简化实现：分页式 KV Cache 管理 | 高 — vLLM 核心，显存碎片与共享 |
| [quantization.py](quantization.py) + [quantization.md](quantization.md) | 基础量化：INT8/INT4 对称/非对称量化 | 高 — 模型压缩与部署必考 |

## 面试焦点速览

### 1. KV Cache
- **为什么需要**：自回归生成中，若不缓存，每步都要对完整前缀重新计算 Attention，复杂度 $O(n^2)$。
- **代价**：显存随序列长度线性增长，成为长上下文瓶颈。
- **延伸**：MQA/GQA 压缩 KV Cache；KV Cache 量化进一步减半显存。

### 2. PageAttention
- **核心洞察**：将逻辑 token 序列与物理显存解耦，类比操作系统虚拟内存分页。
- **解决痛点**：消除内部/外部碎片；通过引用计数实现 beam search、并行采样的内存共享。
- **性能收益**：相同显存下并发量提升 **2~4x**。

### 3. Quantization
- **本质**：用 scale 和 zero-point 将浮点映射到整型，降低存储与带宽。
- **对称 vs 非对称**：权重通常对称；激活常非对称（分布有偏）。
- **per-tensor vs per-channel**：per-channel 精度更高，LLM 权重常用 per-channel 或 per-group。
- **延伸**：PTQ（训练后量化）部署快；QAT（量化感知训练）精度高但成本高；GPTQ/AWQ 是 LLM 4-bit 量化的代表算法。

## 学习建议

1. **先理解原理**：阅读每个 `.md` 的「核心思想」和「数学公式」部分，确保能白板推导。
2. **再读代码**：`.py` 文件均为 from-scratch 实现，配有中文注释，建议逐行跟读并运行自测块。
3. **面试串联**：这三个技术并非孤立——KV Cache 解决计算冗余，PageAttention 解决 KV Cache 的显存管理问题，Quantization 解决权重与 KV Cache 的存储瓶颈。面试时可按此脉络递进阐述。

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- [Hugging Face Quantization Guide](https://huggingface.co/docs/transformers/quantization)
