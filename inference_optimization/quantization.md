# Quantization (Basic)

## 算法简介

量化（Quantization）是深度学习模型压缩与推理加速的核心技术之一。它通过将高精度浮点数（FP32/FP16）映射到低精度整数（INT8/INT4），在显著降低模型体积和显存带宽的同时，利用硬件对整数运算的加速能力提升推理吞吐。

## 核心思想

神经网络权重和激活的数值范围通常有限，无需 32 位浮点的巨大动态范围。量化的本质是用两个参数——**缩放因子（scale）** 和 **零点（zero-point）**——将浮点区间线性映射到整型区间：

- **对称量化**：假设浮点范围关于 0 对称，直接映射到 $[-Q_{max}, Q_{max}]$，省去 zero-point。
- **非对称量化**：不假设对称，用 zero-point 对齐浮点最小值与整数 0，适用于分布明显偏向一侧的数据（如 ReLU 激活）。

此外，scale 可以按 **per-tensor**（整个张量一个 scale）或 **per-channel**（输出通道各一个 scale）计算，后者精度更高。

## 数学公式

### 对称量化

$$
Q_{max} = 2^{b-1} - 1
$$

$$
scale = \frac{\max(|x|)}{Q_{max}}
$$

$$
x_q = \text{clamp}\left(\text{round}\left(\frac{x}{scale}\right), -Q_{max}-1, Q_{max}\right)
$$

反量化：

$$
x \approx x_q \cdot scale
$$

### 非对称量化

$$
Q_{max} = 2^{b} - 1
$$

$$
scale = \frac{x_{max} - x_{min}}{Q_{max}}, \quad
zero\_point = \text{round}\left(\frac{-x_{min}}{scale}\right)
$$

$$
x_q = \text{clamp}\left(\text{round}\left(\frac{x}{scale}\right) + zero\_point, 0, Q_{max}\right)
$$

反量化：

$$
x \approx (x_q - zero\_point) \cdot scale
$$

## 时间/空间复杂度

- **量化过程**：$O(N)$ 遍历张量一次，$N$ 为元素数
- **反量化过程**：$O(N)$
- **空间节省**：
  - INT8 vs FP32：权重体积减少 **4x**，带宽减少 **4x**
  - INT4 vs FP32：权重体积减少 **8x**
- **推理加速**：在支持 INT8/INT4 的硬件（NVIDIA Tensor Core、ARM NEON、Intel AMX）上，整数矩阵乘通常比 FP16 快 **2~4 倍**

## 面试高频考点

1. **问题：对称量化与非对称量化的区别？分别适用于什么场景？**
   **答案**：对称量化假设数据分布关于 0 对称，公式简单、无需 zero-point，适合权重（通常近似对称）。非对称量化通过 zero-point 处理任意分布，适合激活值（如 ReLU 后全为非负，或经过 LayerNorm 后有偏置的分布）。

2. **问题：per-tensor 和 per-channel 量化的优劣？**
   **答案**：per-tensor 只有一个 scale，存储开销小，但不同输出通道的数值范围差异大时精度损失明显；per-channel 为每个输出通道单独计算 scale，精度更高，但存储略增（scale 数 = 输出通道数）。LLM 权重通常使用 per-channel 或 per-group（如 GPTQ 的 group_size=128）。

3. **问题：INT4 量化为什么比 INT8 更难保持精度？**
   **答案**：INT4 只有 16 个离散值，表示能力极其有限；微小 scale 估计误差就会被放大。因此 INT4 通常需要更精细的算法（如 GPTQ、AWQ、GGUF 的 Q4_K_M 等），通过分组量化、重要性加权、异常值隔离等手段补偿精度损失。

4. **问题：PTQ 与 QAT 的区别？**
   **答案**：PTQ（Post-Training Quantization）在模型训练完成后直接量化，无需重新训练，部署快但精度损失较大；QAT（Quantization-Aware Training）在训练过程中模拟量化误差（fake quant），让网络自适应低精度表示，精度更高但训练成本高。

## 代码解析

### symmetric_quantize / asymmetric_quantize

- 支持 `dim` 参数控制 per-tensor 或 per-channel；
- `clamp` 确保量化后的值不溢出目标位宽；
- 对 scale=0 做保护，避免除零错误。

### QuantizedLinear

- 教学级模拟：先量化权重保存 `weight_q` + `scale`，推理时反量化回浮点再做 `F.linear`；
- 真实部署中（如 `torch.ao.quantization`、`auto-gptq`、`llama.cpp`），会直接调用低精度矩阵乘 kernel，不会显式反量化。

### compute_quantization_error

- 用 MSE 评估量化-反量化后的数值重建误差，是量化算法调优的基础指标。

## 参考资料

- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877) — 经典量化论文
- [PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html) — PyTorch 官方量化文档
- [Hugging Face Quantization Concept Guide](https://huggingface.co/docs/transformers/quantization/concept_guide) — 对称/非对称量化概念
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) — 4-bit 权重量化经典方法
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978) — 保护显著权重通道的量化方法
