# Mixture of Experts (MoE)

本目录包含混合专家模型（Mixture of Experts）的核心实现，涵盖从基础 MoE 层到 Switch Transformer 的完整技术栈。

## 文件说明

| 文件 | 说明 |
|------|------|
| [moe_layer.py](moe_layer/moe_layer.py) | 稀疏门控 MoE 层（Top-K 专家选择） |
| [moe_layer.md](moe_layer/moe_layer.md) | MoE 层算法详解、面试考点 |
| [load_balance.py](load_balance/load_balance.py) | 负载均衡损失（Switch Transformer 风格 + Aux-Loss-Free） |
| [load_balance.md](load_balance/load_balance.md) | 负载均衡原理、数学公式、工程实践 |
| [switch_transformer.py](switch_transformer/switch_transformer.py) | Switch Transformer 完整实现（Top-1 + Capacity Factor） |
| [switch_transformer.md](switch_transformer/switch_transformer.md) | Switch Transformer 论文解读、面试高频问题 |

## 核心概念速查

### MoE vs 稠密模型

| 特性 | 稠密 Transformer | MoE Transformer |
|------|------------------|-----------------|
| 参数量 | $O(d_{model}^2)$ | $O(E \cdot d_{model} \cdot d_{ff})$ |
| 计算量（每 token） | $O(d_{model} \cdot d_{ff})$ | $O(K \cdot d_{model} \cdot d_{ff})$ |
| 激活参数量 | 100% | $K/E$（通常 1/8 ~ 2/8） |
| 训练难度 | 简单 | 需要负载均衡机制 |
| 推理通信 | 无 | all-to-all（专家并行时） |

### 关键超参数

- **num_experts (E)**：专家总数，常见 8、16、64。
- **top_k (K)**：每个 token 激活的专家数，Switch Transformer 取 1，Mixtral 取 2。
- **capacity_factor**：容量因子，控制每个专家最多处理多少 token。训练时 1.0~1.25，推理时 1.0。
- **aux_loss_coef**：负载均衡损失的权重系数，通常 0.01。

### 面试一句话总结

> MoE 通过“稀疏激活”实现模型容量的指数级扩展：总参数量随专家数线性增长，但每个输入只激活少数专家，计算量保持不变。

## 运行自测

```bash
cd d:\Code\Python_Workplace\HandwrittenZoo\moe
python moe_layer/moe_layer.py
python load_balance/load_balance.py
python switch_transformer/switch_transformer.py
```
