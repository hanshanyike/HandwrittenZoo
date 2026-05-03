# Fine-Tuning Methods (PEFT)

本目录包含大模型参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）的核心实现，涵盖从 LoRA 到 QLoRA、Prefix Tuning 的完整技术栈。

## 文件说明

| 文件 | 说明 |
|------|------|
| [lora.py](lora/lora.py) | LoRA 低秩适配实现（可注入、可合并） |
| [lora.md](lora/lora.md) | LoRA 算法详解、面试高频考点 |
| [qlora.py](qlora/qlora.py) | QLoRA 4-bit 量化 + LoRA（含双量化、分页优化器概念） |
| [qlora.md](qlora/qlora.md) | QLoRA 显存优化原理、NF4 量化、工程实践 |
| [prefix_tuning.py](prefix_tuning/prefix_tuning.py) | Prefix Tuning 前缀微调（含 MLP 重参数化） |
| [prefix_tuning.md](prefix_tuning/prefix_tuning.md) | Prefix Tuning 注意力干预机制、与 Prompt Tuning 对比 |

## 核心概念速查

### PEFT 方法对比

| 方法 | 修改位置 | 可训练参数 | 推理开销 | 多任务切换 | 典型场景 |
|------|---------|-----------|---------|-----------|---------|
| **Full Fine-tuning** | 全部权重 | 100% | 无 | 需完整模型副本 | 数据充足、算力充沛 |
| **LoRA** | 权重矩阵旁路 | 0.1%~1% | 无（可合并） | 切换 LoRA 权重 | 通用首选 |
| **QLoRA** | 4-bit 权重 + LoRA | 0.1%~1% | 反量化开销 | 切换 LoRA 权重 | 消费级 GPU 微调大模型 |
| **Prefix Tuning** | 每层 K/V 前缀 | 0.1%~0.5% | 序列长度增加 | 切换前缀 | 生成任务（NLG） |
| **Prompt Tuning** | 输入嵌入层 | < 0.01% | 序列长度增加 | 切换 prompt | 超大模型（>10B） |

### 超参数选择指南

#### LoRA / QLoRA

- **rank (r)**：4、8、16、32、64。任务越复杂，rank 越大。
- **alpha**：通常设为 `2 * r` 或 `r`。控制 LoRA 更新幅度。
- **dropout**：0.0~0.1。防止过拟合，小数据集建议加 dropout。
- **target_modules**：推荐 `["q_proj", "v_proj"]` 或全部线性层。

#### Prefix Tuning

- **prefix_len**：10~200。简单任务 10~20，复杂任务 50+。
- **use_reparam**：训练时建议 True（MLP 重参数化），推理时可缓存后丢弃。

### 面试一句话总结

> LoRA 通过低秩矩阵近似权重更新，QLoRA 在此基础上将基座量化到 4-bit，Prefix Tuning 则在每层 K/V 前添加可学习前缀——三者共同目标都是：用极少参数实现大模型的高效适配。

## 运行自测

```bash
cd d:\Code\Python_Workplace\HandwrittenZoo\fine_tuning
python lora/lora.py
python qlora/qlora.py
python prefix_tuning/prefix_tuning.py
```
