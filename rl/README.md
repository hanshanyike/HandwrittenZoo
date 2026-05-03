# RL — Reinforcement Learning for LLM Alignment

本目录收录大语言模型（LLM）人类反馈强化学习（RLHF）与偏好优化（Preference Optimization）的核心算法实现，覆盖从奖励建模到策略优化的完整链路。

## 算法清单

| 文件 | 算法 | 简介 | 面试热度 |
|------|------|------|----------|
| [reward_model.py](reward_model.py) + [reward_model.md](reward_model.md) | **Reward Model (Bradley-Terry)** | 将人类偏好建模为标量奖励，RLHF 阶段 2 的核心 | ★★★★★ |
| [ppo.py](ppo.py) + [ppo.md](ppo.md) | **Proximal Policy Optimization (PPO)** | 经典 RLHF 阶段 3 算法，通过 clip 机制稳定策略更新 | ★★★★★ |
| [dpo.py](dpo.py) + [dpo.md](dpo.md) | **Direct Preference Optimization (DPO)** | 无需奖励模型和在线 RL，直接优化偏好数据（2024-2025 面试焦点） | ★★★★★ |
| [grpo.py](grpo.py) + [grpo.md](grpo.md) | **Group Relative Policy Optimization (GRPO)** | DeepSeek-R1 核心算法，去价值网络、组相对优势（2025 最热门） | ★★★★★ |

## 面试焦点

### 2024-2025 年高频考点
1. **DPO 与 PPO 的对比**：DPO 为什么不需要奖励模型？各自的优缺点和适用场景？
2. **GRPO 的创新点**：相比 PPO 去掉了什么？组相对优势如何计算？为什么能省显存？
3. **Bradley-Terry 模型**：如何从偏好概率推导出交叉熵损失？与 Elo 评分系统的关系？
4. **Clip 机制**：PPO/GRPO 中的 `min(ratio * A, clip(ratio) * A)` 为什么取 `min` 而不是 `max`？
5. **KL 散度惩罚**：在 RLHF 中为什么必须加 KL 惩罚？不加会导致什么问题？

### 算法演进路线
```
SFT -> Reward Model (BT) -> PPO + Critic + RM  (经典 RLHF，3 阶段)
SFT -> DPO               -> 无需 RM，离线偏好优化  (2023 简化路线)
SFT -> GRPO              -> 无需 Critic，在线组优化  (2024 DeepSeek 路线)
```

## 快速开始

所有 `.py` 文件均包含自测模块，可直接运行验证：

```bash
python reward_model.py
python ppo.py
python dpo.py
python grpo.py
```

## 依赖

- Python >= 3.8
- PyTorch >= 1.12

无额外第三方库（如 transformers、trl），所有算法从零实现，便于理解核心原理。
