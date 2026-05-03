# Reward Model (Bradley-Terry)

## 算法简介
Reward Model（奖励模型）是 RLHF 流程的第二阶段，负责将人类对模型输出的偏好（chosen vs rejected）建模为标量奖励信号，为后续强化学习阶段提供优化目标。

## 核心思想
1. **Bradley-Terry 假设**：假设存在一个标量奖励函数 $r(x, y)$，使得人类偏好满足概率形式
   $$P(y_w \succ y_l \mid x) = \sigma\big(r(x, y_w) - r(x, y_l)\big)$$
   其中 $\sigma$ 为 sigmoid 函数，$y_w$ 为被偏好的回答，$y_l$ 为不被偏好的回答。

2. **端到端训练**：无需显式让人类打分，只需收集成对偏好数据，通过最大似然估计即可训练奖励模型。

3. **标量输出**：训练完成后，模型对任意单个 response 输出一个标量 reward，可直接用于 PPO/GRPO 的奖励信号。

## 数学公式

### Bradley-Terry 偏好概率
$$P(y_w \succ y_l \mid x) = \frac{\exp\big(r_\theta(x, y_w)\big)}{\exp\big(r_\theta(x, y_w)\big) + \exp\big(r_\theta(x, y_l)\big)} = \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big)$$

### 训练损失（负对数似然）
$$\mathcal{L}_{\text{BT}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \Big[ \log \sigma\big(r_\theta(x, y_w) - r_\theta(x, y_l)\big) \Big]$$

### 带标签平滑的变体
$$\mathcal{L}_{\text{smooth}} = -(1 - \alpha) \log \sigma(\Delta r) - \alpha \log \sigma(-\Delta r)$$
其中 $\Delta r = r_\theta(x, y_w) - r_\theta(x, y_l)$，$\alpha$ 为标签平滑系数。

## 时间/空间复杂度
- **时间复杂度**：$O(B \cdot L \cdot d^2)$，$B$ 为 batch size，$L$ 为序列长度，$d$ 为隐藏维度
- **空间复杂度**：$O(d^2)$，主要来自奖励头的线性层
- **与替代方案对比**：
  - 相比回归式奖励模型（直接拟合人类打分），BT 模型只需偏好对，数据收集成本更低。
  - 相比 Pointwise 模型，Pairwise 形式对噪声更鲁棒，且天然兼容 RLHF 流程。

## 面试高频考点

1. **为什么 Reward Model 最后一层不加 softmax？**
   **答案**：Reward Model 输出的是标量奖励值（实数），用于比较大小和计算 sigmoid 概率。加 softmax 会将输出限制为概率分布，失去标量可比性，无法直接用于 PPO 的奖励信号。

2. **Bradley-Terry 模型和 Elo 评分系统有什么关系？**
   **答案**：BT 模型是 Elo 系统的概率基础。Elo 的期望胜率公式 $E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$ 本质上是 logistic 形式的 BT 模型，两者都基于选手/物品的相对强度建模偏好概率。

3. **Reward Hacking 是什么？如何缓解？**
   **答案**：Reward Hacking 指策略模型找到奖励模型的漏洞，生成高奖励但低质量的输出。缓解方法包括：
   - 在奖励中加入 KL 散度惩罚，限制策略偏离参考模型；
   - 使用多轮迭代训练，持续更新奖励模型；
   - 引入规则-based 的硬约束（如格式检查、安全过滤）。

4. **为什么用最后一个 token 而不是 CLS token 作为句子表示？**
   **答案**：对于生成任务（如 GPT 类模型），最后一个有效 token 聚合了前面所有 token 的上下文信息，比 CLS token 更能代表整个序列的语义。CLS token 在 BERT 的预训练目标中优化，对生成任务并非最优。

## 代码解析

### RewardModel 类
```python
self.reward_head = nn.Sequential(
    nn.Dropout(dropout),
    nn.Linear(hidden_size, 1, bias=False)
)
```
奖励头是一个简单的线性投影，将编码器的隐藏状态映射为标量。不使用 bias 可减少一个自由度，让模型更关注相对差值。

### 取最后一个有效 token
```python
seq_lengths = attention_mask.sum(dim=1) - 1
pooled = last_hidden[torch.arange(batch_size), seq_lengths]
```
通过 attention_mask 计算实际长度，避免取到 pad token 的向量，保证句子表示的准确性。

### Bradley-Terry 损失
```python
diff = reward_chosen - reward_rejected
loss = -F.logsigmoid(diff)
```
利用 PyTorch 的 `logsigmoid` 保证数值稳定性，避免先 sigmoid 再 log 导致的下溢。

## 参考资料
- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [RLHF Workflow: From Reward Modeling to Online RLHF](https://arxiv.org/abs/2405.07863)
- [Bradley-Terry Model (Wikipedia)](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)
- [Reward Modeling Part 1: Bradley-Terry Model (RLHFlow Blog)](https://rlhflow.github.io/posts/2024-03-23-bradley-terry-reward-model/)
