# Proximal Policy Optimization (PPO)

## 算法简介
PPO（近端策略优化）是 OpenAI 于 2017 年提出的策略梯度算法，通过引入 clipped surrogate objective 限制策略更新幅度，解决了传统策略梯度方法训练不稳定的问题。在 RLHF 流程中，PPO 是第三阶段的核心算法，负责根据 Reward Model 的反馈优化语言模型策略。

## 核心思想
1. **重要性采样复用数据**：利用旧策略 $\pi_{\text{old}}$ 采集一批数据，通过重要性采样比率 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ 复用该数据进行多轮梯度更新。
2. **Clipped Surrogate 约束**：将 $r_t(\theta)$ 裁剪到 $[1-\epsilon, 1+\epsilon]$ 区间，防止策略因单步更新过大而崩溃（policy collapse）。
3. **KL 散度惩罚**：在奖励中显式加入与参考模型的 KL 散度，防止策略模型偏离原始预训练分布过远，缓解 reward hacking。

## 数学公式

### Clipped Surrogate Objective
$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \Big[ \min\big( r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \, A_t \big) \Big]$$
其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$，$A_t$ 为优势函数估计。

### 完整 PPO 目标（含 value loss 和 entropy bonus）
$$L^{\text{PPO}}(\theta) = \mathbb{E}_t \Big[ L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 H\big(\pi_\theta(\cdot|s_t)\big) \Big]$$
- $L^{\text{VF}}$：value function 的均方误差损失
- $H(\pi_\theta)$：策略熵，鼓励探索
- $c_1, c_2$：超参数权重

### GAE 优势估计
$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$$
其中 TD 残差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。

### KL 惩罚（RLHF 场景）
$$r_t^{\text{KL}} = r_t^{\text{RM}} - \beta \cdot D_{\text{KL}}\big(\pi_\theta(\cdot|s_t) \,\|\, \pi_{\text{ref}}(\cdot|s_t)\big)$$

## 时间/空间复杂度
- **时间复杂度**：$O(B \cdot L \cdot d^2)$ 每次更新 epoch
- **空间复杂度**：$O(B \cdot L)$ 用于存储 rollout 数据（log probs、advantages、values）
- **与替代方案对比**：
  - 相比 TRPO（Trust Region Policy Optimization），PPO 无需计算 Fisher 信息矩阵的二阶优化，实现更简单、训练更快。
  - 相比 REINFORCE，PPO 通过 clipped objective 和 GAE 大幅降低方差，样本效率更高。

## 面试高频考点

1. **PPO 的 clip 机制为什么能防止策略崩溃？**
   **答案**：当 advantage $A_t > 0$ 时，模型倾向于增大该动作的概率。若不加限制，ratio $r_t(\theta)$ 可能变得极大，导致策略过度优化某个动作而忽略其他。clip 将 $r_t$ 限制在 $[1-\epsilon, 1+\epsilon]$，使得超出范围的部分不再贡献梯度，相当于给策略更新加了"安全护栏"。

2. **为什么要用 GAE 而不是简单的 TD 误差？**
   **答案**：简单 TD 误差（$\delta_t$）偏差小但方差大；蒙特卡洛回报（$G_t$）无偏但方差极大。GAE 通过参数 $\lambda \in [0,1]$ 在两者之间插值：$\lambda=0$ 退化为 TD（低方差高偏差），$\lambda=1$ 退化为 MC（高方差无偏）。实际中 $\lambda=0.95$ 可在偏差和方差间取得良好平衡。

3. **PPO 中的 KL 惩罚和 clip 是什么关系？**
   **答案**：两者都是约束策略更新幅度的机制，但作用层面不同：
   - **Clip** 在单样本级别限制 importance ratio，直接裁剪目标函数。
   - **KL Penalty** 在分布级别约束整个策略与参考模型的偏离，通过修改奖励信号间接影响梯度。
   实践中两者常同时使用，clip 提供硬约束，KL 提供软约束。

4. **PPO 在 LLM 场景下与游戏 RL 有什么不同？**
   **答案**：
   - **动作空间**：游戏 RL 动作空间小且固定；LLM 的动作空间是词表（数万 token），每个动作是一个离散采样。
   - **序列决策**：LLM 的每个 token 是一个时间步，序列长度可达数千，需要逐 token 计算 reward 和 advantage。
   - **参考模型**：LLM RLHF 必须引入参考模型和 KL 惩罚，防止模型输出乱码或偏离人类语言分布；游戏 RL 通常无此需求。

## 代码解析

### GAE 计算
```python
for t in reversed(range(seq_len)):
    delta = rewards[:, t] + gamma * next_value * masks[:, t] - values[:, t]
    last_gae = delta + gamma * lam * last_gae * masks[:, t]
    advantages[:, t] = last_gae
```
从序列末尾向前递推，利用动态规划高效计算多步优势估计。`masks` 用于处理变长序列的 pad 位置。

### Clipped Surrogate
```python
ratio = torch.exp(new_log_probs - old_log_probs)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
policy_loss = -torch.min(surr1, surr2)
```
`torch.min` 是关键：当 advantage 为正时，若 ratio 超过 $1+\epsilon$，clip 后的 surr2 更小，梯度不再推动 ratio 继续增大；当 advantage 为负时，若 ratio 低于 $1-\epsilon$，clip 阻止 ratio 继续减小。

### Value Loss 裁剪
```python
value_pred_clipped = old_values + torch.clamp(new_values - old_values, -clip, clip)
value_loss = torch.max(F.mse_loss(new_values, returns), F.mse_loss(value_pred_clipped, returns))
```
借鉴策略 clip 的思想，对 value prediction 也做裁剪，防止 Critic 的剧烈更新破坏训练稳定性。

## 参考资料
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [Training language models to follow instructions with human feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438)
- [PPO Implementation Details (Costa Huang Blog)](https://costa.sh/blog-the-32-implementation-details-of-ppo.html)
