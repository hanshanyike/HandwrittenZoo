# Group Relative Policy Optimization (GRPO)

## 算法简介
GRPO（组相对策略优化）是 DeepSeek 团队于 2024 年在 DeepSeekMath 论文中提出的强化学习算法，被 DeepSeek-R1 采用为核心训练方法。它通过"组内相对奖励"机制完全替代了 PPO 中的价值网络（Critic），在大幅降低显存占用的同时提升了训练稳定性。

## 核心思想
1. **去价值网络**：PPO 需要维护一个与策略模型同等规模的价值模型（Critic），显存开销巨大。GRPO 完全去掉 Critic，改为对同一 prompt 采样 $G$ 个回答，用组内奖励的统计量计算优势。
2. **组内相对优势**：同一组内的 $G$ 个回答共享同一个 baseline（组均值），优势仅取决于该回答在组内的相对表现。这天然消除了不同 prompt 之间的奖励尺度差异，无需额外归一化。
3. **PPO 风格裁剪**：保留 PPO 的 clipped surrogate objective，限制单步策略更新幅度，保证训练稳定。
4. **逐 token KL 惩罚**：对每个生成 token 计算与参考模型的 KL 散度，防止策略偏离原始分布。

## 数学公式

### 组相对优势
$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G) + \epsilon}$$
其中 $r_i$ 为第 $i$ 个输出的奖励，优势对所有 token 相同（per-output advantage）。

### GRPO 目标函数（per token）
$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G \cdot |o|} \sum_{i=1}^G \sum_{t=1}^{|o_i|} \Big[ \min\big( r_t(\theta) \hat{A}_i, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \, \hat{A}_i \big) - \beta \cdot D_{\text{KL}}\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big) \Big]$$
其中 $r_t(\theta) = \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\text{old}}(o_{i,t}|q, o_{i,<t})}$ 为重要性采样比率。

### KL 散度（逐 token）
$$D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}}) = \sum_{v} \pi_\theta(v) \big( \log \pi_\theta(v) - \log \pi_{\text{ref}}(v) \big)$$

## 时间/空间复杂度
- **时间复杂度**：$O(G \cdot B \cdot L \cdot d^2)$ 每步（$G$ 为组大小）
- **空间复杂度**：$O(G \cdot B \cdot L)$ 用于存储组 rollout 数据
- **与替代方案对比**：
  - 相比 PPO：省去同等规模的价值模型，显存节省约 40%~50%；无需训练 Critic，实现更简单。
  - 相比 DPO：GRPO 是在线 RL 算法，可利用规则奖励、编译器反馈等在线信号；DPO 仅适用于离线偏好数据。
  - 相比 REINFORCE：GRPO 通过组内 baseline 和 clip 机制大幅降低方差，训练更稳定。

## 面试高频考点

1. **GRPO 相比 PPO 最大的改进是什么？**
   **答案**：GRPO 完全去掉了价值网络（Critic）。PPO 需要维护一个与策略模型同等规模的 Critic 来估计状态价值，显存和计算开销巨大。GRPO 通过对同一 prompt 采样 $G$ 个输出，用组内奖励的均值和标准差作为 baseline 和归一化因子，既省去了 Critic，又天然消除了不同 prompt 间的奖励尺度差异。

2. **为什么 GRPO 的 advantage 对所有 token 都一样？**
   **答案**：GRPO 的优势是基于整个输出（output-level）的奖励计算的，而非逐 token 的奖励。在 LLM 生成场景中，通常只在序列末尾获得一个标量奖励（如答案正确性、格式合规性），因此该输出内所有 token 共享同一个 advantage。这与 PPO 中逐 token 分配 reward 的场景不同。

3. **GRPO 中的 KL 惩罚和 PPO 中的 KL 惩罚有什么区别？**
   **答案**：两者形式相同，都是 $D_{\text{KL}}(\pi_\theta \|\| \pi_{\text{ref}})$，但施加位置不同：
   - PPO（InstructGPT 风格）：将 KL 作为 reward shaping 的一部分，$r^{\text{KL}} = r^{\text{RM}} - \beta \cdot D_{\text{KL}}$，再输入 GAE。
   - GRPO（DeepSeek 风格）：将 KL 直接放入目标函数中作为正则项，与 clipped surrogate 同时优化，更稳定。

4. **GRPO 的 group_size 如何选取？**
   **答案**：group_size $G$ 越大，组内 baseline 估计越准确，方差越低，但显存和采样开销线性增长。典型值：
   - 小规模实验：$G=4 \sim 8$
   - 生产环境（如 DeepSeek-R1）：$G=16 \sim 64$
   需在估计精度和计算成本间权衡。

5. **GRPO 适合什么类型的奖励？**
   **答案**：GRPO 特别适合稀疏、延迟的标量奖励，如：
   - 数学问题：答案正确性（0/1 奖励）
   - 代码生成：编译/测试通过率
   - 格式约束：输出是否符合指定 XML/JSON 格式
   这类奖励天然适合 output-level 的组内比较，不适合需要 dense reward 的场景（如逐 token 的困惑度奖励）。

## 代码解析

### 组相对优势计算
```python
mean_rewards = rewards.mean(dim=1, keepdim=True)
std_rewards = rewards.std(dim=1, keepdim=True)
advantages = (rewards - mean_rewards) / (std_rewards + eps)
```
沿 group 维度计算均值和标准差，实现组内归一化。`eps` 防止标准差为 0 时除零（当组内所有奖励相同时）。

### 逐 token KL 散度
```python
policy_log_probs = F.log_softmax(policy_logits, dim=-1)
ref_log_probs = F.log_softmax(ref_logits, dim=-1)
kl = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)
```
严格按定义计算 $D_{\text{KL}}(p \| q) = \sum p (\log p - \log q)$，不是对称的。GRPO 使用 $\pi_\theta$ 作为前向分布，惩罚策略模型主动偏离参考模型的行为。

### GRPO 损失组装
```python
surr1 = ratio * advantages_expanded
surr2 = torch.clamp(ratio, 1-eps, 1+eps) * advantages_expanded
surrogate = torch.min(surr1, surr2)
per_token_loss = -(surrogate - beta * kl_penalty)
```
与 PPO 的 clip 逻辑完全一致，但优势是 per-output 而非 per-token。KL 惩罚直接放入目标函数，而非通过 reward shaping。

## 参考资料
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., 2024)](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (DeepSeek-AI, 2025)](https://arxiv.org/abs/2501.12948)
- [GRPO Trainer (Hugging Face TRL)](https://github.com/huggingface/trl/blob/main/docs/source/grpo_trainer.md)
- [GRPO: Eliminating the Value Network (Vitor Sousa Blog)](https://www.vitorsousa.com/blog/grpo/)
