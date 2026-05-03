# Direct Preference Optimization (DPO)

## 算法简介
DPO（直接偏好优化）是 Rafailov 等人于 2023 年提出的 LLM 对齐算法，它绕过了传统 RLHF 中独立的奖励模型和复杂的 PPO 强化学习阶段，直接在人类偏好数据上进行监督学习，实现了"无需奖励模型的 RLHF"。

## 核心思想
1. **最优策略的闭式解**：从 Bradley-Terry 模型出发，可以证明最优策略满足
   $$\pi^*(y|x) \propto \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$
   其中 $\beta$ 为温度系数，控制策略偏离参考模型的程度。

2. **消去奖励模型**：将上式变形，得到奖励函数的显式表达
   $$r(x,y) = \beta \log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$
   代入 Bradley-Terry 损失后，$Z(x)$ 被消去，最终只剩下策略模型和参考模型的对数概率比。

3. **转化为二分类问题**：DPO 的最终形式是一个简单的二元交叉熵损失，直接比较 chosen 和 rejected 的隐式奖励差值。

## 数学公式

### DPO 损失函数
$$\mathcal{L}_{\text{DPO}}(\theta; \mathcal{D}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

### 隐式奖励定义
$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

### 带标签平滑的变体
$$\mathcal{L}_{\text{smooth}} = -(1-\alpha) \log \sigma(\Delta \hat{r}) - \alpha \log \sigma(-\Delta \hat{r})$$
其中 $\Delta \hat{r} = \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)$，$\alpha$ 为标签平滑系数。

## 时间/空间复杂度
- **时间复杂度**：$O(B \cdot L \cdot d^2)$ 每步（与一次语言模型前向传播同阶）
- **空间复杂度**：$O(B \cdot L)$ 用于存储 chosen/rejected 的 log probs
- **与替代方案对比**：
  - 相比 PPO+RM 的 RLHF：DPO 无需训练奖励模型，无需在线采样 rollout，训练更稳定、实现更简单。
  - 相比 SFT：DPO 直接利用偏好信号，能学到人类偏好的相对排序，而非仅仅拟合 chosen 文本。
  - 局限性：DPO 对偏好数据质量要求高，数据分布外泛化能力可能弱于 PPO；且无法利用在线奖励信号。

## 面试高频考点

1. **DPO 为什么不需要奖励模型？**
   **答案**：DPO 从 RLHF 的最优策略闭式解出发，将奖励函数 $r(x,y)$ 用策略模型与参考模型的对数概率比显式表示。代入 Bradley-Terry 损失后，奖励模型被解析消去，最终损失只涉及两个语言模型的概率比，因此无需单独训练奖励模型。

2. **DPO 中的 β 参数有什么作用？**
   **答案**：$\beta$ 是温度系数，控制策略偏离参考模型的程度：
   - $\beta \to 0$：策略极度偏离参考模型，严格追求偏好对齐，但可能过拟合或产生乱码。
   - $\beta \to \infty$：策略退化为参考模型，完全不对齐。
   典型值 0.1~0.5，需要在偏好强度和分布保持之间权衡。

3. **DPO 和 PPO 的优缺点对比？**
   **答案**：
   | 维度 | DPO | PPO |
   |------|-----|-----|
   | 是否需要 RM | 否 | 是 |
   | 训练稳定性 | 高（监督学习） | 中（RL 方差大） |
   | 计算开销 | 低（离线数据） | 高（在线采样） |
   | 数据需求 | 高质量偏好对 | 偏好对 + 在线交互 |
   | 分布外泛化 | 较弱 | 较强 |
   | 工程复杂度 | 低 | 高 |

4. **DPO 的隐式奖励在什么情况下会失效？**
   **答案**：当参考模型与策略模型的概率分布差异过大时（如经过多轮 DPO 迭代后），隐式奖励的数值可能变得极不稳定，导致训练崩溃。此外，如果偏好数据本身存在矛盾或噪声，DPO 会直接拟合这些噪声，因为没有奖励模型做平滑。

5. **IPO（Identity Preference Optimization）与 DPO 的区别？**
   **答案**：IPO 是 DPO 的改进版，将 DPO 中的 log-sigmoid 替换为平方损失，避免了 DPO 在强偏好数据下过度自信的问题。IPO 的损失形式为 $(\hat{r}_\theta(x,y_w) - \hat{r}_\theta(x,y_l) - \frac{1}{2\beta})^2$，对异常值更鲁棒。

## 代码解析

### 序列对数概率计算
```python
log_probs_all = F.log_softmax(logits, dim=-1)
log_probs = torch.gather(log_probs_all, dim=2, index=labels.unsqueeze(-1)).squeeze(-1)
log_probs = (log_probs * attention_mask.float()).sum(dim=1)
```
先对整个词表做 log_softmax，再用 `gather` 取出目标 token 的 log prob，最后按有效长度求和。这是语言模型计算序列似然的标准做法。

### 隐式奖励与损失
```python
chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
```
`policy_logps - reference_logps` 即为对数概率比，乘以 $\beta$ 得到隐式奖励。`logsigmoid` 保证数值稳定性，避免先 exp 再 log 的下溢问题。

### 标签平滑
```python
loss = (-F.logsigmoid(logits_diff) * (1 - alpha)
        - F.logsigmoid(-logits_diff) * alpha).mean()
```
将硬标签 1 替换为 $1-\alpha$，防止模型对噪声偏好过度自信，提升泛化能力。

## 参考资料
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
- [A General Theoretical Paradigm to Understand Learning from Human Preferences (Azar et al., 2023)](https://arxiv.org/abs/2310.12036)
- [IPO: Identity Preference Optimization (Azar et al., 2024)](https://arxiv.org/abs/2310.12036)
- [DPO Implementation in torchtune (Meta)](https://pytorch.org/torchtune/main/_modules/torchtune/rlhf/loss/dpo.html)
