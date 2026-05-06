# Speculative Decoding（投机解码）

## 算法简介

投机解码是一种 2022-2023 年兴起的 LLM 推理加速技术，通过"小模型猜测 + 大模型验证"的策略，将串行自回归解码转化为准并行解码。该技术的核心优势在于：**无损加速**——不需要量化、不需要剪枝、不需要修改模型权重，却能实现 2-4 倍的吞吐提升，同时保证输出分布与原始大模型完全一致。

## 核心思想

### 1. 动机：自回归解码的瓶颈

LLM 的自回归生成面临一个根本性问题：**串行性**。每个 token 的生成都依赖前一个 token 的输出，导致 GPU 的并行计算能力在 decode 阶段严重闲置：

- **Prefill 阶段**：处理输入 prompt，GPU 可充分并行，吞吐量高
- **Decode 阶段**：逐 token 生成，每步仅做一次小型矩阵乘法，GPU 利用率低

实测数据显示，72B 参数模型在 A100 上：
- Prefill 可达 3000+ tokens/s
- Decode 通常只有 20-40 tokens/s

**100 倍的性能落差**就是投机解码要解决的问题。

### 2. 两阶段工作流程

```
草稿阶段（Draft）              验证阶段（Verify）
┌─────────────────┐           ┌─────────────────────────┐
│  草稿模型（小）   │           │   目标模型（大）         │
│                 │           │                         │
│  生成 K 个候选   │   ──►    │  一次前向传播验证 K 个位置 │
│  自回归、速度快  │           │  同时计算 K 个 logits   │
└─────────────────┘           └─────────────────────────┘
                                          │
                                          ▼
                               ┌─────────────────────────┐
                               │  接受/拒绝决策            │
                               │  匹配则接受，不匹配则重采样 │
                               └─────────────────────────┘
```

**关键洞察**：大模型一次前向传播的成本和生成 1 个 token 几乎相同（因为是 memory-bound），但这次前向传播可以同时算出 K 个位置的概率分布。

### 3. 接受率与加速比

设草稿模型每个 token 的接受概率为 $\alpha$，则生成 K 个候选 token 时，期望接受长度为：

$$E[\text{accepted\_tokens}] = 1 + \alpha + \alpha^2 + \alpha^3 + ... = \frac{1 - \alpha^K}{1 - \alpha}$$

| $\alpha$ (接受率) | K=3 | K=5 | K=8 |
|-------------------|-----|-----|-----|
| 0.6 | 1.64 | 1.95 | 2.24 |
| 0.7 | 2.00 | 2.59 | 3.22 |
| 0.8 | 2.44 | 3.36 | 4.44 |
| 0.9 | 3.00 | 4.10 | 5.59 |

**加速比** ≈ $1 + (\alpha - 1) \times K / (1 - \alpha)$，当 $\alpha = 0.8, K = 5$ 时，加速比约 **2.5x**。

## 数学公式

### 接受/拒绝采样

设草稿模型在位置 $i$ 生成的 token 为 $t_i$，其概率为 $q(t_i)$；目标模型在同一位置的概率为 $p(t_i)$。

验证规则（Nezhaov et al., 2022）：

$$
\text{接受} \iff r < \frac{p(t_i)}{q(t_i)}
$$

其中 $r \sim \text{Uniform}(0, 1)$。

如果拒绝，则从目标模型的条件分布 $p(\cdot | \text{context})$ 重新采样。

### 无损保证

投机解码的输出分布与纯目标模型生成完全一致。证明基于以下恒等式：

$$p(\text{accept } i) \cdot p(\text{output } | \text{accept}) = p(\text{target output})$$

这意味着：**没有质量损失**，加速来自更好地利用 GPU 并行性，而非降低精度。

## 时间/空间复杂度

- **时间复杂度**：$O(K \cdot d_{\text{draft}})$ 草稿生成 + $O(1)$ 验证（目标模型一次前向 ≈ 单步 decode）
- **空间复杂度**：额外 $O(K \cdot d_{\text{draft}})$ 显存存储草稿模型的 K 个位置
- **关键优势**：草稿模型可以比目标模型小 8-10 倍，因此草稿阶段的计算开销可忽略

## 面试高频考点

1. **问题**：投机解码为什么能加速？
   **答案**：大模型推理是 memory-bound，单次前向传播计算 K 个 token 和 1 个 token 的时间几乎相同。草稿模型快速生成候选后，大模型一次前向可验证 K 个位置，实现准并行。

2. **问题**：投机解码是否有质量损失？
   **答案**：**没有**。接受/拒绝采样保证输出分布与纯大模型完全一致。被拒绝的 token 从目标模型真实分布重新采样。

3. **问题**：草稿模型如何选择？
   **答案**：三条原则：① 同系列优先（tokenizer 一致，接受率高 10-15%）；② 大小比例 8:1~10:1；③ 任务匹配（代码生成用 CodeLlama）。

4. **问题**：投机解码的适用场景？
   **答案**：适合 **batch 推理、高并发、长输出** 场景。不适合：超短序列（启动开销不划算）、高温度采样（接受率低）、显存极度紧张（无法同时加载两个模型）。

5. **问题**：除了双模型方案，还有哪些实现方式？
   **答案**：① **Self-Speculative / Medusa**：目标模型头部加预测头，无额外显存开销；② **Draft-from-Attention**：利用上下文重复模式直接复制候选。

## 代码解析

### 验证与接受逻辑

```python
def _verify_and_accept(self, draft_tokens, draft_probs, target_logits):
    target_probs = F.softmax(target_logits, dim=-1)

    accepted = torch.ones(batch_size, K, dtype=torch.bool)
    for i in range(K):
        draft_prob_i = draft_probs[:, i].gather(1, draft_tokens[:, i:i+1])
        target_prob_i = target_probs[:, i].gather(1, draft_tokens[:, i:i+1])

        # 接受概率 = min(1, target/draft)
        acceptance_ratio = (target_prob_i / (draft_prob_i + 1e-10)).clamp(0, 1)

        r = torch.rand(batch_size, device=self.device)
        accepted[:, i] = r < acceptance_ratio

        # 遇到第一个拒绝，后续全部截断
        if not accepted[:, i].all():
            accepted[:, i + 1:] = False
            break

    return accepted_tokens, n_accepted, n_rejected
```

**关键点**：
- 接受率是 $\min(1, p/q)$，确保概率比值大于 1 时一定接受
- 从第一个位置开始逐个验证，一旦拒绝，后续全部截断
- 被拒绝位置从目标模型重新采样

### 整体生成流程

```python
def generate(self, prompt_ids, max_new_tokens):
    generated = prompt_ids.clone()

    for step in range(max_new_tokens):
        # 1. 草稿阶段：快速生成 K 个候选
        draft_tokens = []
        for k in range(self.max_draft_len):
            logits = self.draft(generated)
            next_token = self._sample_token(logits[:, -1])
            draft_tokens.append(next_token)
            generated = torch.cat([generated, next_token], dim=1)

        # 2. 验证阶段：目标模型一次前向
        target_logits = self.target(generated)

        # 3. 接受/拒绝
        accepted, n_accepted, n_rejected = self._verify_and_accept(...)
        if n_rejected > 0:
            break

    return generated
```

## 参考资料

- [Fast Speculative Decoding paper](https://arxiv.org/abs/2211.17192)
- [Medusa: Simple Speculative Decoding](https://arxiv.org/abs/2401.10774)
- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decoding.html)
