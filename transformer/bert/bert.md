# BERT (Bidirectional Encoder Representations from Transformers)

## 算法简介

BERT 是 Google 于 2018 年提出的基于 Transformer Encoder 的双向预训练语言模型（Devlin et al., 2019）。与 GPT（单向）和 ELMo（浅层双向）不同，BERT 通过深层 Transformer Encoder 实现真正的深层双向上下文建模，在多项 NLP 任务上取得突破性进展，开创了"预训练 + 微调"（Pre-training + Fine-tuning）范式。

## 核心思想

1. **深层双向表示**：使用 Transformer Encoder（非 Decoder），通过 Self-Attention 让每个 token 同时看到左右两侧的所有上下文信息，而非像 GPT 那样只能看到左侧。
2. **MLM（Masked Language Modeling）**：随机遮罩输入序列中 15% 的 token，让模型根据上下文预测被遮罩的词。这迫使模型学习真正的双向语义，而非简单的从左到右语言模型。
3. **NSP（Next Sentence Prediction）**：输入两个句子，预测第二个句子是否是第一个句子的真实下一句。帮助模型理解句子间关系和篇章结构。
4. **预训练 + 微调**：先在大规模无标注语料上预训练，再针对下游任务（分类、NER、QA 等）添加简单输出层进行微调。

## 数学公式

### MLM 损失函数

$$
\mathcal{L}_{\text{MLM}} = -\mathbb{E}_{x \in \mathcal{M}} \log P(x \mid \hat{x})
$$

其中 $\mathcal{M}$ 是被遮罩的位置集合，$\hat{x}$ 是遮罩后的输入。

### NSP 损失函数

$$
\mathcal{L}_{\text{NSP}} = -\mathbb{E}_{(A,B)} \left[ y \log P(\text{IsNext}) + (1-y) \log P(\text{NotNext}) \right]
$$

总损失：$\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$

### 三重嵌入

$$
E = E_{\text{token}} + E_{\text{position}} + E_{\text{segment}}
$$

BERT 使用**可学习的位置嵌入**（Learned Positional Embedding），而非 Transformer 的正弦编码。

## 时间/空间复杂度

| 项目 | 复杂度 | 说明 |
|------|--------|------|
| Self-Attention | $O(n^2 \cdot d)$ | 与 Transformer Encoder 相同 |
| FFN | $O(n \cdot d \cdot d_{ff})$ | $d_{ff} = 4d$ |
| 总参数量（Base） | 110M | $V \cdot d + N \cdot (4d^2 + 2d \cdot d_{ff})$ |
| 总参数量（Large） | 340M | $d=1024, N=24$ |

**与 GPT 对比**：BERT 使用双向 Attention，无法直接用于文本生成（没有 Causal Mask）；GPT 使用单向 Attention，天然适合自回归生成。

## 面试高频考点

### Q1：BERT 的预训练任务 MLM 具体是怎么做的？

**答案**：
1. 随机选择输入序列中 15% 的 token 作为预测目标。
2. 被选中的 token 中：
   - 80% 概率替换为 `[MASK]` 特殊 token；
   - 10% 概率替换为词表中的随机 token；
   - 10% 概率保持不变。
3. 模型输出对应位置的词表维度 logits，与原始 token 计算交叉熵损失。
4. 这样做的好处是：模型不会只在 `[MASK]` 出现时学习，而是对所有 token 都保持鲁棒的表示。

### Q2：为什么 BERT 使用可学习的位置嵌入，而不是正弦位置编码？

**答案**：
- 原始论文实验发现可学习的位置嵌入效果与正弦编码相当，但更简单直接。
- 可学习嵌入可自动适应特定任务的位置模式。
- 现代大模型（如 GPT、LLaMA）中，RoPE 等旋转位置编码因更好的外推性而取代两者，成为主流。

### Q3：BERT 的 `[CLS]` 和 `[SEP]` 分别有什么作用？

**答案**：
- `[CLS]`：放在输入序列开头，其最终隐藏状态经过 Pooler 后作为整个序列的聚合表示，用于分类任务（如 NSP、情感分类）。
- `[SEP]`：分隔符，用于区分句子 A 和句子 B（如 NSP 任务中的两个句子），也标记序列结束。

### Q4：为什么 BERT 不适合做文本生成？

**答案**：
- BERT 使用双向 Self-Attention，每个位置都能看到两侧所有 token，没有 Causal Mask。
- 文本生成需要自回归地逐个预测下一个 token，必须保证当前位置只能看到已生成的内容。
- 因此生成任务通常使用 GPT 等 Decoder-Only 模型，而非 BERT 等 Encoder-Only 模型。

### Q5：BERT 和 GPT 在预训练目标上的本质区别是什么？

**答案**：
- **BERT**：MLM 是降噪自编码器（Denoising Autoencoder）目标，从被污染的输入中恢复原始信息，学习双向表示。
- **GPT**：标准语言模型（Autoregressive LM）目标，最大化 $P(x_t | x_{<t})$，学习单向生成能力。
- 本质区别：BERT 是"看完上下文填空"，GPT 是"根据前文续写"。

## 代码解析

### 三重嵌入（Token + Position + Segment）

```python
embeddings = (
    self.token_emb(input_ids)
    + self.pos_emb(pos_ids)
    + self.seg_emb(segment_ids)
)
```

BERT 的创新之一：除了词嵌入和位置嵌入外，还增加了段嵌入（Segment Embedding）来区分句子 A 和句子 B，这是 NSP 任务的必要设计。

### MLM 数据构造

```python
# 80% 替换为 [MASK]
mlm_input_ids[mask_replacement] = mask_token_id
# 10% 替换为随机 token
mlm_input_ids[random_replacement] = random_words[random_replacement]
# 10% 保持不变（已在 clone 中）
```

这一策略防止模型在微调阶段（没有 `[MASK]`）出现表示不匹配的问题，称为"Mask 偏差缓解"。

### NSP 数据构造

```python
if random.random() < 0.5:
    sent_b = sentences[2 * i + 1]  # 真实下一句
    label = 0
else:
    rand_idx = random.randint(0, len(sentences) - 1)
    sent_b = sentences[rand_idx]   # 随机句子
    label = 1
```

50% 正例 + 50% 负例的采样策略，让模型学习句子间的连贯性判断。

## 参考资料

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Devlin et al., 2019
- [The Illustrated BERT, ELMo, and co.](https://jalammar.github.io/illustrated-bert/) — Jay Alammar
- [Hugging Face BERT Documentation](https://huggingface.co/docs/transformers/model_doc/bert)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — Liu et al., 2019
