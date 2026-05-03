# SentencePiece (Unigram)

## 算法简介

SentencePiece是由Google提出的语言无关分词框架，被T5、ALBERT、XLNet等模型采用。它不是一个单一算法，而是支持BPE和Unigram两种算法的统一框架。其核心创新在于**将文本视为原始Unicode字节流**，不依赖任何语言特定的预分词规则（如英文空格、中文分词工具），通过将空格编码为特殊符号"▁"（U+2581），实现了对中文、日文、韩文等无空格分隔语言的统一处理。

本实现演示SentencePiece框架下的**Unigram语言模型算法**。

## 核心思想

BPE和WordPiece在训练时都是"从小到大"逐步合并，而Unigram采取**"从大到小"**的策略：

1. 从一个非常大的初始词汇表（包含所有可能的子串）开始
2. 为每个子词分配一个概率，使得整个语料的似然最大
3. 使用Viterbi算法对输入文本进行概率最优分词
4. 通过EM迭代不断优化子词概率，并裁剪掉低概率子词

Unigram的洞察是：**分词可以看作一个概率图模型问题**。给定词汇表中每个子词的概率，一个句子的最优分词就是概率乘积最大的路径。这与HMM中的Viterbi解码完全类似。

## 数学公式

1. **子词概率模型**：
   设词汇表为 $V$，每个子词 $x \in V$ 有概率 $P(x)$，满足 $\sum_{x \in V} P(x) = 1$。

2. **句子分词概率**：
   对句子 $S$ 的一种分词 $T = (t_1, t_2, ..., t_k)$：
   $$
   P(T) = \prod_{i=1}^{k} P(t_i)
   $$

3. **Viterbi动态规划（编码）**：
   设 $dp[i]$ 为前 $i$ 个字符的最大对数概率：
   $$
   dp[i] = \max_{j < i, S[j:i] \in V} \left( dp[j] + \log P(S[j:i]) \right)
   $$

4. **EM训练（简化Viterbi-EM）**：
   - **E步**：用当前概率对语料做Viterbi分词，统计每个子词的出现次数 $c(x)$
   - **M步**：重新估计概率
   $$
   P(x) = \frac{c(x)}{\sum_{x' \in V} c(x')}
   $$

5. **词汇表裁剪**：
   保留使语料似然下降最小的子词集合，通常直接保留概率最高的 $K$ 个子词（同时保留所有单字符）。

## 时间/空间复杂度

- **训练时间复杂度**：$O(I \cdot N \cdot L \cdot K)$
  - $I$：EM迭代轮数（通常5-10轮）
  - $N$：语料句子数
  - $L$：平均句子长度
  - $K$：最大子词长度
  - 每轮需要对每句话运行Viterbi算法，复杂度为 $O(L \cdot K)$。

- **编码时间复杂度**：$O(L \cdot K)$
  - $L$：输入文本长度
  - $K$：最大子词长度
  - Viterbi动态规划需要枚举所有以位置 $i$ 结尾的合法子词。

- **空间复杂度**：$O(V \cdot K)$
  - $V$：词汇表大小
  - $K$：最大子词长度
  - 需要存储词汇表和动态规划表。

- **与替代方案对比**：
  - 训练比BPE/WordPiece慢（需要多轮EM迭代）
  - 编码与WordPiece同级（都是动态规划/贪心匹配）
  - 优势在于语言无关性和概率框架的灵活性（可采样多种分词结果做数据增强）

## 面试高频考点

1. **问题**：SentencePiece和BPE/WordPiece最本质的区别是什么？
   **答案**：
   - **语言无关性**：SentencePiece不依赖空格预分词，将空格视为普通字符"▁"，因此对中文、日文等无空格语言同样有效。
   - **算法方向**：BPE/WordPiece是"从小到大"合并；Unigram是"从大到小"裁剪。
   - **概率框架**：Uniggram基于语言模型概率，可以采样多种分词路径（Subword Regularization），BPE/WordPiece是确定性的。

2. **问题**：SentencePiece如何处理中文（无空格分隔）？
   **答案**：SentencePiece将文本视为原始字节/字符序列，不依赖空格作为词边界。中文文本中的每个汉字都是序列中的一个字符，算法通过统计自动学习哪些汉字组合应该作为整体token。空格被显式编码为"▁"符号，因此中文文本中没有"▁"（除非原文有空格），模型通过字符共现频率自然学习词边界。

3. **问题**：Unigram的Viterbi算法和BPE的贪心合并有什么区别？
   **答案**：
   - BPE编码时从左到右贪心合并，只考虑局部最优（最早训练的pair优先）。
   - Unigram的Viterbi算法考虑全局最优，通过动态规划找到概率乘积最大的分词路径。
   - 举例：词汇表有"ab", "bc", "abc"，输入"abc"。BPE可能先合并"ab"得到["ab", "c"]；Viterbi会比较 $P(ab) \cdot P(c)$ 和 $P(abc)$，选择概率更大的。

4. **问题**：为什么Unigram可以实现"Subword Regularization"（子词正则化）？
   **答案**：因为Unigram为每个子词定义了概率，一个句子有多种合法分词方式，每种方式都有确定的概率。训练时可以从这些分词路径中按概率采样，而不是总是用最优路径。这相当于对输入做数据增强，让模型看到同一句话的不同tokenization，增强对分词边界的鲁棒性。BPE和WordPiece是确定性的，无法实现这一点。

5. **问题**：SentencePiece中的"▁"符号有什么作用？
   **答案**："▁"（U+2581）是SentencePiece用来显式编码空格的特殊符号。原始文本中的每个空格被替换为"▁"后，整个文本变成连续的字符序列，不再需要预分词。解码时，将"▁"替换回空格即可恢复原始文本。这使得SentencePiece可以语言无关地处理文本：英文有空格、中文无空格，对算法来说都只是"▁"出现与否的问题。

## 代码解析

### 1. 语言无关预处理

```python
processed = text.replace(" ", "▁")
```

将空格替换为"▁"，这是SentencePiece的核心技巧。处理后文本成为连续字符流，不再需要按空格预分词。

### 2. Viterbi动态规划分词

```python
dp = [-float("inf")] * (n + 1)
dp[0] = 0.0
for i in range(1, n + 1):
    for j in range(max(0, i - 8), i):
        subword = text[j:i]
        if subword in self.vocab:
            _, log_prob = self.vocab[subword]
            score = dp[j] + log_prob
            if score > dp[i]:
                dp[i] = score
                prev[i] = j
```

标准的Viterbi动态规划。$dp[i]$ 表示前 $i$ 个字符的最大对数概率。通过枚举所有以 $i$ 结尾的合法子词，更新最优路径。时间复杂度 $O(L \cdot K)$，$K$ 为最大子词长度限制。

### 3. EM训练迭代

```python
for iteration in range(self.num_iterations):
    # E步：用当前概率做Viterbi分词，统计子词出现次数
    expected_counts, total_log_prob = self._expectation_step(processed)
    # M步：根据计数重新估计概率
    new_probs = self._maximization_step(expected_counts)
    # 裁剪低概率子词
    probs = self._prune_vocab(new_probs)
```

E步用Viterbi得到当前最优分词，统计每个子词的出现次数。M步用最大似然估计更新概率。然后裁剪掉概率最低的子词，缩小词汇表。重复直到收敛。

### 4. 词汇表裁剪策略

```python
single_chars = {k: v for k, v in probs.items() if len(k) == 1}
others = {k: v for k, v in probs.items() if len(k) > 1}
keep_num = max(0, self.vocab_size - len(single_chars))
sorted_others = sorted(others.items(), key=lambda x: x[1], reverse=True)
kept_others = dict(sorted_others[:keep_num])
```

单字符必须保留，否则可能出现无法编码的字符。多字符子词按概率排序，保留top-K。这是简化版裁剪策略，工业级实现会计算删除每个子词对语料似然的边际影响。

## 参考资料

- Kudo, T. (2018). [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/abs/1804.10959). EMNLP.
- Kudo, T., & Richardson, J. (2018). [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226). EMNLP.
- Google SentencePiece GitHub: [https://github.com/google/sentencepiece](https://github.com/google/sentencepiece)
- Hugging Face Unigram Documentation: [https://huggingface.co/learn/nlp-course/chapter6/7](https://huggingface.co/learn/nlp-course/chapter6/7)
