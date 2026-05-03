# WordPiece

## 算法简介

WordPiece是Google提出的子词分词算法，被BERT、DistilBERT、ALBERT等模型采用。与BPE类似，它通过迭代合并构建子词词汇表，但合并标准从"频率最高"改为"似然增益最大"，使得合并后的token能更好地提升训练数据的语言模型似然。

## 核心思想

BPE只看pair出现的绝对频率，这可能导致一些常见但意义不紧密的字符对被过早合并。WordPiece的洞察是：**应该合并那些"共现频率远超独立出现频率期望"的pair**。如果两个token经常一起出现，且这种共现不是偶然的，那么它们应该被合并。

WordPiece使用以下分数衡量共现强度：

$$
\text{score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \times \text{freq}(b)}
$$

这个分数类似于点互信息（PMI），值越大说明两个token的关联性越强。

## 数学公式

1. **pair频率统计**：
   $$
   \text{freq}(a, b) = \sum_{w \in D} \sum_{i=1}^{|w|-1} \mathbb{1}[w_i = a, w_{i+1} = b] \cdot \text{count}(w)
   $$

2. **token频率统计**：
   $$
   \text{freq}(t) = \sum_{w \in D} \sum_{i=1}^{|w|} \mathbb{1}[w_i = t] \cdot \text{count}(w)
   $$

3. **WordPiece分数（合并标准）**：
   $$
   \text{score}(a, b) = \frac{\text{freq}(a, b)}{\text{freq}(a) \cdot \text{freq}(b)}
   $$

4. **贪心编码（最长匹配）**：
   对输入单词 $w$，从位置 $i=0$ 开始：
   $$
   t^* = \arg\max_{t \in V, w_{i..i+|t|-1} = t} |t|
   $$
   选择能匹配的最长子词，然后 $i \leftarrow i + |t^*|$，重复直到 $i = |w|$。

## 时间/空间复杂度

- **训练时间复杂度**：$O(N \cdot M)$
  - $N$：语料总长度
  - $M$：合并轮数
  - 每轮需要统计token频率和pair频率，然后计算分数。

- **编码时间复杂度**：$O(L \cdot |V|)$
  - $L$：输入文本长度
  - $|V|$：词汇表大小
  - 对每个位置，需要在词汇表中寻找最长匹配前缀。
  - 实际工程中可用Trie树优化到 $O(L \cdot K)$，$K$ 为最大子词长度。

- **空间复杂度**：$O(V + M)$
  - $V$：词汇表大小
  - $M$：合并规则数

- **与替代方案对比**：
  - 比BPE训练稍慢（需额外计算token频率和分数）
  - 编码策略不同：BPE按merge rank合并，WordPiece用贪心最长匹配
  - 子词标记方式不同：WordPiece用"##"标记非首子词，BPE无此标记

## 面试高频考点

1. **问题**：WordPiece和BPE的核心区别是什么？
   **答案**：
   - **合并标准不同**：BPE选频率最高的pair，WordPiece选score最高的pair，score = freq(ab) / (freq(a) * freq(b))。
   - **编码方式不同**：BPE按训练顺序（merge rank）依次合并；WordPiece用贪心最长匹配（从词首找最长子词）。
   - **子词标记不同**：WordPiece在非首子词前加"##"（如["play", "##ing"]），BPE没有这种标记。

2. **问题**：WordPiece的score公式为什么这样设计？
   **答案**：该公式衡量两个token共现的频率相对于它们独立出现频率的乘积。如果 $a$ 和 $b$ 独立出现，则期望共现频率为 $freq(a) \cdot freq(b)$（归一化后）。实际共现频率远高于期望值，说明它们有强搭配关系，应该合并。这本质上是点互信息（PMI）的一种变体。

3. **问题**：BERT为什么使用WordPiece而不是BPE？
   **答案**：Google在开发BERT时选择了WordPiece，主要因为：
   - WordPiece的似然导向合并能生成更符合语言学规律的子词边界。
   - "##"标记让模型能区分词首和词中子词，有助于学习词边界信息。
   - 贪心最长匹配编码简单高效，适合大规模预训练。
   不过，后续研究表明两种算法在实际效果上差异不大，选择更多取决于工程惯性。

4. **问题**：WordPiece如何处理未登录词（OOV）？
   **答案**：WordPiece通过贪心最长匹配将OOV词拆分为已知的子词。如果某个字符没有任何子词能匹配，则输出[UNK]。由于初始词汇表包含所有单字符，理论上任何文本都可以被拆分为字符序列，不会出现完全无法编码的情况（除非字符不在初始表中）。

5. **问题**：WordPiece的"##"标记有什么作用？
   **答案**："##"标记表示该子词不是单词的开头部分。例如"playing"被编码为["play", "##ing"]。这让模型能区分：
   - "play"作为独立单词的语义
   - "##ing"作为后缀的语法功能
   如果没有"##"，"play"在两种情况下看起来一样，模型需要依赖上下文推断，增加了学习难度。

## 代码解析

### 1. 初始词汇表构建

```python
def _build_initial_vocab(self, word_freqs):
    chars = set()
    for word in word_freqs:
        for char in word.split():
            chars.add(char)
    sorted_chars = sorted(chars)
    self.vocab = {ch: i for i, ch in enumerate(sorted_chars)}
```

从语料中提取所有唯一字符，按字典序构建初始词汇表。与字节级BPE不同，WordPiece从字符级开始，而非字节级。

### 2. WordPiece分数计算

```python
def _compute_pair_scores(self, splits, word_freqs):
    token_freqs = defaultdict(int)
    for word, tokens in splits.items():
        for token in tokens:
            token_freqs[token] += word_freqs[word]

    pair_freqs = defaultdict(int)
    for word, tokens in splits.items():
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_freqs[pair] += word_freqs[word]

    scores = {}
    for pair, freq in pair_freqs.items():
        a, b = pair
        scores[pair] = freq / (token_freqs[a] * token_freqs[b])
    return scores
```

先统计每个token的总频率和每个pair的共现频率，然后计算score。这是WordPiece与BPE最核心的区别。

### 3. 贪心最长匹配编码

```python
def encode_word(self, word):
    tokens = []
    remaining = word
    is_first = True
    while remaining:
        longest_match = None
        longest_len = 0
        for token_str in self.vocab:
            if not is_first and token_str.startswith("##"):
                raw = token_str[2:]
                if remaining.startswith(raw) and len(raw) > longest_len:
                    longest_match = token_str
                    longest_len = len(raw)
            elif is_first and not token_str.startswith("##"):
                if remaining.startswith(token_str) and len(token_str) > longest_len:
                    longest_match = token_str
                    longest_len = len(token_str)
        if longest_match is None:
            # fallback to [UNK] or char
            break
        tokens.append(self.vocab[longest_match])
        remaining = remaining[longest_len:]
        is_first = False
    return tokens
```

从单词开头开始，在词汇表中寻找最长匹配。首token不加"##"，后续token匹配"##"前缀的版本。这是BERT WordPieceTokenizer的标准行为。

## 参考资料

- Schuster, M., & Nakajima, K. (2012). [Japanese and Korean voice search](https://ieeexplore.ieee.org/document/6289079). ICASSP.
- Devlin, J., et al. (2018). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). NAACL.
- Hugging Face WordPiece Documentation: [https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#models.WordPiece](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#models.WordPiece)
- WordPiece vs BPE comparison: [Hugging Face Course - Chapter 6](https://huggingface.co/learn/nlp-course/chapter6/6)
