# Byte Pair Encoding (BPE)

## 算法简介

Byte Pair Encoding（BPE，字节对编码）是GPT系列模型（GPT-2、GPT-3、GPT-4、LLaMA等）采用的核心分词算法。它通过迭代合并语料中出现频率最高的相邻字节对，构建一个大小可控的子词词汇表，从而在保证无未登录词（OOV）的同时压缩序列长度。

## 核心思想

传统词级分词会遇到未登录词问题，字符级分词则导致序列过长。BPE的洞察在于：**高频出现的字符组合应该被当作一个整体token，而低频组合可以拆分为更小的子词**。这样，"playing"可能被编码为["play", "ing"]，而"xyz"可能被编码为["x", "y", "z"]。

BPE从字节级（256种可能）开始，这意味着它可以表示任何Unicode文本，不会出现无法编码的字符。

## 数学公式

设当前词汇表为 $V$，训练语料为 $D$。算法维护一个合并规则集合 $M$：

1. **频率统计**：
   $$
   \text{count}(x, y) = \sum_{w \in D} \sum_{i=1}^{|w|-1} \mathbb{1}[w_i = x, w_{i+1} = y]
   $$

2. **合并选择（贪心策略）**：
   $$
   (x^*, y^*) = \arg\max_{(x,y)} \text{count}(x, y)
   $$

3. **新token生成**：
   $$
   z = x^* \oplus y^*, \quad V \leftarrow V \cup \{z\}, \quad M \leftarrow M \cup \{(x^*, y^*) \to z\}
   $$

4. **编码时合并优先级**：
   对输入序列，按训练顺序（merge rank）依次应用合并规则：
   $$
   \text{rank}(x, y) = \text{order of } (x,y) \text{ in } M
   $$
   每次选择可合并且rank最小的pair进行合并。

## 时间/空间复杂度

- **训练时间复杂度**：$O(N \cdot M)$
  - $N$：语料总长度（字节数）
  - $M$：合并轮数（通常 $M = V_{target} - 256$）
  - 每轮需要遍历语料统计pair频率，然后全局替换。

- **编码时间复杂度**：$O(L \cdot M)$
  - $L$：输入文本长度
  - 最坏情况下需要扫描所有合并规则。

- **空间复杂度**：$O(V + M)$
  - $V$：词汇表大小
  - $M$：合并规则数

- **与替代方案对比**：
  - 比WordPiece训练更快（无需计算似然分数）
  - 比Unigram解码更简单（无需Viterbi动态规划）
  - 但编码结果不如Unigram灵活（贪心合并 vs 概率最优路径）

## 面试高频考点

1. **问题**：BPE和WordPiece的核心区别是什么？
   **答案**：BPE选择**频率最高**的pair进行合并，WordPiece选择使**训练数据似然增益最大**的pair。BPE的合并标准是 $ \arg\max \text{count}(x,y) $，WordPiece的标准是 $ \arg\max \frac{\text{count}(xy)}{\text{count}(x)\text{count}(y)} $。

2. **问题**：为什么GPT系列使用字节级BPE？
   **答案**：字节级BPE以256个基础字节为起点，可以编码**任何**Unicode字符，不会出现OOV。即使遇到训练时未见过的字符或生僻字，也能拆分为字节序列表示。

3. **问题**：BPE编码时为什么要按merge rank（训练顺序）合并，而不是按频率？
   **答案**：训练时每一轮的频率统计都依赖于之前所有的合并结果。merge rank反映了合并的"优先级"或"粒度"，按rank合并才能保证与训练时一致的拆分逻辑。如果按频率合并，可能会先合并一个细粒度的pair，导致与训练时不一致的分词结果。

4. **问题**：BPE有什么已知缺陷？
   **答案**：
   - **数字和空格敏感**："9.11"可能被拆分为["9", ".", "11"]，导致GPT在数字比较上出错。
   - **语言无关性不足**：不同语言的字符频率差异大，共享词汇表效率不高。
   - **贪心策略非最优**：局部最优的合并序列不一定是全局最优的分词结果。

5. **问题**：如果输入文本包含训练时未出现的字符，BPE如何处理？
   **答案**：字节级BPE可以处理任何字符，因为任何Unicode字符最终都可以编码为1-4个UTF-8字节，而这些字节一定在初始的0-255词汇表中。只是未见过的新字符会被拆成更多token，而不是产生[UNK]。

## 代码解析

### 1. 初始化与词汇表构建

```python
self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
```

初始词汇表包含256个字节，这是字节级BPE的基础。每个token_id对应一个具体的字节值，确保任何文本都可以被表示。

### 2. 频率统计

```python
def _get_stats(self, token_ids: List[int]) -> Dict[Tuple[int, int], int]:
    counts = defaultdict(int)
    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        counts[pair] += 1
    return counts
```

线性扫描序列，统计所有相邻pair的出现次数。时间复杂度为 $O(L)$，$L$ 为序列长度。

### 3. 合并操作

```python
def _merge(self, token_ids, pair, new_id):
    new_ids = []
    i = 0
    while i < len(token_ids):
        if i < len(token_ids) - 1 and (token_ids[i], token_ids[i+1]) == pair:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(token_ids[i])
            i += 1
    return new_ids
```

从左到右扫描，将匹配到的pair替换为新token。注意扫描是**贪心左到右**的，不会重叠匹配。

### 4. 编码策略

```python
best_pair = None
min_rank = float("inf")
for pair in stats:
    if pair in self.merges:
        rank = list(self.merges.keys()).index(pair)
        if rank < min_rank:
            min_rank = rank
            best_pair = pair
```

编码时，在所有可合并的pair中选择**rank最小**（即训练时最早合并）的进行合并。这保证了编码结果与训练时的逻辑一致。

## 参考资料

- Sennrich, R., Haddow, B., & Birch, A. (2015). [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909). ACL.
- Radford, A., et al. (2019). [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2). OpenAI.
- Hugging Face Tokenizers Library: [https://github.com/huggingface/tokenizers](https://github.com/huggingface/tokenizers)
- Andrej Karpathy, "Let's build the GPT Tokenizer": [nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero)
