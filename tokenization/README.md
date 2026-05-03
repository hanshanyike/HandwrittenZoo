# Tokenization（分词算法）

本目录包含现代大语言模型（LLM）核心分词算法的从零实现，涵盖GPT系列、BERT系列和T5/XLNet系列使用的代表性方法。

## 文件结构

```
tokenization/
├── bpe.py              # Byte Pair Encoding 实现（GPT系列核心）
├── bpe.md              # BPE 算法详解与面试考点
├── wordpiece.py        # WordPiece 实现（BERT系列核心）
├── wordpiece.md        # WordPiece 算法详解与面试考点
├── sentencepiece.py    # SentencePiece Unigram 演示实现（T5/XLNet核心）
├── sentencepiece.md    # SentencePiece 算法详解与面试考点
└── README.md           # 本文件：分类概览与面试焦点
```

## 算法概览

| 算法 | 代表模型 | 合并/分词标准 | 预分词依赖 | 核心特点 |
|------|----------|---------------|------------|----------|
| **BPE** | GPT-2/3/4, LLaMA | 频率最高pair | 可选（字节级无依赖） | 简单高效，确定性贪心合并 |
| **WordPiece** | BERT, DistilBERT | 似然增益score = freq(ab)/(freq(a)*freq(b)) | 需要（空格分词） | 语言学边界更合理，"##"标记子词位置 |
| **SentencePiece (Unigram)** | T5, ALBERT, XLNet | 概率最优Viterbi路径 | 无（语言无关） | 概率框架，支持采样正则化，多语言友好 |

## 面试焦点

### 1. 三者核心区别（必问）

**BPE vs WordPiece vs SentencePiece** 是NLP/LLM岗位面试的超高频问题。回答框架：

- **合并标准**：BPE看频率，WordPiece看似然增益（PMI-like score），Unigram看概率最优路径
- **编码策略**：BPE按merge rank贪心合并，WordPiece按最长匹配贪心，Unigram用Viterbi全局最优
- **语言依赖**：BPE/WordPiece通常需要预分词（空格），SentencePiece语言无关（"▁"编码空格）
- **灵活性**：只有Unigram支持概率采样（Subword Regularization）

### 2. 字节级BPE为什么能处理任何文本？

因为任何Unicode字符都可以编码为1-4个UTF-8字节，而字节级BPE的初始词汇表包含所有256种字节。这意味着：**不存在无法表示的字符**，只是新字符会被拆成更多token。

### 3. 分词对模型能力的影响

- **数字比较错误**（如9.11 > 9.9）：源于"9.11"被拆为["9", ".", "11"]，模型逐token比较
- **拼写能力弱**：token是子词而非字符，模型不直接看到字母序列
- **代码缩进敏感**：空格和换行被压缩为特殊token，多空格可能映射到同一个token
- **多语言效率差异**：中文在GPT-3中通常1-2个汉字/token，而英文约0.75个单词/token

### 4. 工业级优化方向

- **Trie树加速编码**：WordPiece/SentencePiece的最长匹配可用前缀树优化到 $O(L)$
- **并行训练**：BPE的pair统计可MapReduce化（参考tiny-BPE的多进程实现）
- **缓存**：对高频词直接查表，避免重复运行分词算法
- **正则化**：Unigram的n-best采样可用于训练时的数据增强

## 学习建议

1. **先理解BPE**：BPE是最简单的子词分词算法，掌握它后再学WordPiece和Unigram会事半功倍。
2. **手推例子**：用"low lower lowest"这个小语料，手推BPE和WordPiece的前3轮合并，体会合并标准的差异。
3. **对比编码**：同一句话用三种算法编码，观察分词边界的差异（如"playing" -> ["play", "ing"] vs ["play", "##ing"]）。
4. **关注面试陷阱**：
   - BPE编码为什么按merge rank而不是频率？（保证与训练一致）
   - WordPiece的score公式为什么分母是乘积？（PMI，衡量共现强度）
   - SentencePiece的"▁"和空格有什么区别？（"▁"是显式编码，空格是分隔符）

## 参考资料

- Sennrich et al. (2015). [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909). ACL. (BPE)
- Schuster & Nakajima (2012). [Japanese and Korean voice search](https://ieeexplore.ieee.org/document/6289079). ICASSP. (WordPiece)
- Kudo (2018). [Subword Regularization](https://arxiv.org/abs/1804.10959). EMNLP. (Unigram)
- Kudo & Richardson (2018). [SentencePiece](https://arxiv.org/abs/1808.06226). EMNLP. (SentencePiece框架)
- Hugging Face Tokenizers Course: [https://huggingface.co/learn/nlp-course/chapter6](https://huggingface.co/learn/nlp-course/chapter6)
