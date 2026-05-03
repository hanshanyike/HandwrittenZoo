"""
WordPiece Tokenizer — BERT核心分词算法

核心思想：
    与BPE类似，WordPiece从字符级初始词汇表出发迭代合并相邻token对。
    关键区别在于合并标准：WordPiece不选择频率最高的pair，而是选择
    使训练数据似然增益最大的pair。具体地，它使用pair频率除以两个
    组成token各自频率的乘积作为分数，衡量两个token的"共现强度"。

时间复杂度：
    - 训练：O(N * M)，与BPE同级，但每轮需额外计算token频率
    - 编码：O(L * |V|) 或 O(L * K)，L为输入长度，K为最大子词长度

空间复杂度：
    - O(V + M)，V为词汇表大小，M为合并规则数

面试频率：高（BERT/NLP相关岗位高频考点）
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class WordPieceTokenizer:
    """
    基于WordPiece算法的子词分词器实现。

    参考：Schuster & Nakajima, "Japanese and Korean voice search", 2012
          Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers", 2018
    """

    def __init__(self, vocab_size: int = 300):
        """
        初始化WordPiece分词器。

        Args:
            vocab_size: 目标词汇表大小（包含初始字符）。
        """
        self.vocab_size = vocab_size
        self.num_merges = vocab_size  # 将在训练时动态确定实际合并数

        # vocab: token_str -> token_id
        self.vocab: Dict[str, int] = {}
        # id_to_token: token_id -> token_str
        self.id_to_token: Dict[int, str] = {}
        # merges: 合并规则列表，记录训练顺序，用于编码时参考优先级
        self.merges: List[Tuple[str, str]] = []

    def _build_initial_vocab(self, word_freqs: Dict[str, int]):
        """
        从语料中构建初始字符级词汇表。

        Args:
            word_freqs: 单词到频率的映射（单词内部以空格分隔字符）。
        """
        chars = set()
        for word in word_freqs:
            for char in word.split():
                chars.add(char)
        # 按字典序分配id，保证可复现性
        sorted_chars = sorted(chars)
        self.vocab = {ch: i for i, ch in enumerate(sorted_chars)}
        self.id_to_token = {i: ch for ch, i in self.vocab.items()}

    def _compute_pair_scores(
        self,
        splits: Dict[str, List[str]],
        word_freqs: Dict[str, int]
    ) -> Dict[Tuple[str, str], float]:
        """
        计算所有相邻token对的WordPiece分数。

        WordPiece分数公式：
            score(a, b) = freq(ab) / (freq(a) * freq(b))

        该分数衡量了两个token共现的频率相对于它们独立出现频率的期望值。
        分数越高，说明这对token越应该被合并为一个整体。

        Args:
            splits: 每个单词当前的分词结果（字符列表）。
            word_freqs: 每个单词的出现频率。

        Returns:
            每个pair及其WordPiece分数的字典。
        """
        # 统计每个token的总频率
        token_freqs: Dict[str, int] = defaultdict(int)
        for word, tokens in splits.items():
            freq = word_freqs[word]
            for token in tokens:
                token_freqs[token] += freq

        # 统计每个pair的频率
        pair_freqs: Dict[Tuple[str, str], int] = defaultdict(int)
        for word, tokens in splits.items():
            freq = word_freqs[word]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq

        # 计算WordPiece分数
        scores = {}
        for pair, freq in pair_freqs.items():
            a, b = pair
            # 避免除零（理论上不会发生，因为pair存在则a、b必存在）
            denom = token_freqs[a] * token_freqs[b]
            if denom > 0:
                scores[pair] = freq / denom
            else:
                scores[pair] = 0.0
        return scores

    def train(self, texts: List[str]):
        """
        在输入文本列表上训练WordPiece模型。

        注意：本实现采用基于空格的预分词（与BERT的原始实现一致），
        实际BERT使用更复杂的预分词器（如BasicTokenizer）。

        Args:
            texts: 训练语料列表，每个元素为一个句子/文档字符串。
        """
        # 预分词：按空格切分单词，并统计词频
        word_freqs: Dict[str, int] = defaultdict(int)
        for text in texts:
            for word in text.strip().split():
                # 每个单词内部以空格分隔字符，末尾添加</w>标记词尾
                # 这是WordPiece/BPE的常见做法，用于区分词内和词尾子词
                spaced_word = " ".join(list(word)) + " </w>"
                word_freqs[spaced_word] += 1

        # 构建初始字符词汇表
        self._build_initial_vocab(word_freqs)

        # 初始化每个单词的拆分状态
        splits: Dict[str, List[str]] = {}
        for word in word_freqs:
            splits[word] = word.split()

        # 迭代合并
        target_merges = self.vocab_size - len(self.vocab)
        for _ in range(target_merges):
            scores = self._compute_pair_scores(splits, word_freqs)
            if not scores:
                break

            # 选择分数最高的pair进行合并
            best_pair = max(scores, key=scores.get)
            best_score = scores[best_pair]
            if best_score <= 0:
                break

            a, b = best_pair
            new_token = a + b

            # 将新token加入词汇表
            new_id = len(self.vocab)
            self.vocab[new_token] = new_id
            self.id_to_token[new_id] = new_token
            self.merges.append(best_pair)

            # 更新所有包含该pair的单词的拆分状态
            for word in list(splits.keys()):
                tokens = splits[word]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                splits[word] = new_tokens

    def encode_word(self, word: str) -> List[int]:
        """
        对单个单词进行WordPiece编码（贪心最长匹配）。

        BERT的WordPiece编码器采用贪心策略：从单词开头开始，
        在词汇表中寻找能匹配的最长子词，然后对剩余部分递归处理。
        如果没有任何子词匹配，则将该字符标记为[UNK]。

        Args:
            word: 输入单词（不含空格）。

        Returns:
            token_id列表。
        """
        # 与BERT一致：非首个子词前加"##"
        # 这里我们简化为：首token不加，后续token加"##"
        tokens = []
        remaining = word
        is_first = True

        while remaining:
            longest_match = None
            longest_len = 0

            # 寻找词汇表中最长的匹配前缀
            for token_str in self.vocab:
                if token_str == "</w>":
                    continue
                # 非首token需要匹配"##"前缀的版本
                if not is_first and token_str.startswith("##"):
                    raw = token_str[2:]
                    if remaining.startswith(raw) and len(raw) > longest_len:
                        longest_match = token_str
                        longest_len = len(raw)
                elif is_first and not token_str.startswith("##") and remaining.startswith(token_str):
                    if len(token_str) > longest_len:
                        longest_match = token_str
                        longest_len = len(token_str)

            if longest_match is None:
                # 未匹配到任何子词，标记为[UNK]
                unk_id = self.vocab.get("[UNK]", -1)
                if unk_id == -1:
                    # 如果没有[UNK]token，则按字符fallback
                    tokens.append(self.vocab.get(remaining[0], 0))
                    remaining = remaining[1:]
                else:
                    tokens.append(unk_id)
                    break
            else:
                tokens.append(self.vocab[longest_match])
                remaining = remaining[longest_len:]
                is_first = False

        return tokens

    def encode(self, text: str) -> List[int]:
        """
        将输入文本编码为token_id序列。

        Args:
            text: 输入文本字符串。

        Returns:
            token_id列表。
        """
        token_ids = []
        for word in text.strip().split():
            token_ids.extend(self.encode_word(word))
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        将token_id序列解码回原始文本。

        Args:
            token_ids: token_id列表。

        Returns:
            解码后的字符串。
        """
        tokens = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, "[UNK]")
            if token.startswith("##"):
                tokens.append(token[2:])
            elif token == "</w>":
                tokens.append(" ")
            else:
                tokens.append(token)
        # 简单拼接，实际BERT解码器会更复杂
        return "".join(tokens)

    def vocab_size_current(self) -> int:
        """返回当前实际词汇表大小。"""
        return len(self.vocab)


if __name__ == "__main__":
    # 自测：用简单语料训练并验证编解码
    corpus = [
        "low lower lowest new newer newest",
        "walk walking walked talk talking",
        "play playing played player players",
        "university universal universe",
    ]

    tokenizer = WordPieceTokenizer(vocab_size=80)
    tokenizer.train(corpus)

    print("训练后词汇表大小:", tokenizer.vocab_size_current())
    print("合并规则数:", len(tokenizer.merges))

    test_text = "lower walking player"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"原文: {test_text}")
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")

    # 展示部分学到的词汇
    print("\n部分词汇表（按id排序前15个）:")
    for i in range(min(15, tokenizer.vocab_size_current())):
        print(f"  {i}: {tokenizer.id_to_token[i]}")
