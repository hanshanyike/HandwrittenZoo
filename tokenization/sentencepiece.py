"""
SentencePiece Tokenizer — 语言无关分词框架（Unigram算法演示）

核心思想：
    SentencePiece不是一个单一算法，而是一个分词框架，支持BPE和Unigram两种算法。
    其最大特点是"语言无关"：将文本视为原始Unicode字节流，不依赖语言特定的
    预分词器（如英文空格、中文分词工具）。通过将空格编码为特殊符号"▁"（U+2581），
    实现了对中文、日文等无空格语言的统一处理。

    本实现演示基于Unigram语言模型的分词：
    1. 从大规模初始词汇表出发，为每个子词分配概率
    2. 使用Viterbi算法找到概率最优的分词路径
    3. 通过EM迭代优化子词概率，并裁剪低概率子词

时间复杂度：
    - 训练（EM迭代）：O(I * N * L)，I为迭代轮数，N为语料句子数，L为平均句子长度
    - 编码（Viterbi）：O(L * K)，L为输入长度，K为最大子词长度

空间复杂度：
    - O(V * L)，V为词汇表大小，L为平均子词长度（存储Trie或前缀字典）

面试频率：高（多语言/大模型岗位高频考点）
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import math


class SentencePieceTokenizer:
    """
    基于Unigram语言模型的SentencePiece分词器简化实现。

    参考：Kudo, T. (2018). [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/abs/1804.10959)
          Kudo & Richardson (2018). [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)
    """

    def __init__(self, vocab_size: int = 100, num_iterations: int = 5):
        """
        初始化SentencePiece分词器（Unigram模式）。

        Args:
            vocab_size: 目标词汇表大小。
            num_iterations: EM训练迭代轮数。
        """
        self.vocab_size = vocab_size
        self.num_iterations = num_iterations

        # vocab: 子词字符串 -> (token_id, log_prob)
        self.vocab: Dict[str, Tuple[int, float]] = {}
        # id_to_token: token_id -> 子词字符串
        self.id_to_token: Dict[int, str] = {}

    def _build_seed_vocab(self, text: str, max_subword_len: int = 4) -> Dict[str, int]:
        """
        构建初始种子词汇表：从语料中提取所有长度的子串。

        Args:
            text: 训练语料（将空格替换为"▁"后的连续字符串）。
            max_subword_len: 最大子词长度。

        Returns:
            子词到频率的字典。
        """
        freq: Dict[str, int] = defaultdict(int)
        n = len(text)
        # 提取所有长度的子串作为候选子词
        for length in range(1, max_subword_len + 1):
            for i in range(n - length + 1):
                subword = text[i:i + length]
                freq[subword] += 1
        # 同时加入单字符，确保任何文本都可被编码
        for ch in set(text):
            if ch not in freq:
                freq[ch] = 1
        return dict(freq)

    def _viterbi_segment(self, text: str) -> Tuple[List[str], float]:
        """
        使用Viterbi算法找到最优分词路径（概率最大）。

        动态规划定义：
            dp[i] = 文本前i个字符的最大对数概率
            prev[i] = 达到dp[i]时，最后一个子词的起始位置

        Args:
            text: 待分词的文本字符串。

        Returns:
            (最优子词列表, 该路径的对数概率)
        """
        n = len(text)
        # dp[i] 表示前i个字符的最大对数概率
        dp = [-float("inf")] * (n + 1)
        dp[0] = 0.0
        # prev[i] 记录前驱节点位置
        prev = [-1] * (n + 1)
        # token_choice[i] 记录到达i时使用的子词
        token_choice = [""] * (n + 1)

        for i in range(1, n + 1):
            # 枚举所有以位置i结尾的子词
            for j in range(max(0, i - 8), i):  # 限制最大子词长度为8，降低复杂度
                subword = text[j:i]
                if subword in self.vocab:
                    _, log_prob = self.vocab[subword]
                    score = dp[j] + log_prob
                    if score > dp[i]:
                        dp[i] = score
                        prev[i] = j
                        token_choice[i] = subword

        # 回溯得到最优路径
        if dp[n] == -float("inf"):
            # 无法完整覆盖，按字符fallback（只保留在vocab中的字符）
            fallback_tokens = []
            for ch in text:
                if ch in self.vocab:
                    fallback_tokens.append(ch)
                else:
                    # 用第一个单字符token兜底（理论上不会发生）
                    for single in self.vocab:
                        if len(single) == 1:
                            fallback_tokens.append(single)
                            break
            fallback_score = sum(self.vocab.get(t, (0, -10.0))[1] for t in fallback_tokens)
            return fallback_tokens, fallback_score

        tokens = []
        i = n
        while i > 0:
            tokens.append(token_choice[i])
            i = prev[i]
        tokens.reverse()
        return tokens, dp[n]

    def _expectation_step(self, text: str) -> Tuple[Dict[str, float], float]:
        """
        E步：对语料中所有可能的分词路径，计算每个子词的期望出现次数。

        为简化实现，本演示版本采用Viterbi近似（只考虑最优路径），
        而非完整的Forward-Backward算法。工业级实现会使用后者。

        Args:
            text: 训练语料。

        Returns:
            (子词期望计数, 语料总对数概率)
        """
        expected_counts: Dict[str, float] = defaultdict(float)
        total_log_prob = 0.0

        # 按句子切分语料（简单按"▁"后的空格近似）
        # 实际SentencePiece使用更复杂的预分词，这里简化处理
        sentences = text.split("▁")
        for sent in sentences:
            if not sent:
                continue
            # 还原开头的"▁"
            sent = "▁" + sent
            tokens, log_prob = self._viterbi_segment(sent)
            total_log_prob += log_prob
            for token in tokens:
                expected_counts[token] += 1.0

        return dict(expected_counts), total_log_prob

    def _maximization_step(self, expected_counts: Dict[str, float]) -> Dict[str, float]:
        """
        M步：根据期望计数重新估计子词概率。

        使用最大似然估计：
            P(token) = count(token) / sum(count(all tokens))

        Args:
            expected_counts: E步得到的子词期望计数。

        Returns:
            新的子词概率字典。
        """
        total = sum(expected_counts.values())
        if total == 0:
            return {token: 1.0 / len(expected_counts) for token in expected_counts}

        new_probs = {}
        for token, count in expected_counts.items():
            new_probs[token] = count / total
        return new_probs

    def _prune_vocab(self, probs: Dict[str, float], all_chars: set) -> Dict[str, float]:
        """
        裁剪词汇表：保留概率最高的vocab_size个子词，同时保证所有单字符都在词汇表中。

        Args:
            probs: 当前所有子词的概率。
            all_chars: 语料中出现的所有单字符集合。

        Returns:
            裁剪后的概率字典。
        """
        # 单字符必须保留，确保任何文本可编码
        single_chars = {k: v for k, v in probs.items() if len(k) == 1}
        others = {k: v for k, v in probs.items() if len(k) > 1}

        # 按概率排序，保留top-(vocab_size - len(single_chars))
        keep_num = max(0, self.vocab_size - len(single_chars))
        sorted_others = sorted(others.items(), key=lambda x: x[1], reverse=True)
        kept_others = dict(sorted_others[:keep_num])

        pruned = {}
        pruned.update(single_chars)
        pruned.update(kept_others)

        # 确保所有输入语料中出现的单字符都被保留（防御性编程）
        for ch in all_chars:
            if ch not in pruned:
                pruned[ch] = 1e-10

        # 重新归一化概率
        total = sum(pruned.values())
        if total > 0:
            pruned = {k: v / total for k, v in pruned.items()}
        return pruned

    def train(self, text: str):
        """
        在输入文本上训练Unigram模型。

        训练流程：
        1. 构建初始种子词汇表（所有子串）
        2. 初始化均匀概率
        3. EM迭代：E步（Viterbi分词统计）-> M步（概率重估计）-> 裁剪
        4. 最终得到词汇表和概率

        Args:
            text: 训练语料（原始字符串）。
        """
        # 将空格替换为"▁"，实现语言无关处理
        processed = text.replace(" ", "▁")

        # 记录语料中所有出现的单字符，用于后续裁剪时确保覆盖
        all_chars = set(processed)

        # 1. 构建种子词汇表
        seed_freq = self._build_seed_vocab(processed, max_subword_len=4)

        # 2. 初始化均匀概率
        vocab_size_seed = len(seed_freq)
        init_prob = 1.0 / vocab_size_seed
        probs = {token: init_prob for token in seed_freq}

        # 3. EM迭代
        for iteration in range(self.num_iterations):
            # 更新vocab（用于Viterbi）
            self.vocab = {}
            for idx, (token, prob) in enumerate(probs.items()):
                # 概率取对数，避免乘法下溢
                log_prob = math.log(prob + 1e-10)
                self.vocab[token] = (idx, log_prob)
                self.id_to_token[idx] = token

            # E步
            expected_counts, total_log_prob = self._expectation_step(processed)

            # M步
            new_probs = self._maximization_step(expected_counts)

            # 裁剪词汇表
            probs = self._prune_vocab(new_probs, all_chars)

            print(f"  迭代 {iteration + 1}/{self.num_iterations}, 语料log_prob={total_log_prob:.2f}, 词汇表大小={len(probs)}")

        # 最终更新
        self.vocab = {}
        for idx, (token, prob) in enumerate(probs.items()):
            log_prob = math.log(prob + 1e-10)
            self.vocab[token] = (idx, log_prob)
            self.id_to_token[idx] = token

    def encode(self, text: str) -> List[int]:
        """
        将输入文本编码为token_id序列。

        Args:
            text: 输入文本字符串。

        Returns:
            token_id列表。
        """
        processed = text.replace(" ", "▁")
        tokens, _ = self._viterbi_segment(processed)
        return [self.vocab[token][0] for token in tokens]

    def decode(self, token_ids: List[int]) -> str:
        """
        将token_id序列解码回原始文本。

        Args:
            token_ids: token_id列表。

        Returns:
            解码后的字符串。
        """
        chars = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, "")
            chars.append(token)
        text = "".join(chars)
        # 将"▁"还原为空格
        return text.replace("▁", " ")

    def vocab_size_current(self) -> int:
        """返回当前实际词汇表大小。"""
        return len(self.vocab)


if __name__ == "__main__":
    # 自测：用简单语料训练并验证编解码一致性
    corpus = (
        "low lower lowest new newer newest "
        "walk walking walked talks talking "
        "play playing played player players"
    )

    tokenizer = SentencePieceTokenizer(vocab_size=60, num_iterations=3)
    print("开始训练Unigram模型...")
    tokenizer.train(corpus)

    print("\n训练后词汇表大小:", tokenizer.vocab_size_current())

    test_text = "lower walking player"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"原文: {test_text}")
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")

    # 展示部分学到的词汇和概率
    print("\n部分词汇表（按概率排序前15个）:")
    sorted_vocab = sorted(tokenizer.vocab.items(), key=lambda x: x[1][1], reverse=True)
    for token, (tid, log_prob) in sorted_vocab[:15]:
        print(f"  {tid:3d}: '{token}'  log_prob={log_prob:.4f}")
