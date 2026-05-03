"""
Byte Pair Encoding (BPE) Tokenizer — GPT系列核心分词算法

核心思想：
    从字符级（或字节级）初始词汇表出发，迭代合并语料中出现频率最高的相邻token对，
    直到达到目标词汇表大小。通过高频合并，常用词和词缀被压缩为单个token，
    罕见词则被拆分为多个子词token，从而平衡词汇表大小与序列长度。

时间复杂度：
    - 训练：O(N * M)，N为语料总长度，M为合并轮数
    - 编码：O(L * M)，L为输入文本长度，M为合并规则数

空间复杂度：
    - O(V + M)，V为词汇表大小，M为合并规则数

面试频率：高（GPT/LLM相关岗位必问）
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class BPETokenizer:
    """
    基于字节对编码（BPE）的子词分词器实现。

    参考：Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units", 2015
    """

    def __init__(self, vocab_size: int = 256 + 100):
        """
        初始化BPE分词器。

        Args:
            vocab_size: 目标词汇表大小。默认256（字节） + 100次合并。
        """
        # 基础字节词汇表：0-255对应所有可能的字节
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256

        # vocab: token_id -> bytes 的映射
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        # merges: (token_a, token_b) -> merged_token_id
        self.merges: Dict[Tuple[int, int], int] = {}

    def _get_stats(self, token_ids: List[int]) -> Dict[Tuple[int, int], int]:
        """
        统计相邻token对的出现频率。

        Args:
            token_ids: 当前token序列（已用当前词汇表编码）。

        Returns:
            每个相邻token对及其出现次数的字典。
        """
        counts = defaultdict(int)
        # 遍历序列，统计所有相邻pair
        for i in range(len(token_ids) - 1):
            pair = (token_ids[i], token_ids[i + 1])
            counts[pair] += 1
        return counts

    def _merge(self, token_ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """
        将token_ids中所有指定的pair替换为新的token_id。

        Args:
            token_ids: 当前token序列。
            pair: 待合并的token对。
            new_id: 合并后的新token_id。

        Returns:
            合并后的新token序列。
        """
        new_ids = []
        i = 0
        while i < len(token_ids):
            # 如果当前和下一个token正好构成待合并的pair，则合并
            if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair:
                new_ids.append(new_id)
                i += 2  # 跳过两个token
            else:
                new_ids.append(token_ids[i])
                i += 1
        return new_ids

    def train(self, text: str):
        """
        在输入文本上训练BPE模型，学习合并规则。

        Args:
            text: 训练语料（原始字符串）。
        """
        # 将文本编码为原始字节序列，这是字节级BPE的核心：任何Unicode字符都可表示
        token_ids = list(text.encode("utf-8"))

        # 迭代执行合并操作
        for merge_idx in range(self.num_merges):
            stats = self._get_stats(token_ids)
            if not stats:
                break

            # 选择出现频率最高的pair进行合并（贪心策略）
            best_pair = max(stats, key=stats.get)
            new_id = 256 + merge_idx

            # 将合并后的字节串加入词汇表
            merged_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = merged_bytes
            self.merges[best_pair] = new_id

            # 在整个语料上应用本次合并
            token_ids = self._merge(token_ids, best_pair, new_id)

    def encode(self, text: str) -> List[int]:
        """
        将输入文本编码为token_id序列。

        编码策略：对原始字节序列，按合并规则的优先级（训练顺序）依次应用合并。
        实际实现中，为了效率，通常采用"找优先级最高（即最早训练）的可合并pair"策略。

        Args:
            text: 输入文本字符串。

        Returns:
            token_id列表。
        """
        token_ids = list(text.encode("utf-8"))

        # 当还能继续合并时，持续处理
        while len(token_ids) >= 2:
            stats = self._get_stats(token_ids)
            # 只考虑已学习的合并规则，并选取优先级最高（即训练顺序最早）的pair
            # 用min是因为merge_idx越小，优先级越高（我们按顺序存储在merges中）
            best_pair = None
            min_rank = float("inf")
            for pair in stats:
                if pair in self.merges:
                    rank = list(self.merges.keys()).index(pair)
                    if rank < min_rank:
                        min_rank = rank
                        best_pair = pair

            if best_pair is None:
                break

            new_id = self.merges[best_pair]
            token_ids = self._merge(token_ids, best_pair, new_id)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """
        将token_id序列解码回原始文本。

        Args:
            token_ids: token_id列表。

        Returns:
            解码后的字符串。
        """
        # 将所有token对应的字节串拼接，再用utf-8解码
        byte_array = bytearray()
        for tid in token_ids:
            byte_array.extend(self.vocab[tid])
        return byte_array.decode("utf-8", errors="replace")

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

    tokenizer = BPETokenizer(vocab_size=256 + 20)
    tokenizer.train(corpus)

    print("训练后词汇表大小:", tokenizer.vocab_size_current())
    print("合并规则数:", len(tokenizer.merges))

    test_text = "lower walking player"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)

    print(f"原文: {test_text}")
    print(f"编码: {encoded}")
    print(f"解码: {decoded}")
    print(f"一致性检查: {'通过' if decoded == test_text else '失败'}")

    # 展示部分学到的合并规则
    print("\n部分合并规则（前5条）:")
    for i, (pair, new_id) in enumerate(tokenizer.merges.items()):
        if i >= 5:
            break
        print(f"  {pair} -> {new_id} ({tokenizer.vocab[new_id]})")
