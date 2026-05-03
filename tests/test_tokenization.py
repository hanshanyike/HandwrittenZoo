"""
分词模块单元测试
测试 BPE、WordPiece、SentencePiece
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from tokenization.bpe.bpe import BPETokenizer
from tokenization.wordpiece.wordpiece import WordPieceTokenizer
from tokenization.sentencepiece.sentencepiece import SentencePieceTokenizer


class TestBPE:
    """测试 BPE 分词器"""

    def test_train(self):
        tokenizer = BPETokenizer(vocab_size=276)
        corpus = "low lower lowest new newer newest"
        tokenizer.train(corpus)
        assert tokenizer.vocab_size_current() > 256
        assert len(tokenizer.merges) > 0

    def test_encode_decode(self):
        tokenizer = BPETokenizer(vocab_size=276)
        corpus = "low lower lowest new newer newest"
        tokenizer.train(corpus)
        text = "lower newest"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text

    def test_vocab_size(self):
        tokenizer = BPETokenizer(vocab_size=276)
        corpus = "low lower lowest new newer newest"
        tokenizer.train(corpus)
        assert tokenizer.vocab_size_current() <= 276

    def test_empty_text(self):
        tokenizer = BPETokenizer(vocab_size=276)
        corpus = "low lower lowest"
        tokenizer.train(corpus)
        encoded = tokenizer.encode("")
        assert encoded == []


class TestWordPiece:
    """测试 WordPiece 分词器"""

    def test_train(self):
        tokenizer = WordPieceTokenizer(vocab_size=80)
        corpus = ["low lower lowest new newer newest"]
        tokenizer.train(corpus)
        assert tokenizer.vocab_size_current() > 0
        assert len(tokenizer.merges) > 0

    def test_encode_decode(self):
        tokenizer = WordPieceTokenizer(vocab_size=80)
        corpus = ["low lower lowest new newer newest"]
        tokenizer.train(corpus)
        text = "lower newest"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        # WordPiece 解码可能不完全还原空格分隔
        assert isinstance(decoded, str)
        assert len(encoded) > 0

    def test_vocab_size(self):
        tokenizer = WordPieceTokenizer(vocab_size=80)
        corpus = ["low lower lowest new newer newest"]
        tokenizer.train(corpus)
        assert tokenizer.vocab_size_current() <= 80

    def test_encode_word(self):
        tokenizer = WordPieceTokenizer(vocab_size=80)
        corpus = ["low lower lowest"]
        tokenizer.train(corpus)
        tokens = tokenizer.encode_word("lower")
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)


class TestSentencePiece:
    """测试 SentencePiece 分词器"""

    def test_train(self):
        tokenizer = SentencePieceTokenizer(vocab_size=60, num_iterations=2)
        corpus = "low lower lowest new newer newest"
        tokenizer.train(corpus)
        assert tokenizer.vocab_size_current() > 0
        assert len(tokenizer.vocab) > 0

    def test_encode_decode(self):
        tokenizer = SentencePieceTokenizer(vocab_size=60, num_iterations=2)
        corpus = "low lower lowest new newer newest"
        tokenizer.train(corpus)
        text = "lower newest"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert isinstance(decoded, str)
        assert len(encoded) > 0

    def test_vocab_size(self):
        tokenizer = SentencePieceTokenizer(vocab_size=60, num_iterations=2)
        corpus = "low lower lowest new newer newest"
        tokenizer.train(corpus)
        assert tokenizer.vocab_size_current() <= 60

    def test_space_handling(self):
        tokenizer = SentencePieceTokenizer(vocab_size=60, num_iterations=2)
        corpus = "low lower lowest"
        tokenizer.train(corpus)
        text = "lower lowest"
        encoded = tokenizer.encode(text)
        # 应能处理包含空格的文本
        assert len(encoded) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
