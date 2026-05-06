"""
Speculative Decoding（投机解码）
================================

投机解码是一种无需修改模型权重的 LLM 推理加速方案，通过"小模型猜测 + 大模型验证"的策略，
将串行自回归解码转化为准并行解码，实现 2-4 倍的吞吐提升。

核心思想：
    - 用轻量级草稿模型（Draft Model）自回归地快速生成 K 个候选 token
    - 用目标大模型对这 K 个候选 token 一次性验证
    - 从第一个位置开始逐个接受匹配的 token，遇不匹配则截断并重新采样

时间复杂度：O(K) 次草稿生成 + O(1) 次大模型验证 ≈ O(n) 但每个 batch 输出 K 个 token
空间复杂度：O(K * d) 草稿模型额外显存
面试频率：极高（2024-2025 LLM 推理优化必考）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class SimpleDraftModel(nn.Module):
    """
    简化的草稿模型（用于演示）。

    实际场景中，草稿模型通常是目标模型的缩小版本（如 Llama-70B 的草稿用 Llama-8B）。
    这里用一个轻量级 Transformer 来模拟。
    """

    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            _SimpleTransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.embedding(input_ids)
        for block in self.transformer_blocks:
            x = block(x)
        return self.lm_head(x)


class _SimpleTransformerBlock(nn.Module):
    """极简 Transformer 块，用于草稿模型。"""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), x)[0]
        x = x + self.ffn(self.norm2(x))
        return x


class SimpleTargetModel(nn.Module):
    """
    简化的目标大模型（用于演示）。

    实际场景中，这是真正的生产级大模型（如 Llama-70B）。
    这里用一个更深的 Transformer 来模拟。
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            _SimpleTransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.embedding(input_ids)
        for block in self.transformer_blocks:
            x = block(x)
        return self.lm_head(x)


class SpeculativeDecoder:
    """
    投机解码器。

    工作流程：
        1. 草稿阶段：用草稿模型自回归生成 K 个候选 token
        2. 验证阶段：用目标大模型对 K 个位置做一次前向传播
        3. 接受阶段：从位置 0 开始逐个比较，接受概率匹配的 token

    关键洞察：
        大模型一次前向传播的成本与生成 1 个 token 几乎相同（memory-bound），
        但这次前向可以同时算出 K 个位置的概率分布。
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        device: torch.device = torch.device("cpu"),
        max_draft_len: int = 5,
    ):
        self.draft = draft_model
        self.target = target_model
        self.device = device
        self.max_draft_len = max_draft_len
        self.draft.eval()
        self.target.eval()

    def _make_causal_mask(self, seq_len: int) -> torch.Tensor:
        """下三角因果掩码。"""
        return torch.tril(torch.ones(seq_len, seq_len, device=self.device)).bool()

    def _sample_token(self, logits: torch.Tensor, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        """从 logits 中采样一个 token。"""
        if temperature == 0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _verify_and_accept(
        self,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        验证并决定接受哪些 token。

        验证规则（Nezhaov et al., 2022）：
            对于位置 i 的草稿 token t：
                - 生成随机数 r ~ Uniform(0, 1)
                - 如果 r < target_prob[t]，则接受
                - 否则拒绝，并从 target_prob 重新采样

        这种方法保证输出分布与纯大模型生成完全一致（损失无损）。

        Args:
            draft_tokens: (batch, K) 草稿模型生成的 token
            draft_probs: (batch, K, vocab_size) 草稿模型在各位置的概率
            target_logits: (batch, K, vocab_size) 目标模型在各位置的 logits

        Returns:
            accepted_tokens: (batch, accepted_len) 被接受的 token
            n_accepted: 接受数量
            n_rejected: 拒绝数量
        """
        batch_size, K = draft_tokens.shape
        vocab_size = target_logits.shape[-1]

        target_probs = F.softmax(target_logits, dim=-1)

        accepted = torch.ones(batch_size, K, device=self.device, dtype=torch.bool)
        for i in range(K):
            draft_prob_i = draft_probs[:, i].gather(1, draft_tokens[:, i:i+1]).squeeze(-1)
            target_prob_i = target_probs[:, i].gather(1, draft_tokens[:, i:i+1]).squeeze(-1)

            acceptance_ratio = target_prob_i / (draft_prob_i + 1e-10)
            acceptance_ratio = acceptance_ratio.clamp(0, 1)

            r = torch.rand(batch_size, device=self.device)
            accepted[:, i] = r < acceptance_ratio

            if not accepted[:, i].all():
                first_reject = (~accepted[:, i]).nonzero(as_tuple=True)[0]
                if len(first_reject) > 0:
                    accepted[:, i + 1:] = False
                    break

        n_accepted = accepted.sum(dim=1).max().item()
        accepted_tokens = draft_tokens[:, :n_accepted]

        n_rejected = K - n_accepted

        return accepted_tokens, n_accepted, n_rejected

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[dict]]:
        """
        投机解码生成。

        Args:
            prompt_ids: (batch, prompt_len) 初始 prompt
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-K 采样

        Returns:
            generated_ids: (batch, total_len) 生成的完整序列
            stats: 每步的统计信息
        """
        self.draft.eval()
        self.target.eval()

        generated = prompt_ids.clone()
        total_stats = []
        total_accepted = 0
        total_drafted = 0

        for step in range(max_new_tokens):
            current_len = generated.shape[1]
            prompt_len = prompt_ids.shape[1]
            available_len = current_len - prompt_len

            if available_len >= max_new_tokens:
                break

            draft_tokens_list = []
            draft_probs_list = []

            draft_input = generated
            for k in range(self.max_draft_len):
                logits = self.draft(draft_input)
                next_token_logits = logits[:, -1, :]
                next_token = self._sample_token(next_token_logits, temperature, top_k)
                draft_tokens_list.append(next_token)

                probs = F.softmax(next_token_logits, dim=-1)
                draft_probs_list.append(probs)

                draft_input = torch.cat([draft_input, next_token], dim=1)

                if available_len + k + 1 >= max_new_tokens:
                    break

            K = len(draft_tokens_list)
            draft_tokens = torch.cat(draft_tokens_list, dim=1)

            target_logits = []
            for k in range(K):
                logits = self.target(generated)
                target_logits.append(logits[:, -1, :])
                next_token = self._sample_token(target_logits[-1], temperature, top_k)
                generated = torch.cat([generated, next_token], dim=1)

            target_logits = torch.stack(target_logits, dim=1)
            draft_probs = torch.stack(draft_probs_list, dim=1)

            accepted_tokens, n_accepted, n_rejected = self._verify_and_accept(
                draft_tokens, draft_probs, target_logits
            )

            total_accepted += n_accepted
            total_drafted += K

            stats = {
                "step": step,
                "draft_len": K,
                "n_accepted": n_accepted,
                "n_rejected": n_rejected,
                "accept_rate": n_accepted / K if K > 0 else 0,
            }
            total_stats.append(stats)

            if n_rejected > 0:
                break

        final_stats = {
            "total_generated": generated.shape[1] - prompt_ids.shape[1],
            "total_drafted": total_drafted,
            "total_accepted": total_accepted,
            "overall_accept_rate": total_accepted / total_drafted if total_drafted > 0 else 0,
            "steps": total_stats,
        }

        return generated, final_stats


if __name__ == "__main__":
    VOCAB_SIZE = 500
    DEVICE = torch.device("cpu")
    MAX_DRAFT_LEN = 3
    MAX_NEW_TOKENS = 15

    draft = SimpleDraftModel(VOCAB_SIZE, d_model=64, n_heads=2, n_layers=1)
    target = SimpleTargetModel(VOCAB_SIZE, d_model=128, n_heads=4, n_layers=2)

    decoder = SpeculativeDecoder(draft, target, device=DEVICE, max_draft_len=MAX_DRAFT_LEN)

    prompt = torch.randint(1, VOCAB_SIZE, (1, 5))
    print(f"[Speculative Decoding] Prompt length: {prompt.shape[1]}")

    generated, stats = decoder.generate(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.8,
    )

    print(f"[Speculative Decoding] Generated length: {generated.shape[1]}")
    print(f"[Speculative Decoding] Stats:")
    print(f"  - Total drafted tokens: {stats['total_drafted']}")
    print(f"  - Total accepted tokens: {stats['total_accepted']}")
    print(f"  - Overall accept rate: {stats['overall_accept_rate']:.2%}")
    print(f"  - Speedup factor: ~{stats['overall_accept_rate'] * MAX_DRAFT_LEN + (1 - stats['overall_accept_rate']):.2f}x")

    print("\nStep-by-step breakdown:")
    for step_stats in stats["steps"]:
        print(f"  Step {step_stats['step']}: drafted={step_stats['draft_len']}, "
              f"accepted={step_stats['n_accepted']}, "
              f"rate={step_stats['accept_rate']:.2%}")
