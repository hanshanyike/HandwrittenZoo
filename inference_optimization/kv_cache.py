"""
KV Cache (Key-Value Cache)

Algorithm: KV Cache mechanism for transformer autoregressive inference optimization.
Core Idea: Trade space for time — store previously computed Key and Value tensors
    to avoid redundant recomputation during token-by-token generation.
Time Complexity: O(1) incremental per step vs O(n) without cache.
Space Complexity: O(batch * layers * heads * seq_len * head_dim).
Interview Frequency: High — fundamental optimization in every LLM serving system.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCache:
    """
    自回归生成中的 KV 缓存容器。

    在逐 token 生成时，Transformer 的 Attention 需要当前 token 与所有历史 token 做交互。
    若不缓存，每一步都要重新计算所有历史 token 的 K、V，复杂度为 O(n^3)。
    KV Cache 将历史 K、V 张量保留在显存中，新 token 只需计算自身的 K、V 并追加到缓存，
    使单步复杂度降为 O(1)（相对已生成长度）。
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        """
        初始化固定大小的 KV 缓存缓冲区。

        Args:
            batch_size: 批次大小
            max_seq_len: 最大序列长度（预分配显存的上限）
            num_kv_heads: KV 头的数量（GQA/MQA 中可能小于 Q 头数）
            head_dim: 每个头的维度
            dtype: 数据类型，通常与模型权重一致
            device: 计算设备，默认 CPU
        """
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device or torch.device("cpu")

        # 预分配固定显存，避免生成过程中频繁 malloc 导致碎片
        # 形状: (batch, num_kv_heads, max_seq_len, head_dim)
        self.register_buffer("cache_k", None)
        self.register_buffer("cache_v", None)
        self._init_buffers()

        # 记录当前已填充的序列长度，初始为 0
        self.current_len = 0

    def _init_buffers(self):
        """分配零初始化的 K/V 缓存张量。"""
        shape = (self.batch_size, self.num_kv_heads, self.max_seq_len, self.head_dim)
        self.cache_k = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.cache_v = torch.zeros(shape, dtype=self.dtype, device=self.device)

    def reset(self):
        """重置缓存，用于开启新一轮对话或新的 batch。"""
        self.current_len = 0
        self.cache_k.zero_()
        self.cache_v.zero_()

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将新生成的 K、V 写入缓存并返回完整的 K、V。

        Args:
            k_new: 新 token 的 Key，形状 (batch, num_kv_heads, new_len, head_dim)
            v_new: 新 token 的 Value，形状与 k_new 相同

        Returns:
            cache_k: 更新后的完整 Key 缓存，有效区域为 [:, :, :current_len, :]
            cache_v: 更新后的完整 Value 缓存
        """
        batch_size, num_kv_heads, new_len, head_dim = k_new.shape
        assert batch_size == self.batch_size
        assert num_kv_heads == self.num_kv_heads
        assert head_dim == self.head_dim
        assert self.current_len + new_len <= self.max_seq_len, (
            f"KV Cache overflow: {self.current_len + new_len} > {self.max_seq_len}"
        )

        # 将新 K/V 拷贝到预分配缓冲区的对应位置
        start = self.current_len
        end = start + new_len
        self.cache_k[:, :, start:end, :] = k_new
        self.cache_v[:, :, start:end, :] = v_new

        # 更新已填充长度
        self.current_len = end

        # 返回完整缓存，调用方可直接用于 Attention 计算
        return self.cache_k[:, :, :self.current_len, :], self.cache_v[:, :, :self.current_len, :]


class CachedMultiHeadAttention(nn.Module):
    """
    带 KV Cache 的多头自注意力层（简化版，用于演示推理优化）。

    训练时等价于标准 MHA；推理时通过 cache 避免重复计算历史 K/V。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        max_seq_len: int = 2048,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 支持 GQA：若 num_kv_heads 未指定则默认等于 num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        # Q/K/V 投影矩阵
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # 推理时才会实例化的 KV Cache，训练时为 None
        self._kv_cache: Optional[KVCache] = None

    def _init_kv_cache(self, batch_size: int, device: torch.device):
        """在首次推理前根据 batch_size 和设备初始化缓存。"""
        self._kv_cache = KVCache(
            batch_size=batch_size,
            max_seq_len=self.max_seq_len,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=device,
        )

    def reset_cache(self):
        """暴露给调用方，用于对话轮次切换时清空缓存。"""
        if self._kv_cache is not None:
            self._kv_cache.reset()

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播。

        Args:
            x: 输入张量，训练时形状 (batch, seq_len, embed_dim)；
               推理时通常为 (batch, 1, embed_dim) 或首 token 的 (batch, prompt_len, embed_dim)
            use_cache: 是否启用 KV Cache，推理时应设为 True

        Returns:
            out: Attention 输出，形状 (batch, seq_len, embed_dim)
            past_kv: 若 use_cache=True，返回更新后的 (K, V) 元组，否则为 None
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # 线性投影得到 Q/K/V
        q = self.q_proj(x)  # (batch, seq_len, embed_dim)
        k = self.k_proj(x)  # (batch, seq_len, num_kv_heads * head_dim)
        v = self.v_proj(x)  # 同上

        # reshape 为多头形状
        # Q 需要与 num_heads 对齐；K/V 可能因 GQA 而头数更少
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # 此时 q: (batch, num_heads, seq_len, head_dim)
        #     k/v: (batch, num_kv_heads, seq_len, head_dim)

        if use_cache:
            if self._kv_cache is None or self._kv_cache.device != device:
                self._init_kv_cache(batch_size, device)

            # 将当前步的 K/V 追加到缓存，并取回完整历史 K/V
            k_full, v_full = self._kv_cache.update(k, v)
        else:
            k_full, v_full = k, v

        # 若使用 GQA，需要将 K/V 扩展以匹配 Q 的头数
        if self.num_kv_heads != self.num_heads:
            # 每个 KV 头服务 num_heads // num_kv_heads 个 Q 头
            num_groups = self.num_heads // self.num_kv_heads
            k_full = k_full.repeat_interleave(num_groups, dim=1)
            v_full = v_full.repeat_interleave(num_groups, dim=1)

        # 标准缩放点积注意力
        # scores: (batch, num_heads, seq_len, kv_len)
        scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v_full)  # (batch, num_heads, seq_len, head_dim)

        # 合并多头并投影输出
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.o_proj(out)

        past_kv = (self._kv_cache.cache_k[:, :, :self._kv_cache.current_len, :],
                   self._kv_cache.cache_v[:, :, :self._kv_cache.current_len, :]) if use_cache else None
        return out, past_kv


if __name__ == "__main__":
    # 简单自测：验证 KV Cache 在逐 token 生成时输出与全序列前向一致
    torch.manual_seed(42)
    embed_dim = 64
    num_heads = 8
    batch_size = 2
    prompt_len = 5
    gen_len = 3

    mha = CachedMultiHeadAttention(embed_dim, num_heads, max_seq_len=32)
    mha.eval()

    # 构造一个 prompt
    prompt = torch.randn(batch_size, prompt_len, embed_dim)

    # 方式一：一次性前向（无 cache）
    with torch.no_grad():
        out_full, _ = mha(prompt, use_cache=False)

    # 方式二：先过 prompt，再逐 token 生成，使用 KV Cache
    mha.reset_cache()
    with torch.no_grad():
        out_prompt, _ = mha(prompt, use_cache=True)
        # 逐 token 追加
        generated = []
        for _ in range(gen_len):
            next_token = torch.randn(batch_size, 1, embed_dim)
            out_step, _ = mha(next_token, use_cache=True)
            generated.append(out_step)

    # 验证 prompt 部分输出一致
    assert torch.allclose(out_full, out_prompt, atol=1e-5), "Prompt outputs mismatch!"
    print("KV Cache self-test passed: prompt outputs match full forward.")

    # 打印缓存信息
    cache = mha._kv_cache
    print(f"Cache shape: {cache.cache_k.shape}")
    print(f"Current filled length: {cache.current_len} (expected {prompt_len + gen_len})")
