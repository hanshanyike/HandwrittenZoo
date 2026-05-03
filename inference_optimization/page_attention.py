"""
PageAttention (Simplified)

Algorithm: Paged KV Cache management inspired by vLLM.
Core Idea: Decouple logical token sequence from physical memory blocks —
    allocate fixed-size physical blocks on demand and map them via a page table,
    eliminating internal/external fragmentation and enabling memory sharing.
Time Complexity: O(1) block allocation; attention kernel reads blocks via page table.
Space Complexity: O(total_blocks * block_size * kv_hidden_size) — near-zero waste.
Interview Frequency: High — cornerstone of modern LLM serving engines.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockAllocator:
    """
    物理块分配器：管理固定大小的物理块池，按需分配与回收。

    类比操作系统虚拟内存：逻辑块由请求持有，物理块由分配器管理，
    通过页表（page table）将逻辑块号映射到物理块号。
    """

    def __init__(self, num_blocks: int, block_size: int):
        """
        Args:
            num_blocks: 物理块总数，由可用显存 / 每块大小决定
            block_size: 每个块能容纳的 token 数量
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        # 空闲物理块 ID 集合，初始时所有块都空闲
        self.free_blocks = set(range(num_blocks))
        # 记录每个物理块被多少个逻辑序列引用（用于 copy-on-write 共享）
        self.ref_count: Dict[int, int] = {i: 0 for i in range(num_blocks)}

    def allocate(self, num_needed: int = 1) -> List[int]:
        """
        分配指定数量的物理块。

        Returns:
            物理块 ID 列表；若不足则返回已分配的部分（调用方需处理 OOM）
        """
        allocated = []
        for _ in range(num_needed):
            if not self.free_blocks:
                break
            block_id = self.free_blocks.pop()
            self.ref_count[block_id] = 1
            allocated.append(block_id)
        return allocated

    def free(self, block_ids: List[int]):
        """
        释放物理块。采用引用计数：仅当引用降为 0 时才回收到空闲池。

        Args:
            block_ids: 要释放的物理块 ID 列表
        """
        for bid in block_ids:
            self.ref_count[bid] -= 1
            if self.ref_count[bid] <= 0:
                self.free_blocks.add(bid)
                self.ref_count[bid] = 0

    def incr_ref(self, block_ids: List[int]):
        """增加引用计数，用于 fork（如 beam search 多分支共享前缀）。"""
        for bid in block_ids:
            self.ref_count[bid] += 1


class PagedKVCache:
    """
    分页式 KV Cache：每个请求维护一张页表，将逻辑块映射到物理块。

    核心优势：
    1. 消除内部碎片：只在需要时分配块，块内未用满的空间不会阻碍其他请求；
    2. 消除外部碎片：固定大小块池化管理，无传统连续内存分配中的空洞；
    3. 支持共享：多个请求可通过引用计数共享同一块（如 beam search、并行解码）。
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device or torch.device("cpu")

        # 物理块池：形状 (num_blocks, block_size, num_kv_heads, head_dim)
        # 这里将 block_size 放在 seq 维度，便于按块读写
        self.k_cache = torch.zeros(
            (num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=self.device,
        )
        self.v_cache = torch.zeros_like(self.k_cache)

        self.allocator = BlockAllocator(num_blocks, block_size)

        # 每个请求（序列）的页表：逻辑块索引 -> 物理块 ID
        self.page_tables: Dict[int, List[int]] = {}
        # 每个请求当前已填充的 token 数（用于计算逻辑块号 + 块内偏移）
        self.seq_lengths: Dict[int, int] = {}
        self._seq_counter = 0

    def add_sequence(self, init_len: int = 0) -> int:
        """
        注册一个新请求序列，返回 seq_id。

        Args:
            init_len: 初始 prompt 长度，用于预分配所需块
        """
        seq_id = self._seq_counter
        self._seq_counter += 1

        num_blocks_needed = (init_len + self.block_size - 1) // self.block_size
        block_ids = self.allocator.allocate(num_blocks_needed)
        self.page_tables[seq_id] = block_ids
        self.seq_lengths[seq_id] = init_len
        return seq_id

    def fork_sequence(self, parent_seq_id: int) -> int:
        """
        Fork 一个序列（如 beam search 展开），父子共享已有物理块。

        Returns:
            新的 seq_id
        """
        child_id = self._seq_counter
        self._seq_counter += 1

        parent_blocks = self.page_tables[parent_seq_id]
        self.allocator.incr_ref(parent_blocks)
        self.page_tables[child_id] = list(parent_blocks)
        self.seq_lengths[child_id] = self.seq_lengths[parent_seq_id]
        return child_id

    def remove_sequence(self, seq_id: int):
        """结束一个请求，释放其占用的物理块引用。"""
        if seq_id in self.page_tables:
            self.allocator.free(self.page_tables[seq_id])
            del self.page_tables[seq_id]
            del self.seq_lengths[seq_id]

    def _logical_to_physical(self, seq_id: int, logical_token_idx: int) -> Tuple[int, int]:
        """
        将逻辑 token 索引映射为物理块 ID 和块内偏移。

        Returns:
            (physical_block_id, offset_in_block)
        """
        logical_block_idx = logical_token_idx // self.block_size
        offset = logical_token_idx % self.block_size
        physical_block_id = self.page_tables[seq_id][logical_block_idx]
        return physical_block_id, offset

    def append(self, seq_id: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """
        向指定序列追加新的 K/V token。

        Args:
            seq_id: 序列 ID
            k_new: 新 token 的 Key，形状 (num_kv_heads, head_dim) 或 (new_len, num_kv_heads, head_dim)
            v_new: 新 token 的 Value，形状同 k_new
        """
        if k_new.dim() == 2:
            k_new = k_new.unsqueeze(0)
            v_new = v_new.unsqueeze(0)
        new_len = k_new.shape[0]

        start_len = self.seq_lengths[seq_id]
        for i in range(new_len):
            token_idx = start_len + i
            logical_block_idx = token_idx // self.block_size

            # 若当前逻辑块尚未分配物理块，则申请一个新块
            if logical_block_idx >= len(self.page_tables[seq_id]):
                new_blocks = self.allocator.allocate(1)
                if not new_blocks:
                    raise RuntimeError("Out of physical blocks in PagedKVCache")
                self.page_tables[seq_id].append(new_blocks[0])

            physical_block_id, offset = self._logical_to_physical(seq_id, token_idx)
            self.k_cache[physical_block_id, offset] = k_new[i]
            self.v_cache[physical_block_id, offset] = v_new[i]

        self.seq_lengths[seq_id] += new_len

    def get_kv(self, seq_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        按逻辑顺序取出某序列的完整 K/V。

        Returns:
            k, v: 形状 (seq_len, num_kv_heads, head_dim)
        """
        seq_len = self.seq_lengths[seq_id]
        k_out = torch.zeros((seq_len, self.num_kv_heads, self.head_dim), dtype=self.dtype, device=self.device)
        v_out = torch.zeros_like(k_out)

        for token_idx in range(seq_len):
            physical_block_id, offset = self._logical_to_physical(seq_id, token_idx)
            k_out[token_idx] = self.k_cache[physical_block_id, offset]
            v_out[token_idx] = self.v_cache[physical_block_id, offset]
        return k_out, v_out


class PagedAttentionLayer(nn.Module):
    """
    演示如何将 PagedKVCache 集成到 Attention 层中。

    实际 vLLM 中，Attention kernel 会直接在 GPU 上根据 page table 读取分散的块，
    这里为了可读性，先在 Python 侧 gather 成连续张量再计算 Attention。
    """

    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, embed_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        paged_cache: PagedKVCache,
        seq_id: int,
    ) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状 (1, seq_len, embed_dim) 或 (seq_len, embed_dim)
            paged_cache: 分页缓存管理器
            seq_id: 当前请求 ID

        Returns:
            out: Attention 输出
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size, seq_len, embed_dim = x.shape
        assert batch_size == 1, "PagedAttentionLayer demo only supports batch_size=1"

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # 将新 K/V 写入分页缓存
        # 去掉 batch 维度，转为 (seq_len, num_kv_heads, head_dim)
        paged_cache.append(seq_id, k.squeeze(0).transpose(0, 1), v.squeeze(0).transpose(0, 1))

        # 从缓存读取完整历史 K/V（实际 vLLM 中这一步在 kernel 内完成，避免数据传输）
        k_full, v_full = paged_cache.get_kv(seq_id)  # (seq_len_total, num_kv_heads, head_dim)
        k_full = k_full.unsqueeze(0).transpose(1, 2)  # (1, num_kv_heads, seq_len_total, head_dim)
        v_full = v_full.unsqueeze(0).transpose(1, 2)

        # GQA：扩展 KV 头数以匹配 Q 头数
        if self.num_kv_heads != self.num_heads:
            num_groups = self.num_heads // self.num_kv_heads
            k_full = k_full.repeat_interleave(num_groups, dim=1)
            v_full = v_full.repeat_interleave(num_groups, dim=1)

        scores = torch.matmul(q, k_full.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_full)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)


if __name__ == "__main__":
    torch.manual_seed(42)
    embed_dim = 64
    num_heads = 8
    num_kv_heads = 4
    head_dim = embed_dim // num_heads
    block_size = 4
    num_blocks = 16

    # 初始化分页缓存和注意力层
    cache = PagedKVCache(num_blocks, block_size, num_kv_heads, head_dim)
    layer = PagedAttentionLayer(embed_dim, num_heads, num_kv_heads, head_dim)
    layer.eval()

    # 模拟两个并发请求
    seq_a = cache.add_sequence(init_len=0)
    seq_b = cache.add_sequence(init_len=0)

    with torch.no_grad():
        # 请求 A 生成 5 个 token（跨越 2 个 block）
        for _ in range(5):
            tok = torch.randn(1, 1, embed_dim)
            layer(tok, cache, seq_a)

        # 请求 B 生成 3 个 token
        for _ in range(3):
            tok = torch.randn(1, 1, embed_dim)
            layer(tok, cache, seq_b)

    print(f"Seq A length: {cache.seq_lengths[seq_a]}, blocks: {cache.page_tables[seq_a]}")
    print(f"Seq B length: {cache.seq_lengths[seq_b]}, blocks: {cache.page_tables[seq_b]}")

    # 测试 fork：seq_c 共享 seq_a 的前缀
    seq_c = cache.fork_sequence(seq_a)
    print(f"Seq C (forked from A) length: {cache.seq_lengths[seq_c]}, blocks: {cache.page_tables[seq_c]}")
    print(f"Ref counts of A's blocks: {[cache.allocator.ref_count[b] for b in cache.page_tables[seq_a]]}")

    # 回收
    cache.remove_sequence(seq_a)
    cache.remove_sequence(seq_b)
    cache.remove_sequence(seq_c)
    print(f"Free blocks after cleanup: {len(cache.allocator.free_blocks)} / {num_blocks}")
    print("PageAttention self-test passed.")
