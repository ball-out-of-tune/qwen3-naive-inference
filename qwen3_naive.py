from dataclasses import dataclass
from functools import lru_cache
import json
import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
import triton
import triton.language as tl


@triton.jit
def _store_kvcache_kernel(key_ptr, key_stride, value_ptr, value_stride,
                           k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
                           D: tl.constexpr):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key, value, k_cache, v_cache, slot_mapping):
    """将 K/V 写入全局 cache: 一次 kernel, 零中间 tensor, 零 dtype cast."""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    _store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0),
                                 k_cache, v_cache, slot_mapping, D)


@dataclass
class VarlenContext:
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    positions: torch.Tensor


_VARLEN_CTX = None


def set_varlen_context(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, positions):
    global _VARLEN_CTX
    _VARLEN_CTX = VarlenContext(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, positions)


def get_varlen_context():
    return _VARLEN_CTX


def clear_varlen_context():
    global _VARLEN_CTX
    _VARLEN_CTX = None


@dataclass
class Qwen3Config:
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    vocab_size: int = 151936
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000
    attention_bias: bool = False
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    dtype: str = "bfloat16"

    @classmethod
    def from_json(cls, path):
        with open(path, "r") as f:
            d = json.load(f)
        return cls(
            hidden_size=d["hidden_size"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            head_dim=d.get("head_dim", d["hidden_size"] // d["num_attention_heads"]),
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            vocab_size=d["vocab_size"],
            max_position_embeddings=d["max_position_embeddings"],
            rms_norm_eps=d["rms_norm_eps"],
            rope_theta=d.get("rope_theta", 1000000),
            attention_bias=d.get("attention_bias", False),
            hidden_act=d["hidden_act"],
            tie_word_embeddings=d.get("tie_word_embeddings", True),
            bos_token_id=d.get("bos_token_id", 151643),
            eos_token_id=d.get("eos_token_id", 151645),
        )

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def _rms_norm(self, x):
        orig_dtype = x.dtype
        x_f32 = x.float()
        x_f32 = x_f32 * torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f32 * self.weight.float()).to(orig_dtype)

    @torch.compile
    def _add_rms_norm(self, x, residual):
        orig_dtype = x.dtype
        x_f32 = x.float() + residual.float()
        residual = x_f32.to(orig_dtype)
        x_f32 = x_f32 * torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f32 * self.weight.float()).to(orig_dtype), residual

    def forward(self, x, residual=None):
        if residual is None:
            return self._rms_norm(x)
        else:
            return self._add_rms_norm(x, residual)


class BadRMSNorm:
    """不继承 nn.Module —— 演示用"""
    def __init__(self, hidden_size, eps=1e-6):
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ===================== Continuous Batching 组件 =====================

class Sequence:
    """一条推理请求的完整状态"""

    BLOCK_SIZE = 256

    def __init__(self, token_ids: list[int], sampling_params: dict = None):
        self.token_ids = list(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.last_token = token_ids[-1]
        self.status = "WAITING"       # WAITING → RUNNING → FINISHED
        self.block_table = []         # 页表: [3, 7, 12] — 逻辑块→物理块
        self.kv_len = 0                # 已写入 KV cache 的 token 数 (含 prefix cache + prefill + decode)
        self.num_scheduled_tokens = 0 # 本轮要计算的 token 数 (prefill chunk 或 decode=1)
        sp = sampling_params or {}
        self.temperature = sp.get("temperature", 0.6)
        self.max_tokens = sp.get("max_tokens", 256)
        self.response_log_probs: list[float] = []  # GRPO: 每个 response token 采样时的 logP

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id

    @property
    def num_tokens(self):
        return len(self.token_ids)

    @property
    def num_blocks(self):
        return (len(self.token_ids) + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE

    def block(self, i: int) -> list[int]:
        """返回第 i 个逻辑块的 token ID 列表，用于 prefix caching hash"""
        start = i * self.BLOCK_SIZE
        end = min(start + self.BLOCK_SIZE, len(self.token_ids))
        return self.token_ids[start:end]

    @property
    def num_completion_tokens(self):
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def unfinished(self):
        return self.status != "FINISHED"

    @property
    def prompt_len(self):
        return self.num_prompt_tokens


class BlockManager:
    """页式 KV cache 管理：按需分配 256-token 的块，支持 prefix caching。

    核心数据结构:
      free:           空闲物理块队列
      ref_count:      每个 block 被多少个 seq 引用 (prefix caching 共享)
      block_hash:     每个 block 的链式 hash (基于 token ID + 前一个 block hash)
      block_tokens:   每个 block 的 token ID 副本 (防 hash 碰撞)
      hash_to_block:  全局查找表 {chain_hash: block_id}
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free = list(range(num_blocks))
        self.ref_count = [0] * num_blocks
        self.block_hash = [-1] * num_blocks
        self.block_tokens = [None] * num_blocks
        self.hash_to_block: dict[int, int] = {}

    @staticmethod
    def compute_hash(token_ids: list[int], prefix_hash: int = -1) -> int:
        """链式哈希: 每个 block 的 hash 混合了前一个 block 的 hash。
        确保 block N 的共享只有在整个前缀全部一致时才发生。"""
        h = prefix_hash if prefix_hash != -1 else 0
        for t in token_ids:
            h = ((h << 5) + h) ^ t      # djb2 变体
        return h & 0xFFFFFFFFFFFFFFFF    # 64-bit

    # ================================================================
    # Prefix caching 核心: allocate 先查 hash 表共享, 不够再分配新块
    # ================================================================
    def allocate(self, seq: Sequence):
        """分配块给序列，优先共享已缓存的前缀 block。"""
        assert not seq.block_table, "block_table 应该为空"

        h = -1
        num_cached = 0

        # ① 遍历满 block (最后一个可能不满的跳过), 检查能否共享
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            blk = self.hash_to_block.get(h, -1)

            # 检查 hash 命中 + 确认为同一段 token (防碰撞)
            if blk == -1 or self.block_tokens[blk] != token_ids:
                break

            num_cached += 1
            # block 在 free 列表中 (之前的 seq 已释放但 hash 还在)
            if blk in self.free:
                self.free.remove(blk)
            self.ref_count[blk] += 1
            seq.block_table.append(blk)

        # ② 剩余块从 free 分配
        for i in range(num_cached, seq.num_blocks):
            blk = self.free.pop()
            # 如果这个 free block 之前有 hash (旧缓存), 清除
            if self.block_hash[blk] != -1:
                old_h = self.block_hash[blk]
                if self.hash_to_block.get(old_h) == blk:
                    del self.hash_to_block[old_h]
                self.block_hash[blk] = -1
                self.block_tokens[blk] = None
            self.ref_count[blk] = 1
            seq.block_table.append(blk)

        seq.kv_len = num_cached * self.block_size

    # ================================================================
    # deallocate: ref_count--, 降到 0 归还 free (但保留 hash 供复用!)
    # ================================================================
    def deallocate(self, block_table: list[int]):
        """归还块: 减引用计数, 归零后放回 free。hash 不删除, 后续可能被复用。"""
        for b in block_table:
            self.ref_count[b] -= 1
            if self.ref_count[b] == 0:
                self.free.append(b)        # hash 和 block_tokens 保留!

    # ================================================================
    # decode 时按需追加 block
    # ================================================================
    def can_append(self, seq: Sequence) -> bool:
        """是否需要新块 + 是否还有空闲

        判断基准是 kv_len (下一个要写入的 K/V 位置), 不是 num_tokens。
        因为采样后 num_tokens = kv_len + 1, 而 K/V 写入位置是 kv_len。"""
        need = seq.kv_len % self.block_size == 0
        return (not need) or len(self.free) > 0

    def may_append(self, seq: Sequence):
        """需要就追加, 并管理 ref_count + 清除旧 hash"""
        if seq.kv_len % self.block_size == 0:
            blk = self.free.pop()
            if self.block_hash[blk] != -1:
                old_h = self.block_hash[blk]
                if self.hash_to_block.get(old_h) == blk:
                    del self.hash_to_block[old_h]
                self.block_hash[blk] = -1
                self.block_tokens[blk] = None
            self.ref_count[blk] = 1
            seq.block_table.append(blk)

    # ================================================================
    # hash_blocks: prefill 完成后注册满 block 的 hash
    # ================================================================
    def hash_blocks(self, seq: Sequence):
        """prefill 后把本轮写入的 block 注册到 hash 表。

        使用 seq.kv_len (本轮后) - seq.num_scheduled_tokens (本轮前) 确定边界，
        对于 decode (num_scheduled_tokens=0 或仅有尾部不满 block) 则跳过。"""
        old_kv_len = seq.kv_len - seq.num_scheduled_tokens
        start_block = old_kv_len // self.block_size
        end_block = seq.kv_len // self.block_size
        # 对 prefill chunk: hash 所有本轮完整写入的 block
        # 对 decode: 通常不跨 block, start==end, 跳过

        if start_block >= end_block:
            return

        # 找链式起点: 前一个 block 的 hash (可能来自共享或之前注册的)
        h = self.block_hash[seq.block_table[start_block - 1]] if start_block > 0 else -1

        for i in range(start_block, end_block):
            blk = seq.block_table[i]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            self.block_hash[blk] = h
            self.block_tokens[blk] = token_ids
            self.hash_to_block[h] = blk

    @property
    def available(self):
        return len(self.free)


# ================================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, max_position, rope_theta):
        super().__init__()
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        pos = torch.arange(0, max_position, dtype=torch.float)
        freqs = torch.outer(pos, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
        self.register_buffer("cos_sin_cache", cache)   # [max_pos, head_dim] (half cos, half sin)

    def forward(self, q, k, positions):
        orig_dtype_q = q.dtype
        orig_dtype_k = k.dtype

        # cos/sin: [seq_len, head_dim//2] or [total_tokens, head_dim//2] each
        cos_sin = self.cos_sin_cache[positions]            # [N, head_dim]
        cos, sin = cos_sin.chunk(2, dim=-1)                # each [N, head_dim//2]

        # 根据输入维度调整广播形状
        if q.dim() == 4:
            # [batch, seq, heads, dim] → cos/sin: [1, seq, 1, head_dim//2]
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        else:
            # varlen: [tokens, heads, dim] → cos/sin: [tokens, 1, head_dim//2]
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        def rotate(x, x_orig_dtype):
            x_f32 = x.float()
            x1, x2 = x_f32.chunk(2, dim=-1)        # 前半/后半配对，和 HF 一致
            x_new0 = x1 * cos - x2 * sin
            x_new1 = x2 * cos + x1 * sin
            return torch.cat([x_new0, x_new1], dim=-1).to(x_orig_dtype)

        q = rotate(q, orig_dtype_q)
        k = rotate(k, orig_dtype_k)
        return q, k

@lru_cache(8)
def get_rope(head_dim: int, max_position: int, rope_theta: float):
    """共享 RotaryEmbedding：相同参数只创建一次，避免每层重复分配 cos_sin_cache (~21MB)"""
    return RotaryEmbedding(head_dim, max_position, rope_theta)


class Qwen3Attention(nn.Module):
    """GQA + QK-Norm + RoPE —— Qwen3 的 attention 层"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.num_heads = config.num_attention_heads       # 16
        self.num_kv_heads = config.num_key_value_heads     # 8
        self.head_dim = config.head_dim                    # 128
        self.hidden_size = config.hidden_size               # 1024
        self.q_size = self.num_heads * self.head_dim        # 2048
        self.kv_size = self.num_kv_heads * self.head_dim    # 1024
        self.scale = self.head_dim ** -0.5

        # Fused QKV: 一次矩阵乘代替三次, 省 2 次 kernel launch + 2 次显存读写
        self.qkv_proj = nn.Linear(self.hidden_size, self.q_size + 2 * self.kv_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=False)

        # QK-Norm：对 Q 和 K 做逐头归一化（Qwen3 没有 bias 所以有这个）
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # KV Cache（单序列，按需分配）
        self.k_cache = None
        self.v_cache = None

        # RoPE：旋转位置编码
        self.rotary_emb = get_rope(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

    def allocate_cache(self, batch_size, max_len):
        """分配 KV cache 显存：[batch, max_len, kv_heads, head_dim]"""
        self.k_cache = torch.empty(batch_size, max_len, self.num_kv_heads, self.head_dim)
        self.v_cache = torch.empty(batch_size, max_len, self.num_kv_heads, self.head_dim)

    def forward_kv_cache(self, x, positions, is_prefill, cache_pos):
        """
        x:          [batch, seq_len, hidden_size]  prefill=全prompt, decode=1个新token
        positions:  [seq_len]  绝对位置（decode 时 = [当前序号]）
        cache_pos:  [seq_len]  K/V 存入 cache 的位置
        """
        batch, seq_len, _ = x.shape

        # 1. QKV 投影 (fused: 一次 matmul → split)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 2. view
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # 3. QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. RoPE（用绝对位置）
        q, k = self.rotary_emb(q, k, positions)

        # 5. K/V 存入 cache
        self.k_cache[:, cache_pos] = k
        self.v_cache[:, cache_pos] = v

        if is_prefill:
            # prefill: 用刚刚算出的 K/V 做 attention（已存 cache，但 cache 里也只有这些）
            k_used = k
            v_used = v
        else:
            # decode: 从 cache 读所有历史的 K/V（包括刚存进去的）
            k_used = self.k_cache[:, :cache_pos[-1] + 1]  # [batch, all_positions, kv_heads, dim]
            v_used = self.v_cache[:, :cache_pos[-1] + 1]

        # 6. transpose → [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k_used = k_used.transpose(1, 2)
        v_used = v_used.transpose(1, 2)

        # 7. GQA: repeat KV
        k_used, v_used = self._repeat_kv(k_used, v_used)

        # 8. Scaled dot-product attention
        attn_scores = (q @ k_used.transpose(-2, -1)) * self.scale

        # prefill 需要 causal mask，decode 不需要（Q 只有 1 token，天然单向）
        if is_prefill:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            neg_inf = torch.tensor(float('-inf'), dtype=attn_scores.dtype, device=attn_scores.device)
            attn_scores = attn_scores.masked_fill(causal_mask, neg_inf)

        attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(attn_scores.dtype)
        attn_output = attn_weights @ v_used

        # 9. merge heads → output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_output)

    def _repeat_kv(self, k, v):
        """GQA：把 KV 从 num_kv_heads 复制到 num_heads"""
        n_repeat = self.num_heads // self.num_kv_heads   # 16 // 8 = 2
        k = k.repeat_interleave(n_repeat, dim=1)
        v = v.repeat_interleave(n_repeat, dim=1)
        return k, v

    def forward(self, x):
        ctx = get_varlen_context()

        if ctx is not None:
            # ===== Varlen 路径: [total_tokens, hidden] =====
            tokens, _ = x.shape

            qkv = self.qkv_proj(x)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            q = q.view(tokens, self.num_heads, self.head_dim)
            k = k.view(tokens, self.num_kv_heads, self.head_dim)
            v = v.view(tokens, self.num_kv_heads, self.head_dim)

            q = self.q_norm(q)
            k = self.k_norm(k)

            q, k = self.rotary_emb(q, k, ctx.positions)

            # flash_attn 原生处理 GQA，q/k/v 头数可以不同
            o = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=ctx.max_seqlen_q,
                cu_seqlens_q=ctx.cu_seqlens_q,
                max_seqlen_k=ctx.max_seqlen_k,
                cu_seqlens_k=ctx.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
            )

            o = o.view(tokens, -1)  # [tokens, q_size]
            return self.o_proj(o)

        # ===== 原路径: [batch, seq, hidden] =====
        batch, seq_len, _ = x.shape

        # 1. Fused QKV 投影
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 2. view → [batch, seq_len, num_heads, head_dim]
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # 3. QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. RoPE
        positions = torch.arange(0, seq_len, device=x.device)
        q, k = self.rotary_emb(q, k, positions)

        # 5. transpose → [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 6. GQA: repeat KV
        k, v = self._repeat_kv(k, v)

        # 7. Scaled dot-product attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        neg_inf = torch.tensor(float('-inf'), dtype=attn_scores.dtype, device=attn_scores.device)
        attn_scores = attn_scores.masked_fill(causal_mask, neg_inf)
        attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(attn_scores.dtype)
        attn_output = attn_weights @ v

        # 8. merge heads → output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_output)

    # ===== Continuous Batching: Prefill 和 Decode 路径 =====

    def forward_prefill(self, x, positions, cu_seqlens_q, cu_seqlens_k,
                        max_seqlen_q, max_seqlen_k, block_mapping,
                        block_tables=None):
        """
        Prefill: 一次性处理多条序列的 prompt，K/V 写入全局 cache。

        x:             [total_tokens, hidden]    只包含需要计算的 token (不含缓存前缀)
        positions:     [total_tokens]            各 token 在各自序列内的绝对位置
        cu_seqlens_q:  [num_seqs + 1]            新 token 的累积长度
        cu_seqlens_k:  [num_seqs + 1]            全部 token 的累积长度 (含缓存前缀)
        block_mapping: [total_tokens]            新 token 写入 cache 的物理地址
        block_tables:  [batch, max_blocks]|None  有前缀缓存时使用, flash_attn 凭此读缓存
        """
        tokens, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(tokens, self.num_heads, self.head_dim)
        k = k.view(tokens, self.num_kv_heads, self.head_dim)
        v = v.view(tokens, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rotary_emb(q, k, positions)

        # 把 K/V 存入全局 cache (Triton kernel, 零中间 tensor)
        if self.k_cache.numel():
            store_kvcache(k, v, self.k_cache, self.v_cache, block_mapping)
            # q 统一 cast 到 cache dtype (WSL2 上 qkv_proj 可能输出 float32)
            q = q.to(self.k_cache.dtype)

        # 始终从 cache 读 (含刚写入的 K/V), flash_attn 通过 block_tables 寻址
        o = flash_attn_varlen_func(
            q, self.k_cache, self.v_cache,
            max_seqlen_q=max_seqlen_q, cu_seqlens_q=cu_seqlens_q,
            max_seqlen_k=max_seqlen_k, cu_seqlens_k=cu_seqlens_k,
            softmax_scale=self.scale, causal=True,
            block_table=block_tables,
        )

        o = o.view(tokens, -1)
        return self.o_proj(o)

    def forward_decode(self, x, positions, context_lens, block_tables, slot_mapping=None):
        """
        Decode: 每个序列只算 1 个新 token，从全局 cache 读历史 K/V。

        x:             [batch, hidden]   batch 条序列，各 1 个 token
        positions:     [batch]           每个新 token 的绝对位置
        context_lens:  [batch]           每个序列已缓存的 K/V 数量（写新 token 前）
        block_tables:  [batch, max_blocks] 页表，-1 填充
        slot_mapping:  [batch]           每个新 token 写入 cache 的物理地址 (可选, CUDA Graph 用)
        """
        batch = x.shape[0]

        qkv = self.qkv_proj(x)
        q, k_new, v_new = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(batch, self.num_heads, self.head_dim)
        k_new = k_new.view(batch, self.num_kv_heads, self.head_dim)
        v_new = v_new.view(batch, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k_new = self.k_norm(k_new)

        q, k_new = self.rotary_emb(q, k_new, positions)

        # 把新 K/V 写入全局 cache (Triton kernel, 零中间 tensor)
        if self.k_cache.numel():
            store_kvcache(k_new, v_new, self.k_cache, self.v_cache, slot_mapping)
            q = q.to(self.k_cache.dtype)

        # flash_attn 从 cache 读历史 K/V（靠 block_table 寻址）
        o = flash_attn_with_kvcache(
            q.unsqueeze(1),                  # [batch, 1, num_heads, dim]
            self.k_cache,                    # [num_blocks, block_size, kv_heads, dim]
            self.v_cache,
            cache_seqlens=context_lens + 1,  # 刚追加了 1 个
            block_table=block_tables,        # ★ 页表
            softmax_scale=self.scale,
            causal=True,
        )

        o = o.squeeze(1).view(batch, -1)     # [batch, q_size]
        return self.o_proj(o)


class Qwen3MLP(nn.Module):
    """SwiGLU —— Qwen3 的 FFN 层"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        # Fused gate+up: 一次 matmul 出 gate 和 up, 省掉一次读 x 和一次 kernel launch
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    @torch.compile
    def _silu_and_mul(self, gate_up):
        """将 chunk + silu + mul 融合为一个编译 kernel, 减少 kernel launch 开销"""
        gate, up = gate_up.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        hidden = self._silu_and_mul(gate_up)
        return self.down_proj(hidden)


class Qwen3DecoderLayer(nn.Module):
    """一层 Transformer decoder：Pre-Norm → Attention → Pre-Norm → MLP"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(self, x):
        # 和 gpt.py 的 Layer 完全一样的结构：norm → sublayer → add
        x = self.self_attn(self.input_layernorm(x)) + x
        x = self.mlp(self.post_attention_layernorm(x)) + x
        return x

    def forward_kvcache(self, x, positions, is_prefill, cache_pos):
        # RMSNorm 和 MLP 不变，只有 attention 改用 cache 版本
        x = self.self_attn.forward_kv_cache(
            self.input_layernorm(x), positions, is_prefill, cache_pos
        ) + x
        x = self.mlp(self.post_attention_layernorm(x)) + x
        return x

    def forward_prefill(self, x, positions, cu_seqlens_q, cu_seqlens_k,
                        max_seqlen_q, max_seqlen_k, block_mapping,
                        block_tables=None, residual=None):
        if residual is None:
            hidden_states, residual = self.input_layernorm(x), x
        else:
            hidden_states, residual = self.input_layernorm(x, residual)
        hidden_states = self.self_attn.forward_prefill(
            hidden_states, positions,
            cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, block_mapping,
            block_tables,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def forward_decode(self, x, positions, context_lens, block_tables,
                       slot_mapping=None, residual=None):
        if residual is None:
            hidden_states, residual = self.input_layernorm(x), x
        else:
            hidden_states, residual = self.input_layernorm(x, residual)
        hidden_states = self.self_attn.forward_decode(
            hidden_states, positions, context_lens, block_tables, slot_mapping,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """token embedding → 28 层 decoder → 最终 norm"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x):
        # x: [batch, seq_len] token IDs
        h = self.embed_tokens(x)               # [batch, seq, hidden_size]
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return h

    def forward_kvcache(self, x, positions, is_prefill, cache_pos):
        h = self.embed_tokens(x)
        for layer in self.layers:
            h = layer.forward_kvcache(h, positions, is_prefill, cache_pos)
        h = self.norm(h)
        return h

    def forward_prefill(self, x, positions, cu_seqlens_q, cu_seqlens_k,
                        max_seqlen_q, max_seqlen_k, block_mapping,
                        block_tables=None):
        h = self.embed_tokens(x)
        residual = None
        for layer in self.layers:
            h, residual = layer.forward_prefill(
                h, positions, cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k, block_mapping,
                block_tables, residual,
            )
        h, _ = self.norm(h, residual)
        return h

    def forward_decode(self, x, positions, context_lens, block_tables, slot_mapping=None):
        h = self.embed_tokens(x)
        residual = None
        for layer in self.layers:
            h, residual = layer.forward_decode(
                h, positions, context_lens, block_tables, slot_mapping, residual,
            )
        h, _ = self.norm(h, residual)
        return h


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 模型总入口：Model + lm_head"""

    # 告诉 load_weights 如何把 checkpoint 的独立参数拼到 fused 参数里
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", "gate"),
        "up_proj": ("gate_up_proj", "up"),
    }

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # tie_word_embeddings：lm_head 和 embedding 共享权重
        # 只替换 .data (和 nano-vllm 一样), 避免孤儿 tensor 浪费显存
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, x):
        h = self.model(x)                      # [batch, seq, hidden_size]
        return self.lm_head(h)                 # [batch, seq, vocab_size]

    def allocate_kv_cache(self, batch_size, max_len):
        """给所有 attention 层分配 KV cache"""
        for layer in self.model.layers:
            layer.self_attn.allocate_cache(batch_size, max_len)

    @torch.no_grad()
    def generate_kvcache(self, prompt, max_new_tokens=256, temperature=0.6, top_k=20, eos_token_ids=None):
        """KV-Cache 版生成：prefill 一次 + decode N 次"""
        batch_size, prompt_len = prompt.shape
        max_len = prompt_len + max_new_tokens
        self.allocate_kv_cache(batch_size, max_len)

        # === Prefill: 一次性 forward 整个 prompt ===
        positions = torch.arange(0, prompt_len, device=prompt.device)
        cache_pos = torch.arange(0, prompt_len, device=prompt.device)
        h = self.model.forward_kvcache(prompt, positions, is_prefill=True, cache_pos=cache_pos)
        logits = self.lm_head(h)[:, -1, :] / temperature

        # top-k
        if top_k is not None:
            topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
            neg_inf = torch.tensor(float('-inf'), dtype=logits.dtype, device=logits.device)
            logits = torch.full_like(logits, neg_inf).scatter(-1, topk_idx, topk_vals)

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
        generated = torch.cat([prompt, next_token], dim=1)

        # === Decode loop ===
        for step in range(max_new_tokens - 1):
            current_pos = prompt_len + step                                        # 当前绝对位置
            positions = torch.tensor([current_pos], device=prompt.device)
            cache_pos = torch.tensor([current_pos], device=prompt.device)

            one_token = next_token  # already [1, 1] = [batch, 1 token]
            h = self.model.forward_kvcache(one_token, positions, is_prefill=False, cache_pos=cache_pos)
            logits = self.lm_head(h)[:, -1, :] / temperature

            if top_k is not None:
                topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
                neg_inf = torch.tensor(float('-inf'), dtype=logits.dtype, device=logits.device)
                logits = torch.full_like(logits, neg_inf).scatter(-1, topk_idx, topk_vals)

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if eos_token_ids is not None and next_token.item() in eos_token_ids:
                break

            generated = torch.cat([generated, next_token], dim=1)

        return generated

    @torch.no_grad()
    def generate_naive(self, prompt, max_new_tokens=256, temperature=0.6, top_k=20, eos_token_ids=None):
        """自回归生成，遇到 EOS 提前停止"""
        generated = prompt.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(generated)                       # [1, seq, vocab]
            last_logits = logits[:, -1, :] / temperature            # [1, vocab]

            # top-k 过滤：只保留概率最高的 k 个，其余置 -inf
            if top_k is not None:
                topk_vals, topk_idx = torch.topk(last_logits, top_k, dim=-1)
                neg_inf = torch.tensor(float('-inf'), dtype=last_logits.dtype, device=last_logits.device)
                filtered = torch.full_like(last_logits, neg_inf)
                last_logits = filtered.scatter(-1, topk_idx, topk_vals)

            probs = torch.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)   # [1, 1]

            if eos_token_ids is not None and next_token.item() in eos_token_ids:
                break

            generated = torch.cat([generated, next_token], dim=1)
        return generated

    @torch.no_grad()
    def generate_varlen(self, prompt_ids_list, max_new_tokens=256, temperature=0.6,
                        top_k=20, eos_token_ids=None):
        """Varlen 批量生成：多条 prompt 同时推理，不填充，不浪费计算。

        prompt_ids_list: list of [1, prompt_len_i] tensors，每条 prompt 的 tokenized 结果
        """
        batch_size = len(prompt_ids_list)
        # 每条序列当前的完整 token 列表
        seq_tokens = [p[0].tolist() for p in prompt_ids_list]
        unfinished = [True] * batch_size
        all_generated = [[] for _ in range(batch_size)]

        for step in range(max_new_tokens):
            # ---- 1. 构建 varlen 输入 ----
            all_ids = []
            all_positions = []
            cu_seqlens = [0]
            max_seqlen = 0

            for tokens in seq_tokens:
                L = len(tokens)
                all_ids.extend(tokens)
                all_positions.extend(range(L))
                cu_seqlens.append(cu_seqlens[-1] + L)
                max_seqlen = max(max_seqlen, L)

            input_ids = torch.tensor(all_ids, dtype=torch.long, device='cuda')
            cu_sl = torch.tensor(cu_seqlens, dtype=torch.int32, device='cuda')
            positions = torch.tensor(all_positions, dtype=torch.long, device='cuda')

            # ---- 2. 设置 varlen context，forward ----
            set_varlen_context(
                cu_seqlens_q=cu_sl,
                cu_seqlens_k=cu_sl,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                positions=positions,
            )
            logits = self.forward(input_ids)   # [total_tokens, vocab]
            clear_varlen_context()

            # ---- 3. 取每条序列最后一个 token 的 logit ----
            last_logits_list = []
            for i in range(batch_size):
                last_idx = cu_seqlens[i + 1] - 1
                last_logits_list.append(logits[last_idx])
            last_logits = torch.stack(last_logits_list)      # [batch, vocab]

            # ---- 4. temperature + top-k + sample ----
            last_logits = last_logits / temperature

            if top_k is not None:
                topk_vals, topk_idx = torch.topk(last_logits, top_k, dim=-1)
                neg_inf = torch.tensor(float('-inf'), dtype=last_logits.dtype,
                                       device=last_logits.device)
                filtered = torch.full_like(last_logits, neg_inf)
                last_logits = filtered.scatter(-1, topk_idx, topk_vals)

            probs = torch.softmax(last_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)   # [batch, 1]

            # ---- 5. 更新序列状态 ----
            for i in range(batch_size):
                if not unfinished[i]:
                    continue
                token_id = next_tokens[i].item()
                seq_tokens[i].append(token_id)
                all_generated[i].append(token_id)

                if eos_token_ids is not None and token_id in eos_token_ids:
                    unfinished[i] = False

            if not any(unfinished):
                break

        return seq_tokens, all_generated

    # ===================== Continuous Batching =====================

    BLOCK_SIZE = 256

    def _compute_num_blocks(self, gpu_memory_utilization=0.9):
        """Warmup forward → 测量峰值显存 → 计算 KV cache block 数。

        参照 nano-vllm 用 memory_stats().peak 计算。
        WSL2 下 memory_stats() 可能返回异常小的值 (< 10MB),
        此时自动切换到保守估算 (预留 30% 空闲显存给激活)。
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 直接用最大 chunk 长度做 warmup，和 nano-vllm 一样，不需要外推
        warmup_tokens = 2048
        dummy = torch.randint(0, self.config.vocab_size, (1, warmup_tokens), device='cuda')
        with torch.no_grad():
            _ = self.forward(dummy)
        del _, dummy
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # WSL2 上第二次释放更彻底

        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        peak_act = peak - current

        attn0 = self.model.layers[0].self_attn
        block_bytes = (2 * len(self.model.layers) * self.BLOCK_SIZE *
                       attn0.num_kv_heads * attn0.head_dim * 2)

        available = int(total * gpu_memory_utilization - used - peak_act)
        num_blocks = max(available // block_bytes, 1)

        print(f"  KV Cache: total={total/1e9:.2f}GB, used={used/1e6:.0f}MB, "
              f"peak_act={peak_act/1e6:.0f}MB (peak={peak/1e6:.0f}MB, "
              f"current={current/1e6:.0f}MB)")
        print(f"    budget={available/1e6:.0f}MB, block_bytes={block_bytes/1e6:.1f}MB, "
              f"blocks={num_blocks} ({num_blocks * self.BLOCK_SIZE} tokens)")
        return num_blocks

    def allocate_global_kv_cache(self):
        """分配全局 KV cache：[layers, num_blocks, block_size, kv_heads, dim]。

        首次调用时通过 warmup 动态计算 NUM_BLOCKS。
        如果已分配过相同形状则跳过，避免重复分配 OOM。"""
        attn0 = self.model.layers[0].self_attn
        num_layers = len(self.model.layers)
        n_kv = attn0.num_kv_heads
        d = attn0.head_dim

        # 动态计算 NUM_BLOCKS (首次调用)
        num_blocks = getattr(self, '_num_blocks', 0)
        if num_blocks == 0:
            num_blocks = self._compute_num_blocks()
            self._num_blocks = num_blocks

        # 已分配过相同形状则跳过
        expected_shape = (num_blocks, self.BLOCK_SIZE, n_kv, d)
        if hasattr(attn0, 'k_cache') and attn0.k_cache is not None and \
           attn0.k_cache.shape == expected_shape:
            return

        k = torch.empty(num_layers, num_blocks, self.BLOCK_SIZE, n_kv, d,
                        dtype=torch.bfloat16, device='cuda')
        v = torch.empty(num_layers, num_blocks, self.BLOCK_SIZE, n_kv, d,
                        dtype=torch.bfloat16, device='cuda')

        for i, layer in enumerate(self.model.layers):
            layer.self_attn.k_cache = k[i]
            layer.self_attn.v_cache = v[i]

    # ===================== CUDA Graph (decode 加速) =====================

    def _init_cuda_graphs(self):
        """预分配 CUDA Graph buffer tensors + 捕获各 batch size 的图。

        所有图共享一个 memory pool, 避免显存爆炸。
        录制 batch sizes: [1, 2, 4, 8, 16, 32, 48, 64]
        """
        max_bs = 64
        max_blocks = (self.config.max_position_embeddings + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
        hidden = self.config.hidden_size

        self._graph_bufs = {
            "input_ids":    torch.zeros(max_bs, dtype=torch.long,       device='cuda'),
            "positions":    torch.zeros(max_bs, dtype=torch.long,       device='cuda'),
            "slot_mapping": torch.zeros(max_bs, dtype=torch.int32,      device='cuda'),
            "context_lens": torch.zeros(max_bs, dtype=torch.int32,      device='cuda'),
            "block_tables": torch.zeros(max_bs, max_blocks, dtype=torch.int32, device='cuda'),
            "outputs":      torch.zeros(max_bs, hidden, dtype=torch.bfloat16, device='cuda'),
        }
        self._graphs = {}
        self._graph_pool = None
        self._graph_bs = [1, 2, 4, 8, 16, 32, 48, 64]
        self._has_cuda_graphs = False

    def _capture_decode_graphs(self):
        """为每个 batch size 录制 CUDA Graph (decode-only)。

        先 warmup 一次触发 lazy init, 再录制。所有图复用同一个 memory pool。
        """
        if torch.cuda.is_available():
            try:
                bufs = self._graph_bufs
                num_blocks = getattr(self, '_num_blocks', 0)
                max_bt = bufs["block_tables"].shape[1]

                for bs in reversed(self._graph_bs):
                    # warmup: 触达所有 lazy 初始化
                    self.model.forward_decode(
                        bufs["input_ids"][:bs], bufs["positions"][:bs],
                        bufs["context_lens"][:bs], bufs["block_tables"][:bs, :min(num_blocks, max_bt)],
                        bufs["slot_mapping"][:bs],
                    )
                    graph = torch.cuda.CUDAGraph()
                    bt_cols = min(num_blocks, max_bt) if num_blocks > 0 else max_bt
                    with torch.cuda.graph(graph, pool=self._graph_pool):
                        bufs["outputs"][:bs] = self.model.forward_decode(
                            bufs["input_ids"][:bs], bufs["positions"][:bs],
                            bufs["context_lens"][:bs], bufs["block_tables"][:bs, :bt_cols],
                            bufs["slot_mapping"][:bs],
                        )
                    if self._graph_pool is None:
                        self._graph_pool = graph.pool()
                    self._graphs[bs] = graph

                self._has_cuda_graphs = True
                print(f"  CUDA Graph: 捕获 {len(self._graphs)} 个图 "
                      f"(bs={self._graph_bs})")
            except Exception as e:
                print(f"  CUDA Graph 捕获失败, 回退到 eager 模式: {e}")
                self._has_cuda_graphs = False

    def _run_decode_graph(self, bs, input_ids, positions, context_lens,
                          block_tables, slot_mapping):
        """用 CUDA Graph 执行 decode: 拷贝输入 → replay → 返回输出"""
        graph_bs = next((x for x in self._graph_bs if x >= bs), None)
        if graph_bs is None:
            return None

        bufs = self._graph_bufs
        bufs["input_ids"][:bs] = input_ids
        bufs["positions"][:bs] = positions
        bufs["slot_mapping"][:bs] = slot_mapping
        bufs["context_lens"][:bs] = context_lens
        bufs["block_tables"][:bs, :block_tables.shape[1]] = block_tables
        self._graphs[graph_bs].replay()
        # 不 clone: lm_head 立刻消费输出, 下一次 replay 才会覆盖 pool 内存
        return bufs["outputs"][:bs]

    def _prepare_step(self, seqs: list[Sequence]):
        """统一准备混合 batch (prefill chunk + decode): 构建 varlen 输入和 block_mapping。

        对每条序列, 从 kv_len 开始取 num_scheduled_tokens 个 token 作为本轮计算量。
        decode: num_scheduled_tokens=1, 取 1 个新 token 的 input_id
        prefill chunk: num_scheduled_tokens=N, 取 N 个 prompt token

        返回: (input_ids, positions, cu_seqlens_q, cu_seqlens_k,
               max_seqlen_q, max_seqlen_k, block_mapping, block_tables)
        """
        input_ids, positions = [], []
        cu_seqlens_q = [0]      # 新 token 累积长度 (只含本轮要算的)
        cu_seqlens_k = [0]      # 全部 token 累积长度 (含缓存前缀 + 本轮新 token)
        max_seqlen_q = 0
        max_seqlen_k = 0
        block_mapping = []

        for seq in seqs:
            start = seq.kv_len                        # 下一个要算的位置
            end = start + seq.num_scheduled_tokens     # 本轮算几个
            new_tokens = end - start

            input_ids.extend(seq.token_ids[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + new_tokens)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)  # kv_len + new_tokens
            max_seqlen_q = max(max_seqlen_q, new_tokens)
            max_seqlen_k = max(max_seqlen_k, end)

            # 为新 token 构建 block_mapping (写入 cache) — 逐 block 而非逐 token
            start_blk = start // self.BLOCK_SIZE
            end_blk = (end + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
            for blk_idx in range(start_blk, end_blk):
                slot_start = seq.block_table[blk_idx] * self.BLOCK_SIZE
                if blk_idx == start_blk:
                    slot_start += start % self.BLOCK_SIZE
                if blk_idx == end_blk - 1:
                    slot_end = seq.block_table[blk_idx] * self.BLOCK_SIZE + end - blk_idx * self.BLOCK_SIZE
                else:
                    slot_end = seq.block_table[blk_idx] * self.BLOCK_SIZE + self.BLOCK_SIZE
                block_mapping.extend(range(slot_start, slot_end))

        # pin_memory + non_blocking: CPU→GPU 异步传输, 和 GPU kernel 重叠
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        cu_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        blk_map = torch.tensor(block_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # 始终构建 block_tables (混合 batch 必须, decode-only 也必须)
        block_tables = self._build_block_tables(seqs)

        return input_ids, positions, cu_q, cu_k, max_seqlen_q, max_seqlen_k, blk_map, block_tables

    def _build_block_tables(self, seqs: list[Sequence]):
        """构建 block_tables tensor: [batch, max_blocks], 短序列填 -1。一次拼接，一次 GPU 拷贝。"""
        max_blocks = max(len(s.block_table) for s in seqs)
        rows = [s.block_table + [-1] * (max_blocks - len(s.block_table)) for s in seqs]
        return torch.tensor(rows, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def _prepare_decode(self, seqs: list[Sequence]):
        """构建 decode 输入：block_tables + slot_mapping 用于 flash_attn + CUDA Graph"""
        input_ids = torch.tensor([s.last_token for s in seqs], dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor([s.num_tokens - 1 for s in seqs], dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor([s.kv_len for s in seqs], dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        # block_tables: [batch, max_blocks], 短序列填 -1。一次拼接，一次 GPU 拷贝
        bt = self._build_block_tables(seqs)

        # slot_mapping: 每个新 token 写入 cache 的物理地址
        slot_mapping = []
        for s in seqs:
            pos = s.kv_len
            blk = s.block_table[pos // self.BLOCK_SIZE]
            off = pos % self.BLOCK_SIZE
            slot_mapping.append(blk * self.BLOCK_SIZE + off)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

        return input_ids, positions, context_lens, bt, slot_mapping

    @torch.no_grad()
    def generate_continuous(self, prompt_ids_list, max_new_tokens=256,
                            temperature=0.6, top_k=20, eos_token_ids=None):
        """
        Continuous Batching + Chunked Prefill: 序列随时加入/退出,
        prefill 和 decode 可在同一个 batch 中混合执行。

        调度策略: 先遍历 running (decode 优先, 每条 1 token; mid-prefill 继续吃预算),
                 再用剩余 budget 拉 waiting (切 chunk)。
        """
        MAX_TOKENS_PER_STEP = 2048

        self.allocate_global_kv_cache()           # 触发 warmup, 设置 _num_blocks
        bm = BlockManager(self._num_blocks, self.BLOCK_SIZE)

        # 初始化 CUDA Graph (只做一次, 避免重复捕获污染 KV cache)
        if not getattr(self, '_has_cuda_graphs', False):
            self._init_cuda_graphs()
            self._capture_decode_graphs()

        # 用 prompt_ids_list 创建 Sequence
        waiting: list[Sequence] = []
        for pids in prompt_ids_list:
            token_ids = pids[0].tolist()
            waiting.append(Sequence(token_ids, {"temperature": temperature, "max_tokens": max_new_tokens}))

        running: list[Sequence] = []
        all_seqs = list(waiting)  # 保留引用，最终用来取结果

        while waiting or running:
            all_scheduled = []
            budget = MAX_TOKENS_PER_STEP

            # ============================================================
            # Step 1: 遍历 running — decode 优先，mid-prefill 继续吃预算
            # ============================================================
            i = 0
            while i < len(running) and budget > 0:
                seq = running[i]

                if seq.kv_len >= seq.num_prompt_tokens:
                    # ── DECODE 序列 ──
                    # 检查是否到达 max_tokens (采样后才追加，这里预判)
                    if seq.num_completion_tokens >= seq.max_tokens:
                        seq.status = "FINISHED"
                        bm.deallocate(seq.block_table)
                        seq.block_table.clear()
                        running.pop(i)
                        continue

                    if not bm.can_append(seq):
                        # LIFO preemption: 先踢 running 末尾 (最新加入的),
                        # 保护已生成很多步的老同志, 实在只剩自己才踢自己
                        while not bm.can_append(seq):
                            victim = running[-1]      # 找最新的
                            victim.status = "WAITING"
                            bm.deallocate(victim.block_table)
                            victim.block_table.clear()
                            victim.kv_len = 0
                            # preempt 后从 prompt 重新开始, 清除已生成的 token
                            victim.token_ids = victim.token_ids[:victim.num_prompt_tokens]
                            victim.response_log_probs.clear()
                            running.pop()             # 弹出末尾 (LIFO)
                            waiting.insert(0, victim)

                            if victim is seq:         # 踢到自己 — 没法了
                                break

                        if seq.status == "WAITING":   # seq 被踢了
                            continue

                    bm.may_append(seq)                   # 正常情况 + LIFO 踢别人后都需要
                    seq.num_scheduled_tokens = 1
                else:
                    # ── MID-PREFILL 序列 ──
                    # blocks 在 allocate() 时已经全部分配好了，不需要 can_append
                    remaining = seq.num_prompt_tokens - seq.kv_len
                    seq.num_scheduled_tokens = min(remaining, budget)

                all_scheduled.append(seq)
                budget -= seq.num_scheduled_tokens
                i += 1

            # ============================================================
            # Step 2: 用剩余 budget 拉 waiting → 分配 blocks → 切 chunk
            # ============================================================
            while waiting and budget > 0:
                seq = waiting[0]

                if not seq.block_table:
                    # 首次分配 blocks (prefix cache 命中会设 kv_len)
                    if bm.available < seq.num_blocks:
                        break
                    bm.allocate(seq)

                waiting.pop(0)
                seq.status = "RUNNING"
                running.append(seq)

                remaining = seq.num_prompt_tokens - seq.kv_len
                if remaining <= budget:
                    seq.num_scheduled_tokens = remaining   # 一次 prefill 完
                elif len(all_scheduled) == 0:
                    seq.num_scheduled_tokens = budget      # 第一条, 切 chunk
                else:
                    # 已有其他 seq，留给下一步
                    waiting.insert(0, seq)
                    running.pop()
                    break

                all_scheduled.append(seq)
                budget -= seq.num_scheduled_tokens

            if not all_scheduled:
                continue

            # ============================================================
            # Step 3: Forward — 选路径
            # ============================================================
            all_decode = all(s.num_scheduled_tokens == 1 for s in all_scheduled)

            if all_decode:
                # decode-only: 优先 CUDA Graph, 回退到 eager
                input_ids, pos, ctx_lens, block_tables, slot_mapping = self._prepare_decode(all_scheduled)
                bs = len(all_scheduled)
                h = None
                if self._has_cuda_graphs:
                    h = self._run_decode_graph(
                        bs, input_ids, pos, ctx_lens,
                        block_tables, slot_mapping,
                    )
                if h is None:
                    h = self.model.forward_decode(
                        input_ids, pos, ctx_lens, block_tables, slot_mapping,
                    )
            else:
                # 混合 batch 或 prefill-only: 统一 varlen 路径
                input_ids, pos, cu_q, cu_k, m_q, m_k, blk_map, block_tables = self._prepare_step(all_scheduled)
                h = self.model.forward_prefill(input_ids, pos, cu_q, cu_k, m_q, m_k, blk_map, block_tables)

            logits = self.lm_head(h)

            # ============================================================
            # Step 4: 更新 kv_len + hash blocks
            # ============================================================
            for seq in all_scheduled:
                seq.kv_len += seq.num_scheduled_tokens
                bm.hash_blocks(seq)

            # ============================================================
            # Step 5: 条件采样 — 仅 prefill 完成 或 decode
            # ============================================================
            sample_indices = []
            offset = 0
            for seq in all_scheduled:
                if seq.num_scheduled_tokens == 1 or seq.kv_len >= seq.num_prompt_tokens:
                    sample_indices.append(offset + seq.num_scheduled_tokens - 1)
                offset += seq.num_scheduled_tokens

            if sample_indices:
                last_logits = logits[sample_indices].float()
            else:
                # 所有 seq 都是 mid-prefill, 不采样
                for seq in all_scheduled:
                    seq.num_scheduled_tokens = 0
                continue

            last_logits = last_logits / temperature
            if top_k is not None:
                topk_vals, topk_idx = torch.topk(last_logits, top_k, dim=-1)
                neg_inf = torch.tensor(float('-inf'), dtype=last_logits.dtype, device=last_logits.device)
                filtered = torch.full_like(last_logits, neg_inf)
                last_logits = filtered.scatter(-1, topk_idx, topk_vals)

            probs = torch.softmax(last_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)   # [N_sample, 1]

            # ---- 更新采样后的序列状态 ----
            sample_idx = 0
            for seq in all_scheduled:
                if seq.num_scheduled_tokens == 1 or seq.kv_len >= seq.num_prompt_tokens:
                    token_id = next_tokens[sample_idx].item()
                    seq.append_token(token_id)
                    sample_idx += 1

                    if eos_token_ids is not None and token_id in eos_token_ids:
                        seq.status = "FINISHED"
                        bm.deallocate(seq.block_table)
                        seq.block_table.clear()
                        running.remove(seq)
                    elif seq.num_completion_tokens >= seq.max_tokens:
                        seq.status = "FINISHED"
                        bm.deallocate(seq.block_table)
                        seq.block_table.clear()
                        running.remove(seq)

                seq.num_scheduled_tokens = 0   # reset

        # 返回结果
        results = [(s.token_ids, s.token_ids[s.num_prompt_tokens:]) for s in all_seqs]
        seq_tokens, all_generated = zip(*results)
        return list(seq_tokens), list(all_generated)

    @torch.no_grad()
    def generate_continuous_with_logprobs(self, prompt_ids_list, max_new_tokens=256,
                                          temperature=0.6, top_k=20, eos_token_ids=None):
        """
        Continuous Batching + Chunked Prefill，同时记录每个 response token 的 logP。

        与 generate_continuous 的调度逻辑完全一致，区别在于:
          1. 采样后额外计算 log_softmax，记录选中 token 的 logP
          2. 返回值多出 all_log_probs 和 prompt_lens

        返回:
          seq_tokens:     list[list[int]]   每条序列的完整 token IDs
          all_generated:  list[list[int]]   每条序列新生成的 token IDs
          all_log_probs:  list[list[float]] 每条序列每个 response token 的 logP
          prompt_lens:    list[int]         每条序列的 prompt 长度
        """
        MAX_TOKENS_PER_STEP = 2048

        self.allocate_global_kv_cache()
        bm = BlockManager(self._num_blocks, self.BLOCK_SIZE)

        if not getattr(self, '_has_cuda_graphs', False):
            self._init_cuda_graphs()
            self._capture_decode_graphs()

        waiting: list[Sequence] = []
        for pids in prompt_ids_list:
            token_ids = pids[0].tolist()
            waiting.append(Sequence(token_ids, {"temperature": temperature, "max_tokens": max_new_tokens}))

        running: list[Sequence] = []
        all_seqs = list(waiting)

        while waiting or running:
            all_scheduled = []
            budget = MAX_TOKENS_PER_STEP

            # ============================================================
            # Step 1: 遍历 running — decode 优先，mid-prefill 继续吃预算
            # ============================================================
            i = 0
            while i < len(running) and budget > 0:
                seq = running[i]

                if seq.kv_len >= seq.num_prompt_tokens:
                    # ── DECODE ──
                    if seq.num_completion_tokens >= seq.max_tokens:
                        seq.status = "FINISHED"
                        bm.deallocate(seq.block_table)
                        seq.block_table.clear()
                        running.pop(i)
                        continue

                    if not bm.can_append(seq):
                        while not bm.can_append(seq):
                            victim = running[-1]
                            victim.status = "WAITING"
                            bm.deallocate(victim.block_table)
                            victim.block_table.clear()
                            victim.kv_len = 0
                            victim.token_ids = victim.token_ids[:victim.num_prompt_tokens]
                            victim.response_log_probs.clear()
                            running.pop()
                            waiting.insert(0, victim)
                            if victim is seq:
                                break
                        if seq.status == "WAITING":
                            continue

                    bm.may_append(seq)
                    seq.num_scheduled_tokens = 1
                else:
                    # ── MID-PREFILL ──
                    remaining = seq.num_prompt_tokens - seq.kv_len
                    seq.num_scheduled_tokens = min(remaining, budget)

                all_scheduled.append(seq)
                budget -= seq.num_scheduled_tokens
                i += 1

            # ============================================================
            # Step 2: 用剩余 budget 拉 waiting
            # ============================================================
            while waiting and budget > 0:
                seq = waiting[0]

                if not seq.block_table:
                    if bm.available < seq.num_blocks:
                        break
                    bm.allocate(seq)

                waiting.pop(0)
                seq.status = "RUNNING"
                running.append(seq)

                remaining = seq.num_prompt_tokens - seq.kv_len
                if remaining <= budget:
                    seq.num_scheduled_tokens = remaining
                elif len(all_scheduled) == 0:
                    seq.num_scheduled_tokens = budget
                else:
                    waiting.insert(0, seq)
                    running.pop()
                    break

                all_scheduled.append(seq)
                budget -= seq.num_scheduled_tokens

            if not all_scheduled:
                continue

            # ============================================================
            # Step 3: Forward
            # ============================================================
            all_decode = all(s.num_scheduled_tokens == 1 for s in all_scheduled)

            if all_decode:
                input_ids, pos, ctx_lens, block_tables, slot_mapping = self._prepare_decode(all_scheduled)
                bs = len(all_scheduled)
                h = None
                if self._has_cuda_graphs:
                    h = self._run_decode_graph(
                        bs, input_ids, pos, ctx_lens,
                        block_tables, slot_mapping,
                    )
                if h is None:
                    h = self.model.forward_decode(
                        input_ids, pos, ctx_lens, block_tables, slot_mapping,
                    )
            else:
                input_ids, pos, cu_q, cu_k, m_q, m_k, blk_map, block_tables = self._prepare_step(all_scheduled)
                h = self.model.forward_prefill(input_ids, pos, cu_q, cu_k, m_q, m_k, blk_map, block_tables)

            logits = self.lm_head(h)

            # ============================================================
            # Step 4: 更新 kv_len + hash blocks
            # ============================================================
            for seq in all_scheduled:
                seq.kv_len += seq.num_scheduled_tokens
                bm.hash_blocks(seq)

            # ============================================================
            # Step 5: 条件采样 + 记录 logP
            # ============================================================
            sample_indices = []
            offset = 0
            for seq in all_scheduled:
                if seq.num_scheduled_tokens == 1 or seq.kv_len >= seq.num_prompt_tokens:
                    sample_indices.append(offset + seq.num_scheduled_tokens - 1)
                offset += seq.num_scheduled_tokens

            if sample_indices:
                last_logits = logits[sample_indices].float()
            else:
                for seq in all_scheduled:
                    seq.num_scheduled_tokens = 0
                continue

            last_logits = last_logits / temperature
            if top_k is not None:
                topk_vals, topk_idx = torch.topk(last_logits, top_k, dim=-1)
                neg_inf = torch.tensor(float('-inf'), dtype=last_logits.dtype, device=last_logits.device)
                filtered = torch.full_like(last_logits, neg_inf)
                last_logits = filtered.scatter(-1, topk_idx, topk_vals)

            probs = torch.softmax(last_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)   # [N_sample, 1]

            # ★ GRPO 关键: 记录采样 token 的 logP
            log_probs_all = torch.log_softmax(last_logits, dim=-1)         # [N_sample, vocab]
            sampled_log_probs = log_probs_all.gather(-1, next_tokens)      # [N_sample, 1]

            # ---- 更新序列状态 + 存储 logP ----
            sample_idx = 0
            for seq in all_scheduled:
                if seq.num_scheduled_tokens == 1 or seq.kv_len >= seq.num_prompt_tokens:
                    token_id = next_tokens[sample_idx].item()
                    seq.append_token(token_id)
                    # 记录选中 token 的 log probability
                    seq.response_log_probs.append(sampled_log_probs[sample_idx].item())
                    sample_idx += 1

                    if eos_token_ids is not None and token_id in eos_token_ids:
                        seq.status = "FINISHED"
                        bm.deallocate(seq.block_table)
                        seq.block_table.clear()
                        running.remove(seq)
                    elif seq.num_completion_tokens >= seq.max_tokens:
                        seq.status = "FINISHED"
                        bm.deallocate(seq.block_table)
                        seq.block_table.clear()
                        running.remove(seq)

                seq.num_scheduled_tokens = 0

        # 返回 4 元组: token IDs, 新生成部分, logP, prompt 长度
        results = [(s.token_ids, s.token_ids[s.num_prompt_tokens:],
                    s.response_log_probs, s.num_prompt_tokens) for s in all_seqs]
        seq_tokens, all_generated, all_log_probs, prompt_lens = zip(*results)
        return list(seq_tokens), list(all_generated), list(all_log_probs), list(prompt_lens)


def load_weights(model, checkpoint_path):
    """全部参数名和 checkpoint 对齐，直接一一映射，不需要拼接。"""
    import struct

    with open(checkpoint_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode("utf-8"))
        device = next(model.parameters()).device

        packed_mapping = getattr(model, 'packed_modules_mapping', {})

        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name == "lm_head.weight":
                continue   # tie_word_embeddings

            tensor = _read_tensor(f, info, header_size, device).to(device)

            # 检查是否需要拼到 fused 参数 (packed_modules_mapping)
            for packed_key, (fused_suffix, shard_id) in packed_mapping.items():
                if packed_key in name:
                    param_name = name.replace(packed_key, fused_suffix)
                    param = model.get_parameter(param_name)
                    # 计算切片: q/gate→前部, k→中部, v/up→后部
                    if shard_id in ("q", "gate"):
                        offset = 0
                    elif shard_id == "k":
                        offset = param.shape[0] - 2 * tensor.shape[0]  # q_size
                    else:  # "v" or "up"
                        offset = param.shape[0] - tensor.shape[0]
                    param.data[offset:offset + tensor.shape[0]] = tensor
                    break
            else:
                model.get_parameter(name).data.copy_(tensor)

    print("权重加载完成!")


def _read_tensor(f, info, header_size, device):
    """从文件读一个张量：seek → read → frombuffer → reshape（CPU），调用者负责 .to(device)"""
    start, end = info["data_offsets"]
    f.seek(8 + header_size + start)
    raw = f.read(end - start)

    dtype_map = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    dtype = dtype_map[info["dtype"]]

    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(info["shape"])
        


def bench(model, config):
    """Benchmark: 随机 prompt，统计生成吞吐量"""
    import time
    from random import randint, seed

    seed(0)
    # 从 --num-seqs=N 读条数，默认 4
    num_seqs = 4
    for arg in sys.argv:
        if arg.startswith("--num-seqs="):
            num_seqs = int(arg.split("=")[1])
    max_input_len = 16
    max_output_len = 32

    print(f"\nBenchmark: {num_seqs} 条序列, input_len≤{max_input_len}, output_len≤{max_output_len}")
    print("预热中...")

    # 预热：跑一条短序列，触发 CUDA kernel 编译
    warmup = torch.randint(0, config.vocab_size, (1, 4), device='cuda')
    model.generate_naive(warmup, max_new_tokens=4, temperature=0.6, top_k=20)

    # 生成随机 prompt
    prompt_ids_list = [
        torch.randint(0, config.vocab_size, (1, randint(8, max_input_len)), device='cuda')
        for _ in range(num_seqs)
    ]
    output_lens = [randint(32, max_output_len) for _ in range(num_seqs)]

    total_tokens = 0

    torch.cuda.synchronize()
    t = time.time()

    for prompt_ids, out_len in zip(prompt_ids_list, output_lens):
        output = model.generate_naive(
            prompt_ids,
            max_new_tokens=out_len,
            temperature=0.6,
            top_k=20,
            eos_token_ids=None,   # benchmark 不提前停
        )
        total_tokens += output.shape[1] - prompt_ids.shape[1]  # 新生成的 token 数

    torch.cuda.synchronize()
    elapsed = time.time() - t

    throughput = total_tokens / elapsed
    print(f"生成: {total_tokens} tokens / {elapsed:.2f}s = {throughput:.2f} tok/s")
    print(f"VRAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


# 模块级配置：import 时生效，确保模型/cache 都是 bf16 + CUDA
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device('cuda')

if __name__ == '__main__':
    # ===== 端到端推理测试 =====
    import sys
    from transformers import AutoTokenizer

    model_path = "/mnt/c/Users/16874/Downloads/Qwen3-0.6B"

    # 1. 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2. 加载模型
    print("加载模型...")

    config = Qwen3Config.from_json(f"{model_path}/config.json")
    model = Qwen3ForCausalLM(config)
    load_weights(model, f"{model_path}/model.safetensors")

    vram = torch.cuda.memory_allocated() / 1e9
    params = sum(p.numel() for p in model.parameters())
    print(f"参数: {params:,}  |  VRAM: {vram:.2f} GB")

    # --bench 模式：性能测试
    if "--bench" in sys.argv:
        bench(model, config)
        sys.exit(0)

    # 3. 选择生成模式
    use_topk = "--no-topk" not in sys.argv
    top_k = 20 if use_topk else None
    if not use_topk:
        print("(top_k=None, 全词表采样)")

    use_kvcache = "--kvcache" in sys.argv
    use_varlen = "--varlen" in sys.argv
    use_continuous = "--continuous" in sys.argv

    if use_continuous:
        # ===== Continuous Batching 模式 =====
        prompts_file = None
        for arg in sys.argv:
            if arg.startswith("--prompts-file="):
                prompts_file = arg.split("=", 1)[1]

        if prompts_file:
            with open(prompts_file, "r", encoding="utf-8") as f:
                prompt_texts = [line.strip() for line in f if line.strip()]
        else:
            prompt_texts = [
                "你好，请介绍一下你自己",
                "今天天气怎么样？",
                "什么是人工智能？",
                "推荐三本科幻小说",
            ]

        print(f"\nContinuous Batching 模式: {len(prompt_texts)} 条 prompt, {model._num_blocks} blocks")

        prompt_ids_list = []
        for i, text in enumerate(prompt_texts):
            messages = [{"role": "user", "content": text}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            ids = tokenizer.encode(formatted, return_tensors="pt").to('cuda')
            prompt_ids_list.append(ids)
            print(f"  [{i}] {text[:30]}... ({ids.shape[1]} tokens)")

        print(f"\n生成中 (Continuous Batching)...")
        torch.manual_seed(42)
        torch.cuda.synchronize()
        t0 = __import__('time').time()
        seq_tokens, all_generated = model.generate_continuous(
            prompt_ids_list,
            max_new_tokens=256,
            temperature=0.6,
            top_k=top_k,
            eos_token_ids=[config.eos_token_id, config.bos_token_id],
        )
        torch.cuda.synchronize()
        elapsed = __import__('time').time() - t0

        total_new = sum(len(g) for g in all_generated)
        print(f"生成完成: {total_new} new tokens / {elapsed:.1f}s = {total_new/elapsed:.1f} tok/s")

        for i, (seq, generated) in enumerate(zip(seq_tokens, all_generated)):
            output_text = tokenizer.decode(seq, skip_special_tokens=True)
            print(f"\n{'='*50}")
            print(f"[第 {i+1} 条] 生成 {len(generated)} tokens:")
            print(output_text)
            print(f"{'='*50}")

    elif use_varlen:
        # ===== Varlen 模式：多 prompt 批量推理 =====
        # 读取 prompts
        prompts_file = None
        for arg in sys.argv:
            if arg.startswith("--prompts-file="):
                prompts_file = arg.split("=", 1)[1]

        if prompts_file:
            with open(prompts_file, "r", encoding="utf-8") as f:
                prompt_texts = [line.strip() for line in f if line.strip()]
        else:
            prompt_texts = [
                "你好，请介绍一下你自己",
                "今天天气怎么样？",
                "什么是人工智能？",
                "推荐三本科幻小说",
            ]

        print(f"\nVarlen batch 模式: {len(prompt_texts)} 条 prompt")

        # tokenize 每条 prompt
        prompt_ids_list = []
        for i, text in enumerate(prompt_texts):
            messages = [{"role": "user", "content": text}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            ids = tokenizer.encode(formatted, return_tensors="pt").to('cuda')
            prompt_ids_list.append(ids)
            print(f"  [{i}] {text[:30]}... ({ids.shape[1]} tokens)")

        # 生成
        print(f"\n生成中 (Varlen, batch={len(prompt_texts)})...")
        torch.manual_seed(42)
        torch.cuda.synchronize()
        t0 = __import__('time').time()
        seq_tokens, all_generated = model.generate_varlen(
            prompt_ids_list,
            max_new_tokens=256,
            temperature=0.6,
            top_k=top_k,
            eos_token_ids=[config.eos_token_id, config.bos_token_id],
        )
        torch.cuda.synchronize()
        elapsed = __import__('time').time() - t0

        total_new = sum(len(g) for g in all_generated)
        print(f"生成完成: {total_new} new tokens / {elapsed:.1f}s = {total_new/elapsed:.1f} tok/s")

        # 解码输出
        for i, (seq, generated) in enumerate(zip(seq_tokens, all_generated)):
            output_text = tokenizer.decode(seq, skip_special_tokens=True)
            print(f"\n{'='*50}")
            print(f"[第 {i+1} 条] 生成 {len(generated)} tokens:")
            print(output_text)
            print(f"{'='*50}")
    else:
        # ===== 单 prompt 模式（原逻辑）=====
        prompt_text = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "你好，请介绍一下你自己"
        messages = [{"role": "user", "content": prompt_text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(formatted, return_tensors="pt").to('cuda')
        print(f"\nPrompt: {prompt_text}")
        print(f"Token IDs: {prompt_ids.tolist()[0][:8]}...  ({prompt_ids.shape[1]} tokens)")

        # 生成
        generate_fn = model.generate_kvcache if use_kvcache else model.generate_naive
        mode = "KV-Cache" if use_kvcache else "Naive"
        print(f"\n生成中 ({mode})...")
        torch.manual_seed(42)
        torch.cuda.synchronize()
        t0 = __import__('time').time()
        output_ids = generate_fn(
            prompt_ids,
            max_new_tokens=256,
            temperature=0.6,
            top_k=top_k,
            eos_token_ids=[config.eos_token_id, config.bos_token_id],
        )
        torch.cuda.synchronize()
        elapsed = __import__('time').time() - t0
        new_tokens = output_ids.shape[1] - prompt_ids.shape[1]
        print(f"生成完成: {new_tokens} new tokens / {elapsed:.1f}s = {new_tokens/elapsed:.1f} tok/s")

        # 解码输出
        output_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        print(f"\n{'='*50}")
        print(output_text)
        print(f"{'='*50}")