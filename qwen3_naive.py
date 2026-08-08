from dataclasses import dataclass
import json
import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func


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

    def forward(self, x):
        orig_dtype = x.dtype
        x_f32 = x.float()
        x_f32 = x_f32 * torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_f32 * self.weight.float()).to(orig_dtype)


class BadRMSNorm:
    """不继承 nn.Module —— 演示用"""
    def __init__(self, hidden_size, eps=1e-6):
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

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

        # 分开的 Q、K、V 投影——和 checkpoint 名字对齐，不需要拼接
        self.q_proj = nn.Linear(self.hidden_size, self.q_size, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_size, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_size, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.q_size, self.hidden_size, bias=False)

        # QK-Norm：对 Q 和 K 做逐头归一化（Qwen3 没有 bias 所以有这个）
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # KV Cache（单序列，按需分配）
        self.k_cache = None
        self.v_cache = None

        # RoPE：旋转位置编码
        self.rotary_emb = RotaryEmbedding(
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

        # 1. QKV 投影
        q = self.q_proj(x)    # [batch, seq, q_size]
        k = self.k_proj(x)    # [batch, seq, kv_size]
        v = self.v_proj(x)    # [batch, seq, kv_size]

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

            q = self.q_proj(x)    # [tokens, q_size]
            k = self.k_proj(x)    # [tokens, kv_size]
            v = self.v_proj(x)    # [tokens, kv_size]

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

        # 1. 分开投影 Q、K、V
        q = self.q_proj(x)    # [batch, seq, 2048]
        k = self.k_proj(x)    # [batch, seq, 1024]
        v = self.v_proj(x)    # [batch, seq, 1024]

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


class Qwen3MLP(nn.Module):
    """SwiGLU —— Qwen3 的 FFN 层"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        # 分开的 gate 和 up 投影——和 checkpoint 名字对齐
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


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


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 模型总入口：Model + lm_head"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # tie_word_embeddings：lm_head 和 embedding 共享权重
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

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


def load_weights(model, checkpoint_path):
    """全部参数名和 checkpoint 对齐，直接一一映射，不需要拼接。"""
    import struct

    with open(checkpoint_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode("utf-8"))
        device = next(model.parameters()).device

        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name == "lm_head.weight":
                continue   # tie_word_embeddings

            tensor = _read_tensor(f, info, header_size, device).to(device)
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
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device('cuda')

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

    if use_varlen:
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