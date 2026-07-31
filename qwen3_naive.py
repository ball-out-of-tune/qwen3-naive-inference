from dataclasses import dataclass
import json
import torch
import torch.nn as nn

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

        # cos/sin: [seq_len, head_dim//2] each
        cos_sin = self.cos_sin_cache[positions]            # [seq_len, head_dim]
        cos, sin = cos_sin.chunk(2, dim=-1)                # each [seq_len, head_dim//2]
        # 加 batch 和 heads 维度用于广播
        cos = cos.unsqueeze(0).unsqueeze(2)                # [1, seq_len, 1, head_dim//2]
        sin = sin.unsqueeze(0).unsqueeze(2)

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

        # RoPE：旋转位置编码
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )

    def _repeat_kv(self, k, v):
        """GQA：把 KV 从 num_kv_heads 复制到 num_heads"""
        n_repeat = self.num_heads // self.num_kv_heads   # 16 // 8 = 2
        k = k.repeat_interleave(n_repeat, dim=1)
        v = v.repeat_interleave(n_repeat, dim=1)
        return k, v

    def forward(self, x):
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
    num_seqs = 4
    max_input_len = 64
    max_output_len = 128

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

    model_path = "C:/Users/16874/Downloads/Qwen3-0.6B"

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

    # 3. 输入 prompt（使用聊天模板）
    prompt_text = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "你好，请介绍一下你自己"
    messages = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(formatted, return_tensors="pt").to('cuda')
    print(f"\nPrompt: {prompt_text}")
    print(f"Token IDs: {prompt_ids.tolist()[0][:8]}...  ({prompt_ids.shape[1]} tokens)")

    # 4. 生成
    use_topk = "--no-topk" not in sys.argv
    top_k = 20 if use_topk else None
    if not use_topk:
        print("(top_k=None, 全词表采样)")

    print("\n生成中...")
    output_ids = model.generate_naive(
        prompt_ids,
        max_new_tokens=256,
        temperature=0.6,
        top_k=top_k,
        eos_token_ids=[config.eos_token_id, config.bos_token_id],  # Qwen3 用两个 EOS
    )
    print(f"生成完成: {output_ids.shape[1]} tokens total")

    # 5. 解码输出
    output_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    print(f"\n{'='*50}")
    print(output_text)
    print(f"{'='*50}")