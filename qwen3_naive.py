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
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


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
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # 强制转为 bf16，因为 .float() 固定返回 float32
        inv_freq = inv_freq.to(torch.bfloat16)
        pos = torch.arange(0, max_position)
        freqs = torch.outer(pos, inv_freq)
        self.register_buffer("cos_table", freqs.cos())
        self.register_buffer("sin_table", freqs.sin())

    def forward(self, q, k, positions):
        cos = self.cos_table[positions]
        sin = self.sin_table[positions]

        # q size: [batch_size, seq_len, head_num, head_dim]
        q_shape = q.shape
        k_shape = k.shape
        q_reshaped = q.reshape(*q.shape[:-1], -1, 2)
        # q0 : [batch_size, seq_len, head_num, head_dim // 2]
        q0, q1 = q_reshaped[..., 0], q_reshaped[..., 1]
        # 用于下面的乘法 
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
        q_new0 = q0 * cos - q1 * sin
        q_new1 = q0 * sin + q1 * cos
        q = torch.stack([q_new0, q_new1], dim=-1).reshape(q_shape)

        k_reshaped = k.reshape(*k.shape[:-1], -1, 2)
        k0, k1 = k_reshaped[..., 0], k_reshaped[..., 1]
        k_new0 = k0 * cos - k1 * sin
        k_new1 = k0 * sin + k1 * cos
        k = torch.stack([k_new0, k_new1], dim=-1).reshape(k_shape)
        return q, k 

class Qwen3Attention(nn.Module):
    """GQA + QK-Norm + RoPE —— Qwen3 的 attention 层"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.num_heads = config.num_attention_heads       # 16
        self.num_kv_heads = config.num_key_value_heads     # 8
        self.head_dim = config.head_dim                    # 128
        self.q_size = self.num_heads * self.head_dim        # 2048
        self.kv_size = self.num_kv_heads * self.head_dim    # 1024
        self.scale = self.head_dim ** -0.5

        # Q、K、V 由一个 Linear 同时产出（合并投影，推理更快）
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            self.q_size + self.kv_size * 2,    # 2048 + 1024 + 1024 = 4096
            bias=config.attention_bias,         # Qwen3: False
        )
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)

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
        k = k.repeat_interleave(n_repeat, dim=1)          # [batch, 8, seq, 128] → [batch, 16, seq, 128]
        v = v.repeat_interleave(n_repeat, dim=1)
        return k, v

    def forward(self, x):
        # x: [batch, seq_len, hidden_size]
        batch, seq_len, _ = x.shape

        # 1. QKV 投影 + 拆分
        qkv = self.qkv_proj(x)                           # [batch, seq, 4096]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 2. 整理形状 → [batch, seq_len, num_heads, head_dim]（先不 transpose）
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # 3. QK-Norm（对最后一维做，和 RMSNorm 一致）
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 4. RoPE（也按 [batch, seq_len, heads, dim] 格式，和 cos/sin 的 unsqueeze 匹配）
        positions = torch.arange(0, seq_len, device=x.device)
        q, k = self.rotary_emb(q, k, positions)

        # 5. transpose → [batch, num_heads, seq_len, head_dim]（attention 计算需要这个格式）
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 6. GQA：复制 KV
        k, v = self._repeat_kv(k, v)

        # 7. Scaled dot-product attention（和 gpt.py 完全一样）
        scores = (q @ k.transpose(-2, -1)) * self.scale   # [batch, 16, seq, seq]

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        # 用 score 自己的 dtype 的 -inf，避免 bf16 和 float32 混用
        neg_inf = torch.tensor(float('-inf'), dtype=scores.dtype, device=scores.device)
        scores = scores.masked_fill(causal_mask, neg_inf)

        attn_weights = torch.softmax(scores, dim=-1)        # [batch, 16, seq, seq]
        attn_output = attn_weights @ v                      # [batch, 16, seq, 128]

        # 7. 合并多头 → 输出投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attn_output)


class Qwen3MLP(nn.Module):
    """SwiGLU —— Qwen3 的 FFN 层"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        # 两条并行路径，合并成一个 Linear 输出（和 qkv 合并一样，省一次 kernel launch）
        self.gate_up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size * 2,    # 3072 * 2 = 6144，前半是 gate，后半是 up
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, x):
        # x: [batch, seq_len, hidden_size]
        gate_up = self.gate_up_proj(x)                # [batch, seq, 6144]
        gate, up = gate_up.chunk(2, dim=-1)            # 各 [batch, seq, 3072]
        x = torch.nn.functional.silu(gate) * up        # SiLU(a) * b
        return self.down_proj(x)                       # [batch, seq, 1024]


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
    """把 Qwen3 checkpoint 权重加载到我们的模型里。

    用 Python 原生 open/read 读 safetensors，不用 mmap，
    避免 Windows 页面文件限制的问题。
    """
    import struct

    with open(checkpoint_path, "rb") as f:
        # 1. 读目录
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size).decode("utf-8"))

        # 2. 建立层数（用于拼接判断）
        num_layers = sum(1 for m in model.modules() if isinstance(m, Qwen3DecoderLayer))
        device = next(model.parameters()).device

        # 暂存需要拼接的张量
        qkv_pending = {}       # layer_idx → {"q": tensor, "k": tensor, "v": tensor}
        gate_up_pending = {}   # layer_idx → {"gate": tensor, "up": tensor}

        for name, info in header.items():
            if name == "__metadata__":
                continue
            if name == "lm_head.weight":
                continue   # tie_word_embeddings，和 embed_tokens 共享，跳过

            # --- 需要拼接的：先暂存 ---
            matched = False

            # 检查是否 q_proj / k_proj / v_proj
            for layer_idx in range(num_layers):
                prefix = f"model.layers.{layer_idx}.self_attn"
                if name == f"{prefix}.q_proj.weight":
                    qkv_pending.setdefault(layer_idx, {})["q"] = _read_tensor(f, info, header_size, device)
                    matched = True; break
                if name == f"{prefix}.k_proj.weight":
                    qkv_pending.setdefault(layer_idx, {})["k"] = _read_tensor(f, info, header_size, device)
                    matched = True; break
                if name == f"{prefix}.v_proj.weight":
                    qkv_pending.setdefault(layer_idx, {})["v"] = _read_tensor(f, info, header_size, device)
                    matched = True; break
            if matched:
                continue

            # 检查是否 gate_proj / up_proj
            for layer_idx in range(num_layers):
                prefix = f"model.layers.{layer_idx}.mlp"
                if name == f"{prefix}.gate_proj.weight":
                    gate_up_pending.setdefault(layer_idx, {})["gate"] = _read_tensor(f, info, header_size, device)
                    matched = True; break
                if name == f"{prefix}.up_proj.weight":
                    gate_up_pending.setdefault(layer_idx, {})["up"] = _read_tensor(f, info, header_size, device)
                    matched = True; break
            if matched:
                continue

            # --- 直接映射 ---
            tensor = _read_tensor(f, info, header_size, device)
            model.get_parameter(name).data.copy_(tensor)

    # === 拼接 qkv_proj ===
    for layer_idx in range(num_layers):
        parts = qkv_pending.get(layer_idx, {})
        if len(parts) == 3:
            prefix = f"model.layers.{layer_idx}.self_attn"
            merged = torch.cat([parts["q"], parts["k"], parts["v"]], dim=0)
            model.get_parameter(f"{prefix}.qkv_proj.weight").data.copy_(merged)

    # === 拼接 gate_up_proj ===
    for layer_idx in range(num_layers):
        parts = gate_up_pending.get(layer_idx, {})
        if len(parts) == 2:
            prefix = f"model.layers.{layer_idx}.mlp"
            merged = torch.cat([parts["gate"], parts["up"]], dim=0)
            model.get_parameter(f"{prefix}.gate_up_proj.weight").data.copy_(merged)

    print("权重加载完成!")


def _read_tensor(f, info, header_size, device):
    """从文件读一个张量：seek → read → frombuffer → reshape → to device"""
    start, end = info["data_offsets"]
    f.seek(8 + header_size + start)
    raw = f.read(end - start)

    # dtype 映射
    dtype_map = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
    dtype = dtype_map[info["dtype"]]

    tensor = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(info["shape"])
    return tensor.to(device)
        


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

    # 3. 输入 prompt
    prompt_text = sys.argv[1] if len(sys.argv) > 1 else "今天天气真好"
    # Qwen3-0.6B 对聊天模板+思考模式支持不好，纯文本续写反而更稳
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to('cuda')
    print(f"\nPrompt: {prompt_text}")
    print(f"Token IDs: {prompt_ids.tolist()[0][:8]}...  ({prompt_ids.shape[1]} tokens)")

    # 4. 生成
    print("\n生成中...")
    output_ids = model.generate_naive(
        prompt_ids,
        max_new_tokens=256,
        temperature=0.6,
        top_k=20,
        eos_token_ids=[config.eos_token_id, config.bos_token_id],  # Qwen3 用两个 EOS
    )
    print(f"生成完成: {output_ids.shape[1]} tokens total")

    # 5. 解码输出
    output_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    print(f"\n{'='*50}")
    print(output_text)
    print(f"{'='*50}")