# qwen3-naive-inference

纯 PyTorch + Flash Attention，零推理框架依赖，跑通 Qwen3-0.6B 真实对话模型。现已支持 varlen 多 prompt 批处理推理。

**不需要 HuggingFace 的模型代码，不需要推理框架。有 config.json + model.safetensors + PyTorch 就能跑。**

## 快速开始

```bash
# 1. 下载 Qwen3-0.6B 权重到本地
# https://huggingface.co/Qwen/Qwen3-0.6B

# 2. 安装依赖
pip install torch transformers flash-attn

# 3. 单条推理
python qwen3_naive.py "你好，请介绍一下你自己"

# 4. 多 prompt 批量推理（varlen，零填充）
python qwen3_naive.py --varlen
python qwen3_naive.py --varlen --prompts-file=prompts.txt

# 5. KV-Cache 推理
python qwen3_naive.py --kvcache "你好"

# Benchmark
python qwen3_naive.py --bench --num-seqs=64 --no-topk
```

## 代码结构

| 模块 | 做了什么 |
|------|----------|
| `Qwen3Config` | 从 config.json 读取模型参数 |
| `RMSNorm` | 和 PyTorch 内置一致，内部 float32 保证精度 |
| `RotaryEmbedding` | RoPE 旋转位置编码，自动适配 3D(varlen) / 4D(batch×seq) 输入 |
| `Qwen3Attention` | GQA + QK-Norm + RoPE，自动检测 varlen context 走 flash_attn_varlen_func |
| `Qwen3MLP` | SwiGLU（SiLU 门控 × up_proj） |
| `Qwen3DecoderLayer` | Pre-Norm → Attention → Pre-Norm → MLP，残差连接 |
| `Qwen3Model` | Embedding + 28 层 DecoderLayer + 最终 Norm |
| `Qwen3ForCausalLM` | Model + lm_head，tie_word_embeddings 权重共享 |
| `VarlenContext` | 全局上下文：cu_seqlens / positions / max_seqlen，避免改所有函数签名 |
| `load_weights` | `struct.unpack` 解析 safetensors → 逐张量拷进模型 |
| `generate_naive` | 单条自回归：forward → softmax → multinomial → 拼接 |
| `generate_kvcache` | KV-Cache 版生成：prefill 一次 + decode N 次 |
| `generate_varlen` | **批量生成**：多条 prompt 拼接成一个 [total_tokens] 张量，一次 forward

## 关键 Bugs 修复

整个开发中最重要的一课：**推理框架的正确性不在于"像不像论文公式"，而在于"每一步都和训练时一模一样"。**

### RoPE 维度配对（唯一导致模型输出退化的 bug）

```python
# ❌ 错误：相邻配对 (d0,d1) (d2,d3) ...
x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
x0, x1 = x_reshaped[..., 0], x_reshaped[..., 1]

# ✅ 正确：前半后半配对 (d0,d64) (d1,d65) ...
x1, x2 = x.chunk(2, dim=-1)
```

模型权重按 `chunk(2)` 的方式训练，推理时用 `reshape(-1,2)` 会让 64 个旋转频率全部对应错的维度对——每个 token 对所有 token 的注意力权重都错了，28 层累积后输出彻底退化成一个 `<think>` 循环。

### RMSNorm / RoPE 的 bf16 精度

保持和训练时一致：内部用 float32 计算，结果转回 bf16。全程 bf16 会在 28 层后累积可感知的误差。

## Varlen 多 Prompt 批处理：核心洞察

### 1. batch 和 seq 合并成一个维度

传统做法：`[batch, max_len]` 填充张量 → 大量 pad token 浪费算力。

Varlen 做法：所有序列的 token 直接拼接成 `[total_tokens]` 一维数组：

```
序列0: [a, b, c]        长度 3
序列1: [x, y]           长度 2
序列2: [p, q, r, s]     长度 4

→ input_ids = [a, b, c, x, y, p, q, r, s]   # 9 个 token，零填充
→ cu_seqlens = [0, 3, 5, 9]                 # 唯一的边界标记
→ positions  = [0, 1, 2, 0, 1, 0, 1, 2, 3]  # 各自序列内的绝对位置
```

模型全程不知道"有几条序列"——Linear、RMSNorm、MLP、Embedding 都只跟最后一维（hidden_size）打交道，不关心第一维是 batch×seq 还是 total_tokens。**唯一需要特殊处理的是 Attention 层**，靠 `cu_seqlens` 知道哪些 token 之间能互相看到。

### 2. cu_seqlens 是唯一的边界标记

不需要 padding mask，不需要 attention_mask。`cu_seqlens` 传给 `flash_attn_varlen_func`，它内部搞定两件事：
- **跨序列隔离**：token 0~2 不会 attend 到 token 3~8
- **子序列内 causal**：`causal=True` 配合 `cu_seqlens`，自动在每条序列内部做下三角掩码

### 3. 全局 Context 避免改函数签名

当前调用链：`CausalLM → Model → DecoderLayer → Attention`，共 5 层。

用 `set_varlen_context()` / `get_varlen_context()` 全局变量透传元数据，跟 [nano-vllm](https://github.com/ball-out-of-tune/qwen3-naive-inference) 的做法一致。Attention 层自己从 context 取 `cu_seqlens` 和 `positions`，中间所有层签名不变。

### 4. flash_attn 原生支持 GQA

varlen 路径中：
- q: `[total_tokens, num_heads, head_dim]`（16 heads）
- k, v: `[total_tokens, num_kv_heads, head_dim]`（8 heads）

头数不同，`flash_attn_varlen_func` 内部自动做 GQA repeat，**不需要手动调 `_repeat_kv`**。

### 5. 数值差异是预期的

flash_attn 使用分块在线 softmax，和手写 `q@k^T → softmax → @v` 在浮点精度上有微小差异。单条 forward 的 top-1 token 一致，top-20 重叠约 16~18/20。28 层累积后 `multinomial` 采样可能分叉，这是正常现象——flash_attn 本身是经过严格验证的算法。

## Benchmark（RTX 3050 Ti 4GB）

| 配置 | 吞吐 |
|------|------|
| naive（无 KV-cache），1 seq | ~3.4 tok/s |
| varlen batch（无 KV-cache），4 seqs | ~6.3 tok/s |
| nano-vllm（KV-cache + FlashAttn + batching） | ~41 tok/s |

## 下一步优化

| 优先级 | 优化 | 提速 | 状态 |
|--------|------|------|------|
| P0 | KV-Cache | ~10× | ✅ 已实现 `generate_kvcache`，仅支持单序列 |
| P1 | Varlen 多 Prompt 批处理 | ~2× | ✅ 已实现 `generate_varlen`，无填充、零浪费 |
| P2 | Varlen + KV-Cache 结合 | ~5× | ❌ 待实现：deocde 阶段复用 KV cache |
| P3 | Continuous Batching | ~2× | ❌ 待实现：请求完成立刻换新请求，GPU 不空转 |
| P4 | PagedAttention | 省 50% 显存 | ❌ 待实现：KV-cache 按页分配，消除碎片 |
| P5 | torch.compile | ~1.3× | ❌ 加 `@torch.compile`，JIT 编译优化 kernel |

## 参考

- [nano-vllm](https://github.com/ball-out-of-tune/qwen3-naive-inference) — 本项目的工程化版本，包含 KV-Cache、Flash Attention、Continuous Batching
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) — 使用的模型权重
- [RoPE 论文](https://arxiv.org/abs/2104.09864) — Rotary Position Embedding

## License

MIT
