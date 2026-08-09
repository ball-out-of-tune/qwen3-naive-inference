# qwen3-naive-inference

纯 PyTorch + Flash Attention，零推理框架依赖，跑通 Qwen3-0.6B 真实对话模型。已实现 varlen 多 prompt 批处理、Continuous Batching、Paged Attention，以及 GRPO 训练循环。

**不需要 HuggingFace 的模型代码，不需要推理框架。有 config.json + model.safetensors + PyTorch 就能跑。**

## 快速开始

```bash
# 1. 下载 Qwen3-0.6B 权重到本地
# https://huggingface.co/Qwen/Qwen3-0.6B

# 2. 安装依赖
pip install torch transformers flash-attn

# 3. 单条推理（无 KV cache）
python qwen3_naive.py "你好，请介绍一下你自己"

# 4. KV-Cache 推理（单序列 prefill + decode）
python qwen3_naive.py --kvcache "你好"

# 5. Varlen 多 prompt 批量推理（零填充）
python qwen3_naive.py --varlen
python qwen3_naive.py --varlen --prompts-file=prompts.txt

# 6. Continuous Batching + Paged Attention
python qwen3_naive.py --continuous

# 7. GRPO 训练
python qwen3_grpo_train.py

# Benchmark
python qwen3_naive.py --bench --num-seqs=64 --no-topk
```

## 代码结构

| 模块 | 做了什么 |
|------|----------|
| `Qwen3Config` | 从 config.json 读取模型参数 |
| `RMSNorm` | 和 PyTorch 内置一致，内部 float32 保证精度 |
| `RotaryEmbedding` | RoPE 旋转位置编码，自动适配 3D(varlen) / 4D(batch×seq) 输入 |
| `Qwen3Attention` | GQA + QK-Norm + RoPE，三路 forward：普通、varlen prefill、paged decode |
| `Qwen3MLP` | SwiGLU（SiLU 门控 × up_proj） |
| `Qwen3DecoderLayer` | Pre-Norm → Attention → Pre-Norm → MLP，残差连接 |
| `Qwen3Model` | Embedding + 28 层 DecoderLayer + 最终 Norm |
| `Qwen3ForCausalLM` | Model + lm_head，tie_word_embeddings 权重共享 |
| `VarlenContext` | 全局上下文：cu_seqlens / positions / max_seqlen，避免改所有函数签名 |
| `Sequence` | 序列状态机：WAITING → RUNNING → FINISHED，持有 block_table 页表 |
| `BlockManager` | 物理块分配/释放，按需扩展（`can_append` / `may_append`） |
| `load_weights` | `struct.unpack` 解析 safetensors → 逐张量拷进模型 |

### 4 种生成模式

| 方法 | KV Cache | 批处理 | Paged | 说明 |
|------|----------|--------|-------|------|
| `generate_naive` | ❌ | ❌ | ❌ | 每步完整 forward，最基础实现 |
| `generate_kvcache` | ✅ | ❌ | ❌ | 单序列 prefill + decode，Flash Attention 读写 cache |
| `generate_varlen` | ❌ | ✅ | ❌ | 多 prompt 变长拼接成 `[total_tokens]`，一次 forward |
| `generate_continuous` | ✅ | ✅ | ✅ | Continuous Batching + Paged Attention，序列可动态加入/退出 |

### 双模式 Attention 前向

Attention 层有三套 forward，根据场景切换：

| 方法 | 场景 | Flash Attention 调用 | cache 操作 |
|------|------|---------------------|-----------|
| `forward()` | 无 cache、单序列/varlen | `flash_attn_varlen_func` | 不读写 cache |
| `forward_prefill()` | 新序列首次处理（prompt） | `flash_attn_varlen_func` | 写入 KV cache（通过 block_mapping 页表寻址） |
| `forward_decode()` | 已有序列生成下一个 token | `flash_attn_with_kvcache` | 读全部历史 + 写入新 token |

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

### `cached_len` 未更新（导致 Paged Attention 输出乱码）

prefill 完成后必须设置 `seq.cached_len = seq.num_prompt_tokens`。否则 decode 第一步会把第一个生成 token 的 K/V 写到错误位置（覆盖第二个 prompt token），`flash_attn_with_kvcache` 也只读到错误的历史长度，输出变成乱码（中文/泰文/阿拉伯文混搭）。

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

## Continuous Batching + Paged Attention：架构洞察

### 1. Page Table 是整数映射，不是内存拷贝

```
逻辑层（Sequence）           物理层（BlockManager）
┌─────────────────┐         ┌──────────────────────┐
│ seq1.block_table │ ──────→ │ block 3 │ block 7 │ ... │
│ = [3, 7, 12]    │         │ (256 tok K/V each)   │
│ num_tokens = 700 │         └──────────────────────┘
└─────────────────┘

K/V 写入：物理位置 = block_table[pos // 256] × 256 + (pos % 256)
K/V 读取：flash_attn_with_kvcache 直接接收 block_table，内部寻址
```

Block 之间不需要物理连续——页表（`block_table: list[int]`）把逻辑位置映射到物理块号。运行时操作全是整数索引，没有 `torch.cat` / `torch.stack`。

### 2. Prefill 和 Decode 是两条完全不同的路径

| | Prefill | Decode |
|---|---|---|
| 处理量 | 整个 prompt（可能数千 token） | 1 token / 序列 |
| 计算特征 | compute-bound（大矩阵乘法） | memory-bound（读 KV cache） |
| Flash Attn | `flash_attn_varlen_func` | `flash_attn_with_kvcache` |
| KV cache | 写入全部 token 的 K/V | 写入 1 个 token，读全部历史 |
| 线程排布 | varlen 拼接，一次 forward | 逐序列 forward_decode |

### 3. Prefill-First 调度

```
while waiting or running:
    # Phase 1: 优先 prefill（保证新请求尽快开始）
    for seq in waiting:
        if block_manager 有空闲块:
            分配 blocks → prefill → waiting → running
    # Phase 2: decode 已在运行的序列
    for seq in running:
        追加 block（如果需要） → decode 1 token → 采样
        如果 done: deallocate blocks → running → finished
```

为什么 prefill 优先？如果先 decode 再 prefill，新来的序列必须等一整轮才能开始，TTFT（Time To First Token）变差。

### 4. Block 按需分配，惰性扩展

```python
# Prefill: 一次性分配所有需要的 block
seq.num_prompt_tokens = 700  → 需要 ceil(700/256) = 3 blocks
bm.allocate(seq)             → 从 free 队列取 3 个 block

# Decode: 每次检查是否需要新 block
bm.can_append(seq)   → num_tokens % 256 == 0 时需要新 block
bm.may_append(seq)   → 从 free 队列取 1 个 block
```

序列结束时所有 block 归还 free 队列。Block 有借有还，不会泄漏。

### 5. 全局 KV Cache 预分配

```python
# 启动时一次性分配，运行时零 GPU 分配
k_cache = torch.empty(num_layers, NUM_BLOCKS, BLOCK_SIZE, n_kv_heads, head_dim)
v_cache = torch.empty(num_layers, NUM_BLOCKS, BLOCK_SIZE, n_kv_heads, head_dim)
```

运行时所有操作都是**对已有 tensor 的 view 写入**——`k_cache[blk * 256 + off] = k_new`，不产生新的 GPU 分配。这确保了推理延迟稳定，不会因 CUDA malloc 产生抖动。

## GPU 显存分配：我们 vs nano-vllm

### nano-vllm 的动态计算

```python
# model_runner.py — warmup 测激活值峰值，再分配 KV cache
free, total = torch.cuda.mem_get_info()
used = total - free
peak   = torch.cuda.memory_stats()["allocated_bytes.all.peak"]    # warmup 记录
current = torch.cuda.memory_stats()["allocated_bytes.all.current"] # 当前常住

num_blocks = int(total × 0.9 - used - peak + current) // block_bytes
```

每一步的语义：

| 项 | 含义 | 如何得到 |
|----|------|---------|
| `total × 0.9` | 总预算（留 10% 安全边界） | 配置 |
| `used` | 驱动层面物理占用（模型 + CUDA 上下文） | `mem_get_info` |
| `peak` | warmup 时峰值分配（模型 + 最大激活值） | `memory_stats` |
| `current` | 当前常住分配（模型权重，激活值已被 `empty_cache` 释放） | `memory_stats` |
| `peak - current` | **激活值峰值** | warmup 实测 |
| `used - current` | CUDA 驱动开销 + PyTorch allocator 预留 | 实测 |

公式展开就是：

```
KV cache = 0.9 × total - 模型权重 - 激活值峰值 - CUDA 开销
```

**关键设计：`peak - current` 通过 warmup 实测，而非公式估算。** warmup 用 `max_num_batched_tokens` 的最大值跑一次 forward，记录最坏情况的激活值峰值，保证 KV cache 分配后真实推理不会 OOM。

warmup 的规模由配置决定：

```python
seq_len  = min(max_num_batched_tokens, max_model_len)  # 默认: min(16384, 4096) = 4096
num_seqs = min(max_num_batched_tokens // seq_len, max_num_seqs)  # 默认: 4
# 跑 4 × 4096 = 16384 token 同时 forward
```

### 我们当前的实现：硬编码

```python
# qwen3_naive.py — 写死 block 数量
NUM_BLOCKS = 64
BLOCK_SIZE = 256
```

不足：
- 不检查显存是否够用
- 换显卡（如 4090 24GB）不会自动利用更多显存
- 如果 64 blocks × 1.88 GB > 剩余显存，启动即 OOM

下一步应该参考 nano-vllm 的动态分配，加载模型后跑 warmup → 测 `peak - current` → 计算安全的 `NUM_BLOCKS`。

## GRPO 训练

`qwen3_grpo_train.py` — 基于 `qwen3_naive.py` 的单卡 GRPO 训练循环。

核心流程：
1. **Rollout**：每条 prompt 采样 N_ROLLOUT 条 response，记录每步 log_prob + reward
2. **GRPO 优势**：同一 prompt 的多条 response 组内归一化 `(reward - mean) / std`
3. **PPO Clipped Loss**：`ratio = exp(logπ_new - logπ_old)`，clip 到 `[1-ε, 1+ε]`，取 min
4. **Backward + Step**：多 epoch 更新

```python
# 关键超参数
N_ROLLOUT = 4         # 每个 prompt 采样几条
PPO_EPOCHS = 1        # 每批数据训练几轮
CLIP_EPSILON = 0.2    # PPO clip 范围
TEMPERATURE = 0.8     # RL 用高温度鼓励探索
```

## Benchmark（RTX 3050 Ti 4GB）

| 配置 | 吞吐 | 说明 |
|------|------|------|
| naive（无 KV-cache），1 seq | ~3.4 tok/s | 每步完整 forward，二次复杂度 |
| varlen batch（无 KV-cache），4 seqs | ~6.3 tok/s | 多 prompt 零填充批处理 |
| Continuous Batching + Paged，4 seqs | ~22.8 tok/s | KV cache + 批处理 + 页表管理 |
| nano-vllm（完整工程化） | ~41 tok/s | CUDA Graph + 更多 kernel 优化 |

## 优化路线图

| 优先级 | 优化 | 提速/效果 | 状态 |
|--------|------|-----------|------|
| P0 | KV-Cache（单序列） | ~10× | ✅ `generate_kvcache` |
| P1 | Varlen 多 Prompt 批处理 | ~2× | ✅ `generate_varlen` |
| P2 | Continuous Batching + Paged Attention | ~3× + 省显存 | ✅ `generate_continuous` |
| P3 | 动态 KV Cache 分配（warmup 测 peak） | 适配不同显卡 | ❌ 当前硬编码 `NUM_BLOCKS=64` |
| P4 | Chunked Prefill | 避免长 prompt 显存峰值 | ❌ |
| P5 | Preemption（显存不足时驱逐序列） | 提高并发上限 | ❌ |
| P6 | Prefix Caching（hash 检测公共前缀） | 省显存 + 加速 | ❌ |
| P7 | torch.compile | ~1.3× | ❌ |
| P8 | CUDA Graph | 减少 kernel launch 开销 | ❌ |

## 参考

- [nano-vllm](https://github.com/ball-out-of-tune/nano-vllm) — 本项目的工程化参考，包含 KV-Cache、Flash Attention、Continuous Batching、Prefix Caching
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) — 使用的模型权重
- [RoPE 论文](https://arxiv.org/abs/2104.09864) — Rotary Position Embedding
- [GRPO 论文](https://arxiv.org/abs/2402.03300) — Group Relative Policy Optimization
- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention 原始实现

## License

MIT
