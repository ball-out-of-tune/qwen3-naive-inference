# qwen3-naive-inference

纯 PyTorch + Flash Attention，零推理框架依赖，跑通 Qwen3-0.6B 真实对话模型。已实现 Varlen 批处理、Continuous Batching、Paged Attention、Prefix Caching、Chunked Prefill、CUDA Graph、动态 KV Cache 分配、LIFO Preemption，以及 GRPO 训练循环。

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

# 6. Continuous Batching + Paged Attention + Prefix Caching
python qwen3_naive.py --continuous

# 7. Prefix Caching 验证
python test_prefix_cache.py

# 8. GRPO 训练
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
| `BlockManager` | 物理块分配/释放/Prefix Caching：引用计数 + 链式 hash + 全局查找表 |
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

## Prefix Caching：架构洞察

### 1. 核心思想：用 token ID 的 hash 来识别可共享的 KV cache

相同的 token 序列 → 相同的 K/V → 不需要重新计算。Prefix Caching 通过比对 **token ID**（不是 K/V 值）来判断前缀是否可共享：

```
序列 A: [sys prompt 400 tokens | 用户问题A]  → 前 400 token 的 K/V 已在 cache
序列 B: [sys prompt 400 tokens | 用户问题B]  → 前 256 token 命中 hash → 只算后 144 个
```

### 2. 链式 Hash：确保整个前缀链一致才共享

```python
Block 0 hash = hash(token_ids[0:256])
Block 1 hash = hash(token_ids[256:512], prefix=hash_of_block_0)  # 混入前一个 hash
Block 2 hash = hash(token_ids[512:768], prefix=hash_of_block_1)
```

如果只用 token ID 做 hash，两个不同前缀的序列可能在中间某个 block 发生碰撞。链式 hash 确保 block N 的共享**只有在前 N 个 block 全部相同时才发生**。

### 3. 释放 ≠ 删除：LRU 式缓存复用

```python
def deallocate(self, block_table):
    for b in block_table:
        self.ref_count[b] -= 1
        if self.ref_count[b] == 0:
            self.free.append(b)       # block 回空闲池
            # ★ hash 不删除! 后续序列可以捡回来用
```

即使原序列已经 decode 完，block 的 hash 标签还在。一旦有新序列需要相同前缀，block 从 free 池中被捡回来直接复用——K/V 数据还是热的，不需要重算。

只有当 block 被真正分配给**不同用途**时才清除旧 hash：

```python
def _pop_free_block(self):
    blk = self.free.pop()
    if self.block_hash[blk] != -1:    # 这个 block 之前有标签
        del self.hash_to_block[self.block_hash[blk]]  # 撕掉旧标签
    self.block_hash[blk] = -1          # 清空
    self.block_tokens[blk] = None
    self.ref_count[blk] = 1
    return blk
```

### 4. Prefill 时如何工作

有前缀缓存时，prefill 传给 `flash_attn_varlen_func` 的是**整个 k_cache/v_cache + block_table**，而非刚算出来的 K/V：

```python
# 只算新 token (后缀) 的 K/V
k_new, v_new = compute_kv(tokens[cached_len:])

# 写入 cache
k_cache[block_mapping] = k_new

# 有前缀缓存: 传整个 cache，flash_attn 通过 block_table 自己找
if block_tables is not None:
    k, v = k_cache, v_cache

flash_attn_varlen_func(q, k, v, 
    cu_seqlens_q=[0, new_len],        # 只统计新 token
    cu_seqlens_k=[0, cached+new],     # 统计全部 (含缓存前缀)
    block_table=block_tables)         # 页表告诉 flash_attn 去哪读缓存
```

`cu_seqlens_k > cu_seqlens_q` 时，flash_attn 知道前面那段 K/V 不在传入的 tensor 里，需要从 cache 通过 block_table 读取。

### 5. 同批次不共享，跨批次才生效

同一批 prefill 的序列之间不会共享——因为 hash 是 prefill 完成后才注册的。Prefix Caching 的价值体现在**后续批次**中。

### 6. 验证结果

```
第 1 条 prompt: 292 tokens  → prefill 全部 292
第 5 条 prompt: 293 tokens  → prefix 命中 256 tokens → 只 prefill 37 个新 token

top-1 一致: ✅
top-20 overlap: 19/20
block ref_count: 2 (两个 sequence 共享同一个物理 block)
```

## GPU 显存分配 & 优化：我们 vs nano-vllm

### KV Cache 分配公式（两者相同）

```python
# 双方都用同一套公式
free, total = torch.cuda.mem_get_info()
used   = total - free
peak   = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
peak_act = peak - current

num_blocks = int(total × gpu_memory_utilization - used - peak_act) // block_bytes
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^   ^^^^^^^^^
#            总预算 (90%)                        模型   激活值峰值
```

两者都用 2048-token warmup 实测 `peak_act`，不做外推。

### 显存优化洞察：RTX 3050 Ti 4GB WSL2

在 4GB 卡上，KV cache block 数从最初的 **16 → 37 → 35（稳定）**。关键发现：

**1. RoPE `cos_sin_cache` 不能每层独立创建（省 ~567MB）**

```python
# ❌ 我们的旧代码：28 层各创建一份 RotaryEmbedding
class Qwen3Attention:
    def __init__(self, config):
        self.rotary_emb = RotaryEmbedding(head_dim=128, max_position=40960, ...)
        # → 每层 register_buffer("cos_sin_cache", [40960, 128] float32)
        # → 28 层 × 21MB = 588MB 重复数据！

# ✅ nano-vllm 的做法：@lru_cache 共享同一实例
@lru_cache(8)
def get_rope(head_dim, max_position, rope_theta):
    return RotaryEmbedding(head_dim, max_position, rope_theta)
    # → 28 层调用，参数相同，返回同一个对象 → 只有 1 份 21MB cache

class Qwen3Attention:
    def __init__(self, config):
        self.rotary_emb = get_rope(128, 40960, 1000000)
```

**2. `tie_word_embeddings` 要用 `.data =` 而非 Parameter 赋值（省 ~311MB）**

```python
# ❌ 旧代码：lm_head 的旧 weight tensor 变成孤儿，占 311MB
self.lm_head.weight = self.model.embed_tokens.weight

# ✅ nano-vllm 的做法：只替换底层数据，不产生孤儿 tensor
self.lm_head.weight.data = self.model.embed_tokens.weight.data
```

**3. WSL2 CUDA 驱动有 ~842MB 固定开销**

`import torch` 后什么都不做，`mem_get_info` 就显示 used=842MB。这是 WSL2 的 CUDA 驱动翻译层开销，Linux 原生环境下没有。**nano-vllm 同样有这 842MB**。

**4. `torch.set_default_device('cuda')` 应建模后恢复**

nano-vllm 在 ModelRunner  init 结束后恢复 `set_default_device('cpu')`：
```python
torch.set_default_device("cuda")
self.model = _create_model(hf_config)
# ... 初始化完成 ...
torch.set_default_device("cpu")  # ← 恢复！避免后续临时 tensor 默认占 GPU
```

我们的热路径所有 tensor 都显式写了 `device='cuda'`，不受影响；但保持 `default_device('cuda')` 会让 PyTorch allocator 预留更多显存。

### 实测对比（4GB WSL2）

| 指标 | 优化前 | 优化后 | nano-vllm |
|------|--------|--------|-----------|
| `used`（驱动层面） | 2686MB | 2119MB | ~2306MB* |
| KV blocks | 16 | **35** | 47 |
| KV cache 显存 | 470MB | 1030MB | 1382MB |
| Throughput (4 seqs) | 278 tok/s | 276 tok/s | ~280 tok/s |

*nano-vllm 的 `used` 为推断值（total - allocated - kv_cache）*

**剩余差距（35 vs 47 blocks）来源**：
- nano-vllm 使用 fused QKV projection（1 次矩阵乘 vs 3 次），减少中间激活
- nano-vllm 使用 fused residual add（norm 和 add 一步完成），减少中间 tensor
- nano-vllm 使用 Triton `store_kvcache_kernel` 写入 cache，避免 Python 索引 + dtype cast
- 这些优化主要影响 `peak_act`（激活值峰值）和 PyTorch allocator 碎片

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

## Benchmark

### RTX 4090 48GB — 我们的引擎 vs nano-vllm

64 条随机序列批处理 (input≤512, output=256, Qwen3-0.6B)：

| 引擎 | 吞吐 | 备注 |
|------|------|------|
| **我们的引擎** | **3399.9 tok/s** | Chunked Prefill + CUDA Graph + Varlen batch prefill |
| nano-vllm | 2539.9 tok/s | enforce_eager=True（PyTorch 2.12.1 `@torch.compile` inductor bug） |

我们的引擎快 **34%**。优势来自统一的 varlen prefill 路径和 CUDA Graph。

### RTX 3050 Ti 4GB — 开发卡

| 配置 | 吞吐 |
|------|------|
| naive（无 KV-cache），1 seq | ~3.4 tok/s |
| varlen batch，4 seqs | ~6.3 tok/s |
| Continuous Batching + Paged，4 seqs | ~22.8 tok/s |

## 关键设计决策

### can_append/may_append 用 kv_len 而非 num_tokens

`num_tokens` = 序列中已有 token 数（含已采样的）。`kv_len` = KV cache 中实际写入的 token 数。

采样后 `num_tokens = kv_len + 1`，下一轮 decode 前 `can_append` 判断边界时应该用 `kv_len`：

```python
# ❌ 之前: num_tokens
need = seq.num_tokens % 256 == 0    # num_tokens=257 → 257%256=1 → 不加块 → 崩!

# ✅ 现在: kv_len
need = seq.kv_len % 256 == 0        # kv_len=256 → 256%256=0 → 加块 → OK
```

这个 bug 在 prompt 恰好 256 倍数时必现。手动步进测试能过但 `generate_continuous` 里崩，因为 `may_append` 被 LIFO preemption 改动错放进条件分支了。

### CUDA Graph 实现

直接调 `torch.cuda.CUDAGraph()` 底层 API（不经过 `@torch.compile`），为 bs=[1,2,4,8,16,32,48,64] 各录制一个图，所有图共享一个 memory pool。decode-only batch 时优先 `graph.replay()`，失败回退 eager。

**和 nano-vllm 的差异**：nano-vllm 用 `@torch.compile` 触发 CUDA Graph，对 PyTorch 版本敏感（2.12.1 的 inductor bug 导致崩溃）。我们直接调底层 API，不受影响。

### Preemption: LIFO 策略

缺块时驱逐 `running[-1]`（最新加入/生成进度最少的序列），保护已生成多步的老同志。单序列场景下退化为踢自己（和 nano-vllm 一样，此时无解）。

## 优化路线图

| 优先级 | 优化 | 状态 |
|--------|------|------|
| P0 | KV-Cache（单序列） | ✅ |
| P1 | Varlen 多 Prompt 批处理 | ✅ |
| P2 | Continuous Batching + Paged Attention | ✅ |
| P3 | 动态 KV Cache 分配（warmup + WSL2 fallback） | ✅ |
| P4 | Chunked Prefill | ✅ |
| P5 | Preemption（LIFO 驱逐） | ✅ |
| P6 | Prefix Caching（链式 hash + 引用计数） | ✅ |
| P7 | torch.compile | ❌ |
| P8 | CUDA Graph（16 个 bs 图 + graph.replay） | ✅ |

## 参考

- [nano-vllm](https://github.com/ball-out-of-tune/nano-vllm) — 本项目的工程化参考，包含 KV-Cache、Flash Attention、Continuous Batching、Prefix Caching
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) — 使用的模型权重
- [RoPE 论文](https://arxiv.org/abs/2104.09864) — Rotary Position Embedding
- [GRPO 论文](https://arxiv.org/abs/2402.03300) — Group Relative Policy Optimization
- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention 原始实现

## License

MIT
