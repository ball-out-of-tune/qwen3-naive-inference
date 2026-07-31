# qwen3-naive-inference

400 行纯 PyTorch，零框架依赖，跑通 Qwen3-0.6B 真实对话模型。

**不需要 HuggingFace 的模型代码，不需要推理框架。有 config.json + model.safetensors + PyTorch 就能跑。**

## 快速开始

```bash
# 1. 下载 Qwen3-0.6B 权重到本地
# https://huggingface.co/Qwen/Qwen3-0.6B

# 2. 安装依赖
pip install torch transformers

# 3. 运行（需 ~4GB 显存的 NVIDIA GPU）
python qwen3_naive.py "你好，请介绍一下你自己"

# Benchmark
python qwen3_naive.py --bench --num-seqs=64 --no-topk
```

## 代码结构

| 模块 | 行数 | 做了什么 |
|------|------|----------|
| `Qwen3Config` | ~40 行 | 从 config.json 读取模型参数 |
| `RMSNorm` | ~10 行 | 和 PyTorch 内置一致，内部 float32 保证精度 |
| `RotaryEmbedding` | ~30 行 | RoPE 旋转位置编码，cos/sin 预计算查表 |
| `Qwen3Attention` | ~80 行 | GQA + QK-Norm + RoPE + causal mask attention |
| `Qwen3MLP` | ~10 行 | SwiGLU（SiLU 门控 × up_proj） |
| `Qwen3DecoderLayer` | ~10 行 | Pre-Norm → Attention → Pre-Norm → MLP，残差连接 |
| `Qwen3Model` | ~20 行 | Embedding + 28 层 DecoderLayer + 最终 Norm |
| `Qwen3ForCausalLM` | ~20 行 | Model + lm_head，tie_word_embeddings 权重共享 |
| `load_weights` | ~30 行 | `struct.unpack` 解析 safetensors → 逐张量拷进模型 |
| `generate_naive` | ~20 行 | 自回归循环：forward → softmax → multinomial → 拼接 |

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

## Benchmark（RTX 3050 Ti 4GB）

| 配置 | 吞吐 |
|------|------|
| naive（无 KV-cache），4 seqs | ~15 tok/s |
| nano-vllm（KV-cache + FlashAttn + batching） | ~41 tok/s |

## 下一步优化

| 优先级 | 优化 | 提速 | 说明 |
|--------|------|------|------|
| P0 | KV-Cache | ~10× | 缓存历史的 Key/Value，每步只算新 token |
| P1 | Flash Attention | ~3× | 用 `flash_attn_varlen_func` 减少 HBM 读写 |
| P2 | Continuous Batching | ~2× | 请求完成立刻换新请求，GPU 不空转 |
| P3 | PagedAttention | 省 50% 显存 | KV-cache 按页分配，消除碎片 |
| P4 | torch.compile | ~1.3× | 加 `@torch.compile`，JIT 编译优化 kernel |

## 参考

- [nano-vllm](https://github.com/ball-out-of-tune/qwen3-naive-inference) — 本项目的工程化版本，包含 KV-Cache、Flash Attention、Continuous Batching
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B) — 使用的模型权重
- [RoPE 论文](https://arxiv.org/abs/2104.09864) — Rotary Position Embedding

## License

MIT
