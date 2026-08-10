"""
Prefix Caching 验证

测试流程:
  1. 第一批: prefill 4 条 prompt → hash_blocks 注册
  2. 第二批: 第 5 条 prompt, 前缀和第 1 条完全一样
     → 验证前缀命中 (kv_len > 0)
     → 验证只 prefill 新 token (cu_seqlens_q < cu_seqlens_k)
     → 对比: prefix-cached prefill vs 完整 forward (都对同一条 seq5)
"""

import torch
from transformers import AutoTokenizer
from qwen3_naive import (
    Qwen3Config, Qwen3ForCausalLM, load_weights,
    Sequence, BlockManager,
)

MODEL_PATH = "/mnt/c/Users/16874/Downloads/Qwen3-0.6B"


def load_model_and_tokenizer():
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("加载模型...")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    config = Qwen3Config.from_json(f"{MODEL_PATH}/config.json")
    model = Qwen3ForCausalLM(config)
    load_weights(model, f"{MODEL_PATH}/model.safetensors")
    vram = torch.cuda.memory_allocated() / 1e9
    params = sum(p.numel() for p in model.parameters())
    print(f"参数: {params:,}  |  VRAM: {vram:.2f} GB")
    return model, tokenizer, config


def tokenize(text, tokenizer):
    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return tokenizer.encode(formatted, return_tensors="pt").to("cuda")


def clear_caches(model):
    for layer in model.model.layers:
        layer.self_attn.k_cache = None
        layer.self_attn.v_cache = None
    torch.cuda.empty_cache()


def main():
    model, tokenizer, config = load_model_and_tokenizer()

    # ================================================================
    # 构造测试: 长公共前缀 (跨多个 256 block)
    # ================================================================
    common_prefix = ("请详细解释什么是机器学习，包括监督学习、无监督学习和强化学习这三种主要范式，"
                     "它们的区别、应用场景以及各自的优缺点。请分别从数学原理、算法流程、"
                     "实际代码实现三个层面展开论述，并配合具体的案例加以说明。") * 5

    prompts_batch1 = [
        common_prefix + "具体说说监督学习的数学原理。",
        "今天天气真好啊，适合出去运动一下。",
        "用Python写一个二分查找算法，要求考虑边界条件。",
        "什么是深度学习中的反向传播算法？请详细解释。",
    ]

    # 第 5 条: 前缀和第 1 条一样, 后缀不同
    prompt5 = common_prefix + "具体说说无监督学习的数学原理。"

    ids_batch1 = [tokenize(p, tokenizer) for p in prompts_batch1]
    ids5 = tokenize(prompt5, tokenizer)

    print(f"\n第 1 条 prompt: {ids_batch1[0].shape[1]} tokens")
    print(f"第 5 条 prompt: {ids5.shape[1]} tokens")
    print(f"第 5 条需要的 block 数: {(ids5.shape[1] + 255) // 256}")


    # ================================================================
    # 第一步: 完整 forward 算 seq5 — 作为 ground truth
    # ================================================================
    print(f"\n{'='*60}")
    print("Ground Truth: seq5 完整 forward (无 cache)")
    print(f"{'='*60}")

    clear_caches(model)
    with torch.no_grad():
        logits_full = model(ids5)   # [1, seq_len, vocab]
    last_logits_full = logits_full[0, -1, :].float()
    _, full_topk = torch.topk(last_logits_full, 20)
    full_top1 = full_topk[0].item()
    full_top1_text = tokenizer.decode([full_top1])
    print(f"  top-1: {full_top1} → '{full_top1_text}'")


    # ================================================================
    # 第二步: 第一批 prefill 4 条 → 注册 hash
    # ================================================================
    clear_caches(model)
    bm = BlockManager(model.NUM_BLOCKS, model.BLOCK_SIZE)
    model.allocate_global_kv_cache()

    print(f"\n{'='*60}")
    print("第一批: prefill 4 条 prompt + 注册 hash")
    print(f"{'='*60}")

    seqs1 = [Sequence(ids[0].tolist(), {"temperature": 0.6, "max_tokens": 64})
             for ids in ids_batch1]

    for s in seqs1:
        bm.allocate(s)
        print(f"  seq (blocks={len(s.block_table)}, num_cached={s.kv_len})")

    input_ids1, pos1, cu_q1, cu_k1, m_q1, m_k1, blk_map1, bt1 = model._prepare_prefill(seqs1)
    print(f"\n  总 prefill token 数: {input_ids1.shape[0]}")
    print(f"  block_tables is None: {bt1 is None}")

    model.model.forward_prefill(input_ids1, pos1, cu_q1, cu_k1, m_q1, m_k1, blk_map1, bt1)
    for s in seqs1:
        s.kv_len = s.num_prompt_tokens
        bm.hash_blocks(s)

    print(f"  注册后 hash_to_block 有 {len(bm.hash_to_block)} 个条目")

    # ================================================================
    # 第三步: 第二批 — seq5 来, 前缀应命中 seq1
    # ================================================================
    print(f"\n{'='*60}")
    print("第二批: seq5 allocate (应命中前缀)")
    print(f"{'='*60}")

    seq5 = Sequence(ids5[0].tolist(), {"temperature": 0.6, "max_tokens": 64})
    print(f"  allocate 前: block_table={seq5.block_table}, num_cached={seq5.kv_len}")

    bm.allocate(seq5)

    print(f"  allocate 后: block_table={seq5.block_table}")
    print(f"  kv_len={seq5.kv_len}")
    print(f"  共享的 block: {seq5.kv_len // model.BLOCK_SIZE}")
    print(f"  新 block: {len(seq5.block_table) - seq5.kv_len // model.BLOCK_SIZE}")

    # ================================================================
    # 第四步: 比对 prefix-cached prefill vs 完整 forward
    # ================================================================
    print(f"\n{'='*60}")
    print("Prefill: prefix-cached (只算后缀)")
    print(f"{'='*60}")

    input_ids5, pos5, cu_q5, cu_k5, m_q5, m_k5, blk_map5, bt5 = model._prepare_prefill([seq5])
    print(f"  input_ids: {input_ids5.shape[0]} tokens (总共{seq5.num_prompt_tokens}, 缓存{seq5.kv_len})")
    print(f"  cu_q (新token): {cu_q5.tolist()}")
    print(f"  cu_k (含缓存):   {cu_k5.tolist()}")
    print(f"  cu_k > cu_q: {cu_k5[-1].item() > cu_q5[-1].item()}")

    h5 = model.model.forward_prefill(input_ids5, pos5, cu_q5, cu_k5, m_q5, m_k5, blk_map5, bt5)
    logits_cached = model.lm_head(h5)
    last_logits_cached = logits_cached[-1, :].float()

    # ================================================================
    # 第五步: 对比结果
    # ================================================================
    print(f"\n{'='*60}")
    print("结果对比")
    print(f"{'='*60}")

    k_val = 20
    _, ref_topk = torch.topk(last_logits_full, k_val)
    _, test_topk = torch.topk(last_logits_cached, k_val)
    overlap = len(set(ref_topk.tolist()) & set(test_topk.tolist()))

    top1_match = ref_topk[0].item() == test_topk[0].item()

    l2_dist = torch.norm(last_logits_full - last_logits_cached, p=2).item()
    l2_rel = l2_dist / torch.norm(last_logits_full, p=2).item()

    print(f"  prefix 命中: {seq5.kv_len > 0} ({seq5.kv_len} tokens)")
    print(f"  cu_k > cu_q: {cu_k5[-1].item() > cu_q5[-1].item()}")
    print(f"  top-1 一致: {top1_match}")
    print(f"    full: {ref_topk[0].item()} → '{tokenizer.decode([ref_topk[0].item()])}'")
    print(f"    cached: {test_topk[0].item()} → '{tokenizer.decode([test_topk[0].item()])}'")
    print(f"  top-{k_val} overlap: {overlap}/{k_val}")
    print(f"  L2 相对误差: {l2_rel:.6f}")

    # 验证 ref_count
    print(f"\n  引用计数:")
    shared_blocks = seq5.kv_len // model.BLOCK_SIZE
    for i in range(shared_blocks):
        bid = seq5.block_table[i]
        print(f"    block[{i}]={bid}: ref_count={bm.ref_count[bid]} (应为2: seq1+seq5)")

    if seq5.kv_len > 0 and cu_k5[-1].item() > cu_q5[-1].item() and top1_match and overlap >= 15:
        print(f"\n  ✅ PASS — Prefix Caching 工作正常!")
        ok = True
    else:
        print(f"\n  ❌ FAIL")
        ok = False

    clear_caches(model)
    return ok


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
