"""
Chunked Prefill + CUDA Graph 验证

核心测试:
  Case 1: 单条长 prompt 分 chunk, logits vs 非 chunked (同路径对比) — 正确性基石
  Case 3: kv_len 跨 step 正确 (mid-prefill 继续)
  Case 4: 混合 batch (decode + prefill chunk) 不炸
  Case 5: CUDA Graph decode 端到端 — 正常生成不报错
"""

import torch
from transformers import AutoTokenizer
from qwen3_naive import (
    Qwen3Config, Qwen3ForCausalLM, load_weights,
    Sequence, BlockManager, set_varlen_context, clear_varlen_context,
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

    # 4 GiB GPU 显存紧张，block 数减到 24 (KV cache ~700 MB)
    return model, tokenizer, config


def tokenize(text, tokenizer):
    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    ids = tokenizer.encode(formatted, return_tensors="pt").to("cuda")
    return ids


def clear_caches(model):
    for layer in model.model.layers:
        layer.self_attn.k_cache = None
        layer.self_attn.v_cache = None
    torch.cuda.empty_cache()


def make_long_prompt(target_tokens=3000):
    """生成 > 2048 tokens 的长 prompt, 确保触发 chunk"""
    base = ("请详细解释机器学习中的各种概念，包括监督学习、无监督学习和强化学习，"
            "以及深度学习中的卷积神经网络、循环神经网络和 Transformer 架构。"
            "请从数学原理、算法流程和实际应用三个层面展开论述。")
    # 估一下: base ~80 chars → ~40 tokens, 需要 ~75 次重复达到 3000
    return base * 75


def test_case1_single_chunked_vs_full(model, tokenizer, config):
    """Case 1: chunked vs 非 chunked — 同 forward_prefill 路径, 仅 chunk 大小不同

    使用中等长度 prompt (~512 tokens), 强制 MAX_TOKENS=256 切 chunk,
    避免在 4 GiB 卡上 OOM。
    """
    print(f"\n{'='*60}")
    print("Case 1: Chunked Prefill vs 非 Chunked 正确性 (同路径对比)")
    print(f"{'='*60}")

    # 构造 prompt: ~500 tokens, 够验证 chunk 正确性又不会 OOM
    long_text = ("请详细解释机器学习中的各种概念，包括监督学习、无监督学习和强化学习，"
                 "以及深度学习中的卷积神经网络、循环神经网络和 Transformer 架构。"
                 "请从数学原理、算法流程和实际应用三个层面展开论述。") * 12
    ids = tokenize(long_text, tokenizer)
    num_tokens = ids.shape[1]
    print(f"  prompt 长度: {num_tokens} tokens, 用 MAX_TOKENS=256 强制切 chunk")

    token_ids = ids[0].tolist()
    MAX_CHUNK = 256  # 故意设小, 强制多 chunk

    # === A: 非 Chunked (一次 prefill 完) ===
    clear_caches(model)
    model.allocate_global_kv_cache()
    bm = BlockManager(model._num_blocks, model.BLOCK_SIZE)

    seq_nc = Sequence(token_ids, {"temperature": 0.6, "max_tokens": 64})
    bm.allocate(seq_nc)

    seq_nc.num_scheduled_tokens = num_tokens - seq_nc.kv_len
    input_ids, pos, cu_q, cu_k, m_q, m_k, blk_map, bt = model._prepare_step([seq_nc])
    h_nc = model.model.forward_prefill(input_ids, pos, cu_q, cu_k, m_q, m_k, blk_map, bt)
    logits_nc = model.lm_head(h_nc)
    last_logits_nc = logits_nc[-1, :].float()
    seq_nc.kv_len += seq_nc.num_scheduled_tokens

    print(f"  [A] 非 chunked: prefill {num_tokens} tokens 一次完成")

    # === B: Chunked (复用同一 KV cache, 只清内容不重分配) ===
    # 通过 bm.deallocate 归还 seq_nc 的 blocks, 然后重用 cache
    bm.deallocate(seq_nc.block_table)
    bm2 = bm  # 复用同一个 BlockManager
    # 清空 cache 内容 (置零), 重置 free 列表
    bm2.free = list(range(model._num_blocks))
    bm2.ref_count = [0] * model._num_blocks
    bm2.hash_to_block.clear()
    bm2.block_hash = [-1] * model._num_blocks
    bm2.block_tokens = [None] * model._num_blocks

    seq_c = Sequence(token_ids, {"temperature": 0.6, "max_tokens": 64})
    bm2.allocate(seq_c)

    kv_history = []
    while seq_c.kv_len < seq_c.num_prompt_tokens:
        remaining = seq_c.num_prompt_tokens - seq_c.kv_len
        seq_c.num_scheduled_tokens = min(remaining, MAX_CHUNK)
        kv_before = seq_c.kv_len

        input_ids, pos, cu_q, cu_k, m_q, m_k, blk_map, bt = model._prepare_step([seq_c])
        h_c = model.model.forward_prefill(input_ids, pos, cu_q, cu_k, m_q, m_k, blk_map, bt)
        logits_c = model.lm_head(h_c)

        seq_c.kv_len += seq_c.num_scheduled_tokens
        bm2.hash_blocks(seq_c)
        kv_history.append((kv_before, seq_c.kv_len))
        print(f"  [B] Chunk {len(kv_history)}: {kv_before} → {seq_c.kv_len} "
              f"(scheduled={seq_c.num_scheduled_tokens})")
        seq_c.num_scheduled_tokens = 0

    last_logits_c = logits_c[-1, :].float()

    # 对比
    top1_nc = torch.topk(last_logits_nc, 1).indices[0].item()
    top1_c = torch.topk(last_logits_c, 1).indices[0].item()
    top1_match = top1_nc == top1_c

    k_val = 20
    _, nc_topk = torch.topk(last_logits_nc, k_val)
    _, c_topk = torch.topk(last_logits_c, k_val)
    overlap = len(set(nc_topk.tolist()) & set(c_topk.tolist()))

    l2_rel = (torch.norm(last_logits_nc - last_logits_c, p=2)
              / torch.norm(last_logits_nc, p=2)).item()

    print(f"\n  结果:")
    print(f"  top-1: non-chunked={top1_nc}, chunked={top1_c}, match={top1_match}")
    print(f"    → '{tokenizer.decode([top1_nc])}'")
    print(f"  top-20 overlap: {overlap}/20")
    print(f"  L2 相对误差: {l2_rel:.6f}")

    passed = top1_match and overlap >= 19
    print(f"  {'✅ PASS' if passed else '❌ FAIL'} — Case 1")
    return passed, kv_history


def test_case3_kvlen_tracking(model, tokenizer, config):
    """Case 3: kv_len 跨 step 正确递增"""
    print(f"\n{'='*60}")
    print("Case 3: kv_len 跨 Step 追踪")
    print(f"{'='*60}")

    clear_caches(model)
    model.allocate_global_kv_cache()
    bm = BlockManager(model._num_blocks, model.BLOCK_SIZE)

    # 使用较短 prompt (~1500 tokens) + 小 chunk(512), 适配 4 GiB 卡
    long_text = ("请详细解释机器学习中的各种概念，包括监督学习、无监督学习和强化学习，"
                 "以及深度学习中的卷积神经网络、循环神经网络和 Transformer 架构。"
                 "请从数学原理、算法流程和实际应用三个层面展开论述。") * 25
    ids = tokenize(long_text, tokenizer)
    num_tokens = ids.shape[1]
    token_ids = ids[0].tolist()
    print(f"  prompt: {num_tokens} tokens")

    if num_tokens <= 1024:
        print(f"  ⚠️ prompt 不够长, 跳过")
        return True

    seq = Sequence(token_ids, {"temperature": 0.6, "max_tokens": 64})
    bm.allocate(seq)

    MAX_TOKENS = 512   # 4 GiB 卡上保守, 确保 3 个 chunk
    kv_history = []

    while seq.kv_len < seq.num_prompt_tokens:
        remaining = seq.num_prompt_tokens - seq.kv_len
        seq.num_scheduled_tokens = min(remaining, MAX_TOKENS)
        kv_before = seq.kv_len

        input_ids, pos, cu_q, cu_k, m_q, m_k, blk_map, bt = model._prepare_step([seq])
        h = model.model.forward_prefill(input_ids, pos, cu_q, cu_k, m_q, m_k, blk_map, bt)

        seq.kv_len += seq.num_scheduled_tokens
        bm.hash_blocks(seq)

        kv_history.append((kv_before, seq.kv_len, seq.num_scheduled_tokens))

        # 验证 cu_q 和 cu_k
        assert cu_q[-1].item() == seq.num_scheduled_tokens, \
            f"cu_q mismatch: {cu_q[-1].item()} != {seq.num_scheduled_tokens}"
        assert cu_k[-1].item() == kv_before + seq.num_scheduled_tokens, \
            f"cu_k mismatch: {cu_k[-1].item()} != {kv_before + seq.num_scheduled_tokens}"

        print(f"  Chunk {len(kv_history)}: {kv_before} → {seq.kv_len} "
              f"(scheduled={seq.num_scheduled_tokens}), cu_q={cu_q[-1].item()}, cu_k={cu_k[-1].item()} ✓")
        seq.num_scheduled_tokens = 0

    # 验证
    final_ok = seq.kv_len == seq.num_prompt_tokens
    print(f"\n  kv_len={seq.kv_len}, prompt={seq.num_prompt_tokens}, match={final_ok}")

    increasing = all(h[0] < h[1] for h in kv_history)
    print(f"  逐 step 递增: {increasing}")

    # 验证无跳变: 每次 increment = scheduled
    no_jump = all(
        (h[1] - h[0]) == h[2] for h in kv_history
    )
    print(f"  增量 = scheduled: {no_jump}")

    passed = final_ok and increasing and no_jump
    print(f"  {'✅ PASS' if passed else '❌ FAIL'} — Case 3")
    return passed


def test_case4_mixed_batch(model, tokenizer, config):
    """Case 4: 混合 batch (decode + prefill chunk) 不炸"""
    print(f"\n{'='*60}")
    print("Case 4: 混合 Batch — decode + prefill chunk 同框")
    print(f"{'='*60}")

    clear_caches(model)
    model.allocate_global_kv_cache()
    bm = BlockManager(model._num_blocks, model.BLOCK_SIZE)

    # 2 条短 prompt, 1 条长 prompt
    short1_text = "你好。"
    short2_text = "什么是人工智能？"

    # Case 4: 中等长度 (~1200 tokens), 512 chunk, 适配 4 GiB 卡
    long_text = ("请全面分析现代机器学习的发展趋势，包括大语言模型、多模态学习、"
                 "强化学习在机器人中的应用、以及AI安全等方向。"
                 "请从技术原理、产业应用和学术研究三个维度展开讨论。") * 12
    ids_long = tokenize(long_text, tokenizer)
    print(f"  short1/2 很短, long: {ids_long.shape[1]} tokens")

    # 创建 seqs
    seq1 = Sequence(tokenize(short1_text, tokenizer)[0].tolist(), {"temperature": 0.6, "max_tokens": 16})
    seq2 = Sequence(tokenize(short2_text, tokenizer)[0].tolist(), {"temperature": 0.6, "max_tokens": 16})
    seq_long = Sequence(ids_long[0].tolist(), {"temperature": 0.6, "max_tokens": 16})

    # 给 seq1, seq2 prefill 全量 prompt
    for seq in [seq1, seq2]:
        bm.allocate(seq)
        seq.status = "RUNNING"
        seq.num_scheduled_tokens = seq.num_prompt_tokens - seq.kv_len

    input_ids, pos, cu_q, cu_k, m_q, m_k, blk_map, bt = model._prepare_step([seq1, seq2])
    h = model.model.forward_prefill(input_ids, pos, cu_q, cu_k, m_q, m_k, blk_map, bt)

    # 采样第一个 token (prefill 完成后必须有这一步, 否则 token_ids 长度不足)
    logits_prefill = model.lm_head(h)
    for i, seq in enumerate([seq1, seq2]):
        last_idx = cu_q[i + 1].item() - 1
        top1 = torch.argmax(logits_prefill[last_idx]).item()
        seq.append_token(top1)                              # 现在 token_ids 有 prompt+1 个
        seq.kv_len = seq.num_prompt_tokens
        bm.hash_blocks(seq)
        seq.num_scheduled_tokens = 0

    print(f"  seq1,seq2 prefill + 首token: kv_len={seq1.kv_len},{seq2.kv_len}, "
          f"num_tokens={seq1.num_tokens},{seq2.num_tokens}")

    # 同时拉进 seq_long 的第一个 chunk, 形成混合 batch
    bm.allocate(seq_long)
    seq_long.status = "RUNNING"

    for seq in [seq1, seq2]:
        seq.num_scheduled_tokens = 1   # decode: kv_len 已经指向下一个要算的位置
    seq_long.num_scheduled_tokens = min(512, seq_long.num_prompt_tokens - seq_long.kv_len)

    mixed_seqs = [seq1, seq2, seq_long]
    input_ids, pos, cu_q, cu_k, m_q, m_k, blk_map, bt = model._prepare_step(mixed_seqs)
    h = model.model.forward_prefill(input_ids, pos, cu_q, cu_k, m_q, m_k, blk_map, bt)

    for seq in mixed_seqs:
        seq.kv_len += seq.num_scheduled_tokens
        bm.hash_blocks(seq)

    # 验证 cu_q/cu_k (prefill 后的 kv_len = prompt, decode 在此之上 +1)
    expected_cu_q = [0, 1, 2, 2 + seq_long.num_scheduled_tokens]
    expected_cu_k = [0,
                     seq1.kv_len,                          # prompt1 + 1 decode
                     seq1.kv_len + seq2.kv_len,            # + prompt2 + 1 decode
                     seq1.kv_len + seq2.kv_len + seq_long.kv_len]

    cu_q_ok = cu_q.tolist() == expected_cu_q
    cu_k_ok = cu_k.tolist() == expected_cu_k

    max_blocks = max(len(s.block_table) for s in mixed_seqs)
    bt_ok = bt.shape == (3, max_blocks)

    print(f"\n  cu_q: {cu_q.tolist()}, expected: {expected_cu_q}, ok: {cu_q_ok}")
    print(f"  cu_k: {cu_k.tolist()}, expected: {expected_cu_k}, ok: {cu_k_ok}")
    print(f"  block_tables: {bt.shape}, ok: {bt_ok}")

    # 验证 seq_long 是 mid-prefill
    is_mid = seq_long.kv_len < seq_long.num_prompt_tokens
    print(f"  seq_long mid-prefill: {is_mid} (kv_len={seq_long.kv_len} < prompt={seq_long.num_prompt_tokens})")

    # 能继续 prefill 剩余部分
    if is_mid:
        remaining = seq_long.num_prompt_tokens - seq_long.kv_len
        seq_long.num_scheduled_tokens = min(remaining, 512)

        input_ids2, pos2, cu_q2, cu_k2, _, _, blk_map2, bt2 = model._prepare_step([seq_long])
        h2 = model.model.forward_prefill(input_ids2, pos2, cu_q2, cu_k2, m_q, m_k, blk_map2, bt2)
        seq_long.kv_len += seq_long.num_scheduled_tokens
        print(f"  继续 prefill → kv_len={seq_long.kv_len}, 完成={seq_long.kv_len >= seq_long.num_prompt_tokens}")
        print(f"  cu_q(step2): {cu_q2.tolist()}, cu_k(step2): {cu_k2.tolist()}")

    seq_long.num_scheduled_tokens = 0
    for seq in [seq1, seq2]:
        seq.num_scheduled_tokens = 0

    all_ok = cu_q_ok and cu_k_ok and bt_ok
    print(f"  {'✅ PASS' if all_ok else '❌ FAIL'} — Case 4")
    return all_ok


def test_case5_cuda_graph(model, tokenizer, config):
    """Case 5: CUDA Graph decode 端到端 — 正常生成不报错"""
    print(f"\n{'='*60}")
    print("Case 5: CUDA Graph — decode 端到端")
    print(f"{'='*60}")

    clear_caches(model)
    torch.cuda.empty_cache()

    # 2 条短 prompt, 确保大部分 step 走 decode
    prompts = ["你好，请介绍一下你自己。", "今天天气怎么样？"]
    prompt_ids_list = []
    for text in prompts:
        messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer.encode(formatted, return_tensors="pt").to("cuda")
        prompt_ids_list.append(ids)
        print(f"  prompt: {ids.shape[1]} tokens")

    # 用 generate_continuous, max_new_tokens 设小一点
    torch.manual_seed(42)
    seq_tokens, all_generated = model.generate_continuous(
        prompt_ids_list,
        max_new_tokens=16,
        temperature=0.6,
        top_k=20,
        eos_token_ids=[config.eos_token_id],
    )

    has_graph = getattr(model, '_has_cuda_graphs', False)
    print(f"\n  CUDA Graph 已捕获: {has_graph}")
    print(f"  生成 token 数: {[len(g) for g in all_generated]}")

    # 只要不报错就跑通了
    total_generated = sum(len(g) for g in all_generated)
    passed = total_generated > 0
    print(f"  {'✅ PASS' if passed else '❌ FAIL'} — Case 5")
    return passed


def run_case(case_num):
    """跑单个 case, 完成后彻底清理显存"""
    model, tokenizer, config = load_model_and_tokenizer()

    try:
        if case_num == 1:
            ok, _ = test_case1_single_chunked_vs_full(model, tokenizer, config)
        elif case_num == 3:
            ok = test_case3_kvlen_tracking(model, tokenizer, config)
        elif case_num == 4:
            ok = test_case4_mixed_batch(model, tokenizer, config)
        elif case_num == 5:
            ok = test_case5_cuda_graph(model, tokenizer, config)
        else:
            print(f"未知 case: {case_num}")
            return False

        print(f"\n  Case {case_num}: {'✅ PASS' if ok else '❌ FAIL'}")
        return ok
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        clear_caches(model)
        # 强制释放 GPU 内存
        del model
        torch.cuda.empty_cache()
        import gc; gc.collect()


def main():
    import sys
    case_arg = None
    for i, a in enumerate(sys.argv):
        if a == "--case" and i + 1 < len(sys.argv):
            case_arg = int(sys.argv[i + 1])

    if case_arg:
        ok = run_case(case_arg)
        exit(0 if ok else 1)
    else:
        # 全部跑: 每个 case 独立进程, 干净显存
        import subprocess, os
        results = {}
        for cn in [1, 3, 4, 5]:
            print(f"\n{'#'*60}")
            print(f"# 启动独立进程跑 Case {cn}")
            print(f"{'#'*60}")
            r = subprocess.run(
                ["python", __file__, "--case", str(cn)],
                capture_output=False,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            results[f"Case {cn}"] = r.returncode == 0

        print(f"\n{'='*60}")
        print("总结")
        print(f"{'='*60}")
        for case, ok in results.items():
            print(f"  {case}: {'✅' if ok else '❌'}")
        all_pass = all(results.values())
        print(f"  {'全部通过 ✅' if all_pass else '有失败 ❌'}")
        exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
