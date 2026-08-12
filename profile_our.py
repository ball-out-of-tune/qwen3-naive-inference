"""Profile our inference engine — find performance bottlenecks.

Two layers:
  1. Phase-level timing: CUDA events on key methods → per-phase ms breakdown
  2. Kernel-level trace: PyTorch Profiler → chrome trace for visualization

Usage:
    # Phase-level timing (fast):
    python profile_our.py --num-seqs 8 --input-len 256 --output-len 64

    # Kernel-level trace (exports chrome_trace.json):
    python profile_our.py --num-seqs 8 --input-len 256 --output-len 64 --trace

    # Longer run for stable stats:
    python profile_our.py --num-seqs 16 --input-len 512 --output-len 128
"""
import argparse, os, sys, time
from collections import defaultdict
from random import randint, seed as py_seed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen3_naive import Qwen3Config, Qwen3ForCausalLM, load_weights
import torch
from transformers import AutoTokenizer


# ═══════════════════════════════════════════════════════════════════
# CUDA Event Timer — wraps methods with precise GPU timing
# ═══════════════════════════════════════════════════════════════════

class PhaseTimer:
    """Patch methods with CUDA event timing, collect stats per phase."""

    def __init__(self):
        self.times = defaultdict(list)  # phase_name → [elapsed_ms, ...]

    def wrap(self, obj, method_name, phase_name):
        """Replace obj.method_name with a timed version that records to phase_name."""
        original = getattr(obj, method_name)

        def timed(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            # NOTE: don't synchronize here — that would serialize the GPU pipeline.
            # Instead, record the event pair and synchronize later.
            timed._pending.append((phase_name, start, end))
            return result

        timed._pending = []  # shared list for all wraps on the same object
        timed._original = original
        setattr(obj, method_name, timed)
        return timed

    def collect(self, *wrapped_methods):
        """Sync all pending events and collect times."""
        for wm in wrapped_methods:
            for phase_name, start, end in wm._pending:
                end.synchronize()
                self.times[phase_name].append(start.elapsed_time(end))
            wm._pending.clear()

    def unwrap(self, obj, method_name, wrapped):
        """Restore original method."""
        setattr(obj, method_name, wrapped._original)

    def report(self):
        """Print per-phase timing summary."""
        print("\n" + "=" * 65)
        print("  Phase-level Timing Summary")
        print("=" * 65)
        total_ms = sum(sum(ts) for ts in self.times.values())
        for name in sorted(self.times.keys()):
            ts = self.times[name]
            avg = sum(ts) / len(ts)
            pct = sum(ts) / total_ms * 100 if total_ms > 0 else 0
            print(f"  {name:<30s}  avg={avg:7.2f}ms  total={sum(ts):8.1f}ms  "
                  f"({pct:5.1f}%)  n={len(ts)}")
        print(f"  {'─' * 55}")
        print(f"  {'Total':<30s}  {total_ms:>25.1f}ms")
        print("=" * 65)


# ═══════════════════════════════════════════════════════════════════
# PyTorch Profiler trace (kernel-level)
# ═══════════════════════════════════════════════════════════════════

def run_trace(model, prompt_ids_list, output_len, trace_path="chrome_trace.json",
              temperature=0.6, top_k=20):
    """Run generate with PyTorch Profiler, export chrome trace."""
    print(f"\nRunning PyTorch Profiler trace (output_len={output_len})...")
    print(f"Trace will be saved to: {trace_path}")

    # We profile a short generate run — the profiler records ALL CUDA kernels
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,          # stack traces add overhead; omit for cleaner trace
        with_modules=False,
    ) as prof:
        _ = model.generate_continuous(
            prompt_ids_list,
            max_new_tokens=output_len,
            temperature=temperature,
            top_k=top_k,
            eos_token_ids=None,
        )

    # Export
    prof.export_chrome_trace(trace_path)
    print(f"Trace exported to {trace_path}")

    # Print top kernels by CUDA time
    print("\n" + "=" * 80)
    print("  Top CUDA Kernels by Time (key_averages)")
    print("=" * 80)
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=30)
    print(table)
    print("=" * 80)

    # Also print a grouped view
    print("\n" + "=" * 80)
    print("  Top CUDA Kernels by Self Time")
    print("=" * 80)
    table_self = prof.key_averages().table(sort_by="cuda_time_self", row_limit=30)
    print(table_self)
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--trace", action="store_true",
                        help="Also export PyTorch Profiler chrome trace")
    parser.add_argument("--trace-only", action="store_true",
                        help="Only run PyTorch Profiler (skip phase timing)")
    args = parser.parse_args()

    # Auto-detect model path
    if args.model is None:
        for candidate in [
            os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
            "/mnt/c/Users/16874/huggingface/Qwen3-0.6B/",
            "C:/Users/16874/huggingface/Qwen3-0.6B/",
            "/root/huggingface/Qwen3-0.6B/",
        ]:
            if os.path.isfile(os.path.join(candidate, "config.json")):
                args.model = candidate
                break
    if args.model is None:
        print("ERROR: Cannot find model. Use --model to specify path.")
        sys.exit(1)

    # ── Generate prompts (seed=0, same as bench_our.py) ──
    py_seed(0)
    prompt_ids_list = []
    for _ in range(args.num_seqs):
        n = randint(100, args.input_len)
        ids = torch.tensor([[randint(0, 10000) for _ in range(n)]], device='cuda')
        prompt_ids_list.append(ids)

    total_prompt_tokens = sum(p.shape[1] for p in prompt_ids_list)

    print(f"Model: {args.model}")
    print(f"Sequences: {args.num_seqs}")
    print(f"Input len: {args.input_len}, Output len: {args.output_len}")
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print()

    # ── Load model ──
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = Qwen3Config.from_json(f"{args.model}/config.json")
    model = Qwen3ForCausalLM(config)
    load_weights(model, f"{args.model}/model.safetensors")
    model.to('cuda', dtype=torch.bfloat16)
    print(f"  Model dtype after .to(cuda, bf16): {next(model.parameters()).dtype}")
    torch.cuda.empty_cache()
    torch.set_default_device('cpu')

    # Pre-warmup: 触发 torch.compile JIT (prefill + decode 路径都要)
    # 用代表性大小的输入, 避免编译发生在 benchmark 计时里
    print("  Triggering torch.compile warmup...")
    dummy_x = torch.randn(256, 1024, dtype=torch.bfloat16, device='cuda')
    dummy_res = torch.randn(256, 1024, dtype=torch.bfloat16, device='cuda')
    for i, layer in enumerate(model.model.layers):
        _ = layer.input_layernorm(dummy_x)                    # _rms_norm
        _ = layer.input_layernorm(dummy_x, dummy_res)         # _add_rms_norm
        _ = layer.post_attention_layernorm(dummy_x)
        _ = layer.post_attention_layernorm(dummy_x, dummy_res)
    torch.cuda.synchronize()
    print("  torch.compile warmup done.")

    # Warmup (also triggers CUDA Graph capture)
    warmup_ids = [torch.randint(0, 10000, (1, 4), device='cuda')]
    _ = model.generate_continuous(warmup_ids, max_new_tokens=4, temperature=0.6,
                                   eos_token_ids=None)
    torch.cuda.synchronize()
    init_time = time.time() - t0
    print(f"Init + warmup time: {init_time:.1f}s")
    print(f"KV blocks: {model._num_blocks}")
    print()

    # ═══════════════════════════════════════════════════════════
    # Trace-only mode: PyTorch Profiler around entire generate
    # ═══════════════════════════════════════════════════════════
    if args.trace_only:
        run_trace(model, prompt_ids_list, args.output_len,
                  trace_path="chrome_trace.json",
                  temperature=0.6, top_k=20)
        return

    # ═══════════════════════════════════════════════════════════
    # Phase-level timing: wrap key methods with CUDA events
    # ═══════════════════════════════════════════════════════════
    timer = PhaseTimer()

    # Wrap top-level methods on the model
    w_prefill = timer.wrap(model.model, "forward_prefill", "model.forward_prefill")
    w_decode = timer.wrap(model.model, "forward_decode", "model.forward_decode")
    w_run_graph = timer.wrap(model, "_run_decode_graph", "_run_decode_graph")  # CUDA Graph replay
    w_lm_head = timer.wrap(model.lm_head, "forward", "lm_head")   # lm_head 是 nn.Linear 子模块, 必须 wrap .forward
    w_prepare_step = timer.wrap(model, "_prepare_step", "_prepare_step")
    w_prepare_decode = timer.wrap(model, "_prepare_decode", "_prepare_decode")

    # Sampling time = total_step_time - (prepare + forward + lm_head).
    # We'll measure total per-step time separately.

    all_wraps = [w_prefill, w_decode, w_run_graph, w_lm_head, w_prepare_step, w_prepare_decode]

    print(f"Running phase-level profile ({args.num_seqs} seqs, "
          f"output_len={args.output_len})...")
    torch.cuda.synchronize()
    t0 = time.time()

    _ = model.generate_continuous(
        prompt_ids_list,
        max_new_tokens=args.output_len,
        temperature=0.6,
        top_k=20,
        eos_token_ids=None,
    )

    torch.cuda.synchronize()
    total_elapsed = time.time() - t0

    # Collect all pending CUDA events
    timer.collect(*all_wraps)

    # Unwrap
    timer.unwrap(model.model, "forward_prefill", w_prefill)
    timer.unwrap(model.model, "forward_decode", w_decode)
    timer.unwrap(model, "_run_decode_graph", w_run_graph)
    timer.unwrap(model.lm_head, "forward", w_lm_head)
    timer.unwrap(model, "_prepare_step", w_prepare_step)
    timer.unwrap(model, "_prepare_decode", w_prepare_decode)

    # ═══════════════════════════════════════════════════════════
    # Report
    # ═══════════════════════════════════════════════════════════
    total_output_tokens = args.num_seqs * args.output_len
    throughput = total_output_tokens / total_elapsed

    print(f"\n{'='*65}")
    print(f"  Overall")
    print(f"{'='*65}")
    print(f"  Wall-clock:          {total_elapsed:.2f}s")
    print(f"  Throughput:          {throughput:.1f} tok/s")
    print(f"  Output tokens:       {total_output_tokens}")

    timer.report()

    # ── Deeper breakdown ──
    # Count prefill vs decode steps
    n_prefill = len(timer.times.get("model.forward_prefill", []))
    n_decode = len(timer.times.get("model.forward_decode", []))
    print(f"\n  Step counts:  prefill={n_prefill}, decode={n_decode}")
    if n_prefill + n_decode > 0:
        print(f"  Avg steps / sec: {(n_prefill + n_decode) / total_elapsed:.1f}")

    # ═══════════════════════════════════════════════════════════
    # Optional: PyTorch Profiler trace
    # ═══════════════════════════════════════════════════════════
    if args.trace:
        # Re-create model for clean trace (CUDA Graph state is dirty after first run)
        # Actually, reuse the same model — the trace just shows kernel calls.
        # But we need fresh prompt_ids since the first run consumed them.
        py_seed(1)
        prompt_ids_list2 = []
        for _ in range(args.num_seqs):
            n = randint(100, args.input_len)
            ids = torch.tensor([[randint(0, 10000) for _ in range(n)]], device='cuda')
            prompt_ids_list2.append(ids)

        run_trace(model, prompt_ids_list2,
                  min(args.output_len, 32),  # short trace to keep file manageable
                  trace_path="chrome_trace.json",
                  temperature=0.6, top_k=20)


if __name__ == "__main__":
    main()
