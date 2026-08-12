"""Profile nano-vllm — find where GPU time goes, compare with our engine.

Usage:
    # WSL2:
    python profile_nano.py --num-seqs 8 --input-len 256 --output-len 64

    # Server (with --nano-path):
    python profile_nano.py --num-seqs 8 --input-len 256 --output-len 64 \
        --nano-path /root/bench/nano-vllm
"""
import argparse, os, sys, time
from collections import defaultdict
from random import randint, seed as py_seed

import torch

# ── Auto-detect nano-vllm path ──
def _detect_nano_path():
    candidates = [
        "/root/bench/nano-vllm",
        "/mnt/c/Users/16874/AIInfraGuide/nano-vllm",
        "C:/Users/16874/AIInfraGuide/nano-vllm",
        os.path.expanduser("~/AIInfraGuide/nano-vllm"),
    ]
    for p in candidates:
        init_file = os.path.join(p, "nanovllm", "__init__.py")
        if os.path.isfile(init_file):
            return os.path.abspath(p)
    return None

NANO_PATH = _detect_nano_path()
if NANO_PATH is None:
    print("ERROR: Cannot find nano-vllm. Use --nano-path to specify.")
    sys.exit(1)

sys.path.insert(0, NANO_PATH)


# ═══════════════════════════════════════════════════════════════════
# CUDA Event Timer
# ═══════════════════════════════════════════════════════════════════

class PhaseTimer:
    """Wrap methods with CUDA event timing."""

    def __init__(self):
        self.times = defaultdict(list)

    def wrap(self, obj, method_name, phase_name):
        original = getattr(obj, method_name)

        def timed(*args, **kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original(*args, **kwargs)
            end.record()
            timed._pending.append((phase_name, start, end))
            return result

        timed._pending = []
        timed._original = original
        setattr(obj, method_name, timed)
        return timed

    def reset(self):
        self.times.clear()

    def collect(self, *wrapped_methods):
        for wm in wrapped_methods:
            for phase_name, start, end in wm._pending:
                end.synchronize()
                self.times[phase_name].append(start.elapsed_time(end))
            wm._pending.clear()

    def unwrap(self, obj, method_name, wrapped):
        setattr(obj, method_name, wrapped._original)

    def report(self):
        print("\n" + "=" * 70)
        print("  nano-vllm GPU Phase Timing")
        print("=" * 70)
        total_ms = sum(sum(ts) for ts in self.times.values())
        for name in sorted(self.times.keys()):
            ts = self.times[name]
            avg = sum(ts) / len(ts)
            pct = sum(ts) / total_ms * 100 if total_ms > 0 else 0
            print(f"  {name:<35s}  avg={avg:7.2f}ms  total={sum(ts):8.1f}ms  "
                  f"({pct:5.1f}%)  n={len(ts)}")
        print(f"  {'─' * 60}")
        print(f"  {'Total (GPU phases)':<35s}  {total_ms:>25.1f}ms")
        print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# We can't easily patch inside ModelRunner.run() because it already
# uses CUDA events. Instead, we'll:
#   1. Wrap model.forward (model body)
#   2. Wrap model.compute_logits (lm_head)
#   3. Wrap prepare_prefill / prepare_decode (CPU prep)
#   4. Wrap sampler.forward (sampling)
#   5. Wrap run_model (total GPU forward)
# ═══════════════════════════════════════════════════════════════════

def profile_nano(args):
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.model_runner import ModelRunner

    # ── Generate prompts (seed=0, same as bench_our.py) ──
    py_seed(0)
    prompts = []
    total_prompt_tokens = 0
    for _ in range(args.num_seqs):
        n = randint(100, args.input_len)
        ids = [randint(0, 10000) for _ in range(n)]
        prompts.append(ids)
        total_prompt_tokens += n

    print(f"Nano-vllm path: {NANO_PATH}")
    print(f"Model: (auto-detect)")
    print(f"Sequences: {args.num_seqs}")
    print(f"Input len: {args.input_len}, Output len: {args.output_len}")
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print()

    # ── Init ──
    t0 = time.time()
    llm = LLM(
        model=args.model or _detect_model(),
        enforce_eager=False,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=2048,
        max_num_seqs=256,
        max_model_len=args.input_len + args.output_len + 256,
    )
    init_time = time.time() - t0

    # Access the internal ModelRunner
    mr = llm.model_runner
    model = mr.model

    print(f"Init time: {init_time:.1f}s")
    print(f"KV blocks: {mr.config.num_kvcache_blocks}")
    print()

    # ── Wrap key methods ──
    timer = PhaseTimer()

    w_prepare_prefill = timer.wrap(mr, "prepare_prefill", "prepare_prefill")
    w_prepare_decode = timer.wrap(mr, "prepare_decode", "prepare_decode")
    w_model = timer.wrap(model, "forward", "model.forward")
    w_logits = timer.wrap(model, "compute_logits", "compute_logits")
    w_sampler = timer.wrap(mr.sampler, "forward", "sampler.forward")
    w_run_model = timer.wrap(mr, "run_model", "run_model")

    all_wraps = [w_prepare_prefill, w_prepare_decode, w_model, w_logits,
                 w_sampler, w_run_model]

    # ── Warmup ──
    warmup_prompts = [[randint(0, 10000) for _ in range(16)] for _ in range(4)]
    sp = SamplingParams(temperature=0.6, max_tokens=4, ignore_eos=True)
    _ = llm.generate(warmup_prompts, sp, use_tqdm=False)
    torch.cuda.synchronize()
    timer.collect(*all_wraps)  # discard warmup timing
    timer.reset()
    print("Warmup done.\n")

    # ── Profile run ──
    sp = SamplingParams(temperature=0.6, max_tokens=args.output_len, ignore_eos=True)

    print(f"Running profile ({args.num_seqs} seqs, output_len={args.output_len})...")
    torch.cuda.synchronize()
    t0 = time.time()

    _ = llm.generate(prompts, sp, use_tqdm=False)

    torch.cuda.synchronize()
    total_elapsed = time.time() - t0

    # Collect all pending CUDA events
    timer.collect(*all_wraps)

    # ── Unwrap ──
    timer.unwrap(mr, "prepare_prefill", w_prepare_prefill)
    timer.unwrap(mr, "prepare_decode", w_prepare_decode)
    timer.unwrap(model, "forward", w_model)
    timer.unwrap(model, "compute_logits", w_logits)
    timer.unwrap(mr.sampler, "forward", w_sampler)
    timer.unwrap(mr, "run_model", w_run_model)

    # ── Report ──
    total_output_tokens = args.num_seqs * args.output_len
    throughput = total_output_tokens / total_elapsed

    print(f"\n{'='*70}")
    print(f"  nano-vllm Overall")
    print(f"{'='*70}")
    print(f"  Wall-clock:          {total_elapsed:.2f}s")
    print(f"  Throughput:          {throughput:.1f} tok/s")
    print(f"  Output tokens:       {total_output_tokens}")

    timer.report()

    # Deeper breakdown
    n_prefill = len(timer.times.get("model.forward", []))
    n_decode = len(timer.times.get("run_model", []))
    print(f"\n  model.forward calls: {n_prefill}")
    print(f"  run_model calls: {n_decode}")
    print(f"  compute_logits calls: {len(timer.times.get('compute_logits', []))}")


def _detect_model():
    for candidate in [
        os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
        "/mnt/c/Users/16874/huggingface/Qwen3-0.6B/",
        "C:/Users/16874/huggingface/Qwen3-0.6B/",
        "/root/huggingface/Qwen3-0.6B/",
    ]:
        if os.path.isfile(os.path.join(candidate, "config.json")):
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seqs", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--nano-path", type=str, default=None)
    args = parser.parse_args()
    if args.nano_path:
        global NANO_PATH
        NANO_PATH = args.nano_path
        sys.path.insert(0, NANO_PATH)

    profile_nano(args)


if __name__ == "__main__":
    main()
