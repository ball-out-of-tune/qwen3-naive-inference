"""Benchmark nano-vllm on Qwen3-0.6B — matches bench.py methodology."""
import os, sys, time
from random import randint, seed

sys.path.insert(0, '/workspace/dist_infer')
import torch
from nanovllm import LLM, SamplingParams

NUM_RUNS = 5
NUM_SEQS = 48
MAX_INPUT_LEN = 128
MAX_OUTPUT_LEN = 64

MODEL_PATH = "/workspace/models/Qwen3-0.6B"


def main():
    seed(0)
    torch.manual_seed(0)

    print(f"Engine:     nano-vllm")
    print(f"Model:      {MODEL_PATH}")
    print(f"Config:     {NUM_SEQS} seqs, input=50~{MAX_INPUT_LEN}, output=50~{MAX_OUTPUT_LEN}")
    print(f"Runs:       {NUM_RUNS} (after warmup)")
    print()

    print("Loading model...")
    llm = LLM(
        MODEL_PATH,
        enforce_eager=True,
        max_model_len=256,
        max_num_batched_tokens=8192,
        max_num_seqs=64,
        gpu_memory_utilization=0.85,
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(50, MAX_INPUT_LEN))]
        for _ in range(NUM_SEQS)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(50, MAX_OUTPUT_LEN))
        for _ in range(NUM_SEQS)
    ]
    total_tokens = sum(sp.max_tokens for sp in sampling_params)

    # Warmup (discarded)
    print("Warmup...")
    llm.generate(["Benchmark warmup"], SamplingParams())
    torch.cuda.synchronize()
    print("Warmup done.\n")

    # Benchmark runs
    times = []
    for run in range(NUM_RUNS):
        t = time.time()
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.time() - t
        tps = total_tokens / elapsed
        times.append(elapsed)
        print(f"  Run {run+1}/{NUM_RUNS}: {elapsed:.2f}s -> {tps:.2f} tok/s")

    avg_time = sum(times) / len(times)
    min_tps = total_tokens / max(times)
    max_tps = total_tokens / min(times)
    avg_tps = total_tokens / avg_time

    print(f"\n{'='*60}")
    print(f"  nano-vllm Results ({NUM_RUNS} runs):")
    print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print(f"  Sequences:    {NUM_SEQS}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Avg time:     {avg_time:.2f}s")
    print(f"  Throughput:   {avg_tps:.2f} tok/s  (range: {min_tps:.1f} ~ {max_tps:.1f})")
    print(f"  VRAM:         {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"{'='*60}")

    llm.exit()


if __name__ == "__main__":
    main()
