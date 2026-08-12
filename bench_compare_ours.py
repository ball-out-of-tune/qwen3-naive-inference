"""Benchmark our qwen3_naive.py on Qwen3-0.6B — same methodology as bench.py/nano-vllm.

Uses generate_naive() (sequential, no KV cache). Each sequence is generated one at a time.
"""
import os, sys, time
from random import randint, seed

sys.path.insert(0, '/workspace/dist_infer')
import torch
from qwen3_naive import Qwen3Config, Qwen3ForCausalLM, load_weights

NUM_RUNS = 5
NUM_SEQS = 48
MAX_INPUT_LEN = 128
MAX_OUTPUT_LEN = 64

MODEL_PATH = "/workspace/models/Qwen3-0.6B"


def main():
    seed(0)
    torch.manual_seed(0)

    print(f"Engine:     qwen3_naive.py (generate_naive, sequential)")
    print(f"Model:      {MODEL_PATH}")
    print(f"Config:     {NUM_SEQS} seqs, input=50~{MAX_INPUT_LEN}, output=50~{MAX_OUTPUT_LEN}")
    print(f"Runs:       {NUM_RUNS} (after warmup)")
    print()

    # Load model
    print("Loading model...")
    config = Qwen3Config.from_json(f"{MODEL_PATH}/config.json")
    model = Qwen3ForCausalLM(config)
    load_weights(model, f"{MODEL_PATH}/model.safetensors")

    params = sum(p.numel() for p in model.parameters())
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"  Params: {params:,}, VRAM: {vram:.2f} GB")

    # Generate random prompts
    prompt_ids_list = [
        torch.randint(0, config.vocab_size, (1, randint(50, MAX_INPUT_LEN)), device='cuda')
        for _ in range(NUM_SEQS)
    ]
    output_lens = [randint(50, MAX_OUTPUT_LEN) for _ in range(NUM_SEQS)]
    total_tokens = sum(output_lens)

    # Warmup
    print("Warmup...")
    warmup = torch.randint(0, config.vocab_size, (1, 4), device='cuda')
    model.generate_naive(warmup, max_new_tokens=4, temperature=0.6, top_k=20)
    torch.cuda.synchronize()
    print("Warmup done.\n")

    # Benchmark runs
    times = []
    for run in range(NUM_RUNS):
        torch.cuda.synchronize()
        t = time.time()

        for prompt_ids, out_len in zip(prompt_ids_list, output_lens):
            model.generate_naive(
                prompt_ids,
                max_new_tokens=out_len,
                temperature=0.6,
                top_k=20,
                eos_token_ids=None,
            )

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
    print(f"  qwen3_naive.py Results ({NUM_RUNS} runs):")
    print(f"  GPU:          {torch.cuda.get_device_name(0)}")
    print(f"  Sequences:    {NUM_SEQS}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Avg time:     {avg_time:.2f}s")
    print(f"  Throughput:   {avg_tps:.2f} tok/s  (range: {min_tps:.1f} ~ {max_tps:.1f})")
    print(f"  VRAM:         {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
