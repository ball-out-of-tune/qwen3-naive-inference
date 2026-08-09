"""
GRPO Training for Qwen3-0.6B
基于 qwen3_naive.py 的单卡 GRPO 训练循环。
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from qwen3_naive import (
    Qwen3Config,
    Qwen3ForCausalLM,
    load_weights,
    set_varlen_context,
    clear_varlen_context,
)


# ============================================================
# 配置常量
# ============================================================

MODEL_PATH = "/mnt/c/Users/16874/Downloads/Qwen3-0.6B"

# GRPO 超参数
N_ROLLOUT = 4           # 每个 prompt 采样几条 response
PPO_EPOCHS = 1          # 每批数据训练几轮
CLIP_EPSILON = 0.2      # PPO clip 范围
MAX_RESPONSE_LEN = 256  # 生成最大 token 数
TEMPERATURE = 0.8       # 采样温度 (RL 用高温度鼓励探索)
TOP_K = 20
LEARNING_RATE = 5e-6

# Token IDs
EOS_TOKEN_ID = 151645
BOS_TOKEN_ID = 151643


# ============================================================
# ╔═══════════════════════════════════════════════════════════╗
# ║            核心逻辑 1: forward_logprob                    ║
# ║  输入: input_ids [batch, seq]                            ║
# ║  输出: log P(token_t | token_{0:t-1})  [batch, seq]      ║
# ╚═══════════════════════════════════════════════════════════╝
# ============================================================

def forward_logprob(model, input_ids):
    """前向 → 返回每个位置 token 的 log probability"""
    logits = model(input_ids)                           # [batch, seq, vocab]
    log_probs = F.log_softmax(logits.float(), dim=-1)   # [batch, seq, vocab]
    token_log_probs = log_probs.gather(
        dim=-1, index=input_ids.unsqueeze(-1)
    ).squeeze(-1)                                        # [batch, seq]
    return token_log_probs

def forward_token_log_prob_my_own(model, input_ids):
    logits = model(input_ids)
    probs_log = F.log_softmax(logits.float(), dim=-1)
    token_log_probs = probs_log.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    return token_log_probs

# ============================================================
# ╔═══════════════════════════════════════════════════════════╗
# ║         核心逻辑 2: Rollout (生成 + 记录 log_prob)        ║
# ║  输入: prompt_ids [1, prompt_len]                        ║
# ║  输出: input_ids, response_mask, old_log_probs           ║
# ╚═══════════════════════════════════════════════════════════╝
# ============================================================

@torch.no_grad()
def rollout_single(model, prompt_ids, tokenizer):
    """
    对一条 prompt 生成 response，同时记录每个采样 token 的 log_prob。

    返回:
        input_ids:    [1, total_len]     完整序列 (prompt + response)
        old_log_probs: [total_len]       每个位置 token 的 logP
                                            (prompt 部分填 0)
        response_mask: [total_len]       1 = response token, 0 = prompt token
    """
    prompt_len = prompt_ids.shape[1]

    # ---- 预分配 ----
    max_len = prompt_len + MAX_RESPONSE_LEN
    input_ids = torch.full((1, max_len), 0, dtype=torch.long, device="cuda")
    input_ids[0, :prompt_len] = prompt_ids[0]

    old_log_probs = torch.zeros(max_len, device="cuda")
    response_mask = torch.zeros(max_len, device="cuda")

    # ---- 逐 token 生成 ----
    for step in range(MAX_RESPONSE_LEN):
        current_len = prompt_len + step
        pos = current_len - 1  # 生成序列中最后一个 token 的位置

        # 前向
        logits = model(input_ids[:, :current_len])      # [1, current_len, vocab]
        last_logits = logits[0, -1, :] / TEMPERATURE     # [vocab]

        # top-k 过滤
        if TOP_K is not None:
            topk_vals, topk_idx = torch.topk(last_logits, TOP_K)
            neg_inf = torch.tensor(float("-inf"), dtype=last_logits.dtype, device="cuda")
            filtered = torch.full_like(last_logits, neg_inf)
            last_logits = filtered.scatter(-1, topk_idx, topk_vals)

        # 采样
        probs = torch.softmax(last_logits.float(), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 记录选中 token 的 logP
        log_probs_all = torch.log_softmax(last_logits.float(), dim=-1)
        old_log = log_probs_all[next_token.item()]       # scalar

        # 写入
        input_ids[0, current_len] = next_token
        old_log_probs[current_len] = old_log
        response_mask[current_len] = 1.0

        # 遇到 EOS 停止
        if next_token.item() in (EOS_TOKEN_ID, BOS_TOKEN_ID):
            break

    # 裁掉未使用的尾部
    final_len = prompt_len + step + 1
    return (
        input_ids[:, :final_len],
        old_log_probs[:final_len],
        response_mask[:final_len],
    )


# ============================================================
# ╔═══════════════════════════════════════════════════════════╗
# ║          核心逻辑 3: Reward Function                      ║
# ║  输入: response 文本                                     ║
# ║  输出: 标量 reward                                       ║
# ║  TODO: 根据你的任务替换这个函数                            ║
# ╚═══════════════════════════════════════════════════════════╝
# ============================================================

def compute_reward(response_text):
    """
    示例 reward: 用长度做最简单的 reward (鼓励模型生成回复)。
    实际上应替换为: GSM8K答案匹配 / 代码测试通过 / RM模型打分 / 等等
    """
    # 简单示例: 有实质性内容就给正向奖励
    # 实际使用时替换为你的任务相关打分逻辑
    if len(response_text.strip()) > 5:
        return 1.0
    return 0.0


# ============================================================
# ╔═══════════════════════════════════════════════════════════╗
# ║     核心逻辑 4: GRPO 优势计算 (组内归一化)                ║
# ║  Group Relative Policy Optimization:                     ║
# ║  advantage = (reward - group_mean) / (group_std + eps)   ║
# ║  同一 prompt 的多条 response 互相比较                     ║
# ╚═══════════════════════════════════════════════════════════╝
# ============================================================

def compute_grpo_advantage(rewards, prompt_ids):
    """
    rewards:    [total_responses]  outcome reward
    prompt_ids: [total_responses]  每条 response 属于哪个 prompt (0, 0, 0, 0, 1, 1, 1, 1 ...)
    返回: advantages [total_responses]
    """
    advantages = torch.empty_like(rewards)
    for pid in prompt_ids.unique():
        mask = prompt_ids == pid
        group = rewards[mask]
        advantages[mask] = (group - group.mean()) / (group.std() + 1e-8)
    return advantages


# ============================================================
# ╔═══════════════════════════════════════════════════════════╗
# ║       核心逻辑 5: PPO Clipped Loss                        ║
# ║  ratio = exp(logπ_new - logπ_old)                        ║
# ║  loss = -min(ratio*A, clip(ratio)*A)                     ║
# ╚═══════════════════════════════════════════════════════════╝
# ============================================================

def ppo_clipped_loss(old_log_probs, new_log_probs, advantages, response_mask):
    """
    PPO clipped objective. 只对 response 部分算 loss。

    old_log_probs: [batch, seq]  采样时的 logP (rollout 记录的)
    new_log_probs: [batch, seq]  当前模型参数的 logP (训练时重算的)
    advantages:    [batch, seq]  每个 token 的优势值 (response 部分填充 advantage 标量)
    response_mask: [batch, seq]  1=response, 0=prompt (不参与 loss)
    """
    ratio = (new_log_probs - old_log_probs).exp()
    clipped_ratio = ratio.clamp(1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON)
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    loss_per_token = -torch.min(surrogate1, surrogate2)
    loss = (loss_per_token * response_mask).sum() / (response_mask.sum() + 1e-8)
    return loss


# ============================================================
# ╔═══════════════════════════════════════════════════════════╗
# ║       核心逻辑 6: 训练循环                                ║
# ║  1. Rollout  →  old_log_probs                            ║
# ║  2. Reward   →  rewards                                  ║
# ║  3. GRPO     →  advantages                               ║
# ║  4. PPO Loss →  backward →  optimizer.step               ║
# ╚═══════════════════════════════════════════════════════════╝
# ============================================================

@torch.no_grad()
def rollout_batch(model, tokenizer, prompts):
    """
    对一批 prompt 做 rollout: 每个 prompt 生成 N_ROLLOUT 条 response。
    返回拼接后的一整批数据。
    """
    all_input_ids = []
    all_old_log_probs = []
    all_response_masks = []
    all_rewards = []
    all_prompt_ids = []

    for pid, prompt_ids in enumerate(prompts):
        for _ in range(N_ROLLOUT):
            input_ids, old_log_probs, response_mask = rollout_single(model, prompt_ids, tokenizer)

            # 计算 reward (先 decode 成文本)
            response_text = tokenizer.decode(
                input_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True
            )
            reward = compute_reward(response_text)

            all_input_ids.append(input_ids[0])
            all_old_log_probs.append(old_log_probs)
            all_response_masks.append(response_mask)
            all_rewards.append(reward)
            all_prompt_ids.append(pid)

    # 精度: 各条序列长度可能不同, pad 到一致长度
    max_len = max(ids.shape[0] for ids in all_input_ids)
    batch_size = len(all_input_ids)

    def pad_1d(tensors, dtype=None):
        """把不等长的 1D tensor pad 到 max_len"""
        dtype = dtype or tensors[0].dtype
        out = torch.zeros(batch_size, max_len, dtype=dtype, device="cuda")
        for i, t in enumerate(tensors):
            out[i, :t.shape[0]] = t
        return out

    return (
        pad_1d(all_input_ids, dtype=torch.long),       # [batch, max_len]
        pad_1d(all_old_log_probs),                      # [batch, max_len]
        pad_1d(all_response_masks),                     # [batch, max_len]
        torch.tensor(all_rewards, device="cuda"),       # [batch]
        torch.tensor(all_prompt_ids, device="cuda"),    # [batch]
    )


def grpo_train_step(model, optimizer, input_ids, old_log_probs, response_mask, rewards, prompt_ids):
    """
    一次完整的 GRPO 训练步: 算优势 + PPO loss + 反向传播。
    """
    # ---- 1. GRPO 优势 (组内归一化) ----
    advantages = compute_grpo_advantage(rewards, prompt_ids)  # [batch]

    # 扩展到 token 维度: prompt 部分 = 0, response 部分 = 该条的优势值
    token_advantages = advantages.unsqueeze(-1) * response_mask  # [batch, max_len]

    # ---- 2. PPO 多 epoch 更新 ----
    total_loss = 0.0
    for _ in range(PPO_EPOCHS):
        new_log_probs = forward_logprob(model, input_ids)       # [batch, max_len]

        loss = ppo_clipped_loss(
            old_log_probs, new_log_probs, token_advantages, response_mask
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / PPO_EPOCHS


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    # ---- 加载 tokenizer & 模型 ----
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("加载模型...")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    config = Qwen3Config.from_json(f"{MODEL_PATH}/config.json")
    model = Qwen3ForCausalLM(config)
    load_weights(model, f"{MODEL_PATH}/model.safetensors")

    vram_gb = torch.cuda.memory_allocated() / 1e9
    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数: {params_m:.1f}M  |  VRAM: {vram_gb:.2f} GB")

    # ---- 准备 prompt ----
    prompts_text = [
        "请用 Python 写一个快速排序算法。",
        "解释一下什么是梯度下降。",
        "What is the capital of France?",
        "翻译: 人工智能正在改变世界。",
    ]

    prompts = []
    for text in prompts_text:
        messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer.encode(formatted, return_tensors="pt").to("cuda")
        prompts.append(ids)

    # ---- 优化器 ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ---- 训练 ----
    print(f"\n开始 GRPO 训练 (prompts={len(prompts)}, n_rollout={N_ROLLOUT})...")
    print(f"{'='*60}")

    num_epochs = 5
    for epoch in range(num_epochs):
        # Step 1: Rollout
        input_ids, old_log_probs, response_mask, rewards, prompt_ids = rollout_batch(
            model, tokenizer, prompts
        )

        # Step 2: 算奖励 (已经含在 rollout_batch 里了, 只是打印出来看)
        mean_reward = rewards.mean().item()
        print(f"Epoch {epoch+1}/{num_epochs} | mean_reward: {mean_reward:.3f}")

        # Step 3+4: GRPO 优势 + PPO update
        avg_loss = grpo_train_step(
            model, optimizer,
            input_ids, old_log_probs, response_mask, rewards, prompt_ids
        )
        print(f"  loss: {avg_loss:.4f}")

    print(f"\n{'='*60}")
    print("训练完成!")
