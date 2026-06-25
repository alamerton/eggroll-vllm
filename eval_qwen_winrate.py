#!/usr/bin/env python3
"""Judged win-rate of trained qwen checkpoints vs the initial model.

For each run's latest checkpoint: unfuse the merged vLLM weights into
a transformers model (via merge_checkpoint.unfuse_vllm_to_transformers),
generate one summary per held-out post with the trained and the initial
model, with decoding matched to training (a "\nTL;DR:" cue, a newline
stop, full-distribution sampling), and score both with the reward-model
judge. PRIMARY metric is mean_p = mean P(trained > initial); 0.5 is the
null. win_rate (ties credited 0.5) is secondary. Both come with 95% CIs
-- at small n they usually straddle 0.5, so use --n >= 500.

Held-out prompts are taken from beyond the training slice AND are
text-excluded from it (the corpus repeats posts across rows).

    python eval_qwen_winrate.py \
        --run-dir "$SCRATCH/for_es_lora/experiments/qwen06b_dpo_normstd_beta5_seed0-*" \
        --n 500 --out winrates_qwen.json
"""

import argparse
import glob
import json
import math
import os
import re

import torch
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

def unfuse_vllm_to_transformers(vllm_weights, model_config):
    """Unfuse vLLM's fused weights (qkv_proj, gate_up_proj) to
    transformers format. Inlined from merge_checkpoint.py (whose module
    import drags in stale task names and vllm)."""
    num_attention_heads = model_config.num_attention_heads
    num_key_value_heads = model_config.num_key_value_heads
    head_dim = getattr(
        model_config, "head_dim",
        model_config.hidden_size // num_attention_heads,
    )
    intermediate_size = model_config.intermediate_size

    unfused = {}
    for name, weight in vllm_weights.items():
        clean_name = name.replace(".base_layer", "")
        if "qkv_proj" in name:
            prefix = clean_name.replace("qkv_proj.weight", "")
            q_size = num_attention_heads * head_dim
            kv_size = num_key_value_heads * head_dim
            unfused[prefix + "q_proj.weight"] = weight[:q_size, :]
            unfused[prefix + "k_proj.weight"] = weight[q_size:q_size + kv_size, :]
            unfused[prefix + "v_proj.weight"] = weight[q_size + kv_size:, :]
        elif "gate_up_proj" in name:
            prefix = clean_name.replace("gate_up_proj.weight", "")
            unfused[prefix + "gate_proj.weight"] = weight[:intermediate_size, :]
            unfused[prefix + "up_proj.weight"] = weight[intermediate_size:, :]
        else:
            unfused[clean_name] = weight
    return unfused


def latest_checkpoint(run_dir):
    steps = []
    for p in glob.glob(os.path.join(run_dir, "checkpoints", "checkpoint_step_*")):
        m = re.search(r"checkpoint_step_(\d+)$", p)
        if m:
            steps.append((int(m.group(1)), p))
    if not steps:
        raise FileNotFoundError(f"no checkpoints under {run_dir}")
    return max(steps)[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", action="append", required=True, dest="run_dirs",
                    help="experiment dir (glob ok); repeatable")
    ap.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--judge-model", default="OpenAssistant/reward-model-deberta-v3-base")
    ap.add_argument("--repo", default="CarperAI/openai_summarize_comparisons")
    ap.add_argument("--train-slice", type=int, default=2000,
                    help="rows used in training; held-out starts after this")
    ap.add_argument("--n", type=int, default=500,
                    help="held-out prompts; CI ~ +/-0.5/sqrt(n) (~0.044 at 500, ~0.098 at 100)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--prompt-suffix", default="\nTL;DR:",
                    help="cue appended to every prompt; MUST match the task's "
                         "prompt_suffix (preference_tasks.py) so eval == train")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)

    from datasets import load_dataset
    rows = load_dataset(args.repo, split="train")
    prompts = []
    # The comparisons corpus repeats posts across rows. Seed the dedupe set
    # with EVERY training-slice post (by text) so a memorised post can't leak
    # into the held-out set, then dedupe the held-out posts among themselves.
    seen = {rows[j]["prompt"] for j in range(min(args.train_slice, len(rows)))}
    i = args.train_slice
    while len(prompts) < args.n and i < len(rows):
        p = rows[i]["prompt"]
        if p not in seen:
            seen.add(p)
            prompts.append(p + args.prompt_suffix)  # cue, exactly like training
        i += 1
    if len(prompts) < args.n:
        print(f"warn: only {len(prompts)} distinct held-out prompts available "
              f"(< requested {args.n}); CI widens accordingly")

    judge_tok = AutoTokenizer.from_pretrained(args.judge_model)
    judge = AutoModelForSequenceClassification.from_pretrained(args.judge_model)
    judge.to(device).eval()

    def reward(text):
        with torch.no_grad():
            inputs = judge_tok(text, return_tensors="pt", truncation=True,
                               max_length=512).to(device)
            return float(judge(**inputs).logits[0, -1].item())

    def generate(model, prompt, seed):
        torch.manual_seed(seed)
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=448).to(device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=args.max_new_tokens, min_new_tokens=4,
                do_sample=True, temperature=args.temperature,
                # Match training's vLLM SamplingParams: full distribution
                # (top_k/top_p disabled) and stop at the first newline so a
                # one-line TL;DR doesn't run on into garbage the judge scores.
                top_p=1.0, top_k=0,
                stop_strings=["\n"], tokenizer=tokenizer,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        return text.rstrip("\n")  # HF keeps the stop string; vLLM strips it

    print(f"loading initial model {args.model_name}")
    base = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.bfloat16).to(device).eval()
    base_state = {k: v.clone() for k, v in base.state_dict().items()}

    results = {}
    for pattern in args.run_dirs:
        matches = sorted(glob.glob(os.path.expandvars(pattern)))
        if not matches:
            print(f"warn: no match for {pattern}")
            continue
        run_dir = matches[-1]  # latest timestamp
        ckpt = latest_checkpoint(run_dir)
        print(f"\n=== {os.path.basename(run_dir)} ({os.path.basename(ckpt)})")
        vllm_weights = load_file(os.path.join(ckpt, "model_weights.safetensors"))
        unfused = unfuse_vllm_to_transformers(vllm_weights, config)
        missing, unexpected = base.load_state_dict(unfused, strict=False)
        print(f"  loaded ({len(missing)} missing, {len(unexpected)} unexpected keys)")

        wins, ties, probs, samples = 0.0, 0, [], []
        for i, prompt in enumerate(prompts):
            seed = args.seed * 100003 + i
            trained_text = generate(base, prompt, seed)   # base now holds trained weights
            # restore initial weights for the reference generation
            base.load_state_dict(base_state)
            ref_text = generate(base, prompt, seed)
            base.load_state_dict(unfused, strict=False)   # back to trained
            p = 1.0 / (1.0 + math.exp(-(reward(prompt + trained_text)
                                        - reward(prompt + ref_text))))
            probs.append(p)
            # Identical generations (the model barely moved) are a genuine tie,
            # not a loss: half credit. Scoring them as losses floors a non-moving
            # model near (1 - tie_rate) * 0.5. mean_p is the tie-immune headline.
            if trained_text == ref_text:
                wins += 0.5
                ties += 1
            elif p > 0.5:
                wins += 1.0
            if i < 10:
                samples.append({"prompt_tail": prompt[-120:],
                                "trained": trained_text, "initial": ref_text,
                                "p": round(p, 3)})
        base.load_state_dict(base_state)  # clean slate for next run
        n = len(prompts)
        mean_p = sum(probs) / n
        win_rate = wins / n
        # 95% CIs (normal approx): win_rate binomial, mean_p from sample std.
        wr_ci = 1.96 * math.sqrt(win_rate * (1 - win_rate) / n) if n else 0.0
        p_std = (sum((x - mean_p) ** 2 for x in probs) / (n - 1)) ** 0.5 if n > 1 else 0.0
        mp_ci = 1.96 * p_std / math.sqrt(n) if n else 0.0
        results[os.path.basename(run_dir)] = {
            "checkpoint": os.path.basename(ckpt),
            "mean_p": mean_p,            # PRIMARY metric; 0.5 is the null
            "mean_p_ci95": mp_ci,
            "win_rate": win_rate,        # secondary; ties credited 0.5
            "win_rate_ci95": wr_ci,
            "ties": ties,
            "tie_rate": ties / n,
            "n": n,
            "probs": [round(x, 4) for x in probs],  # for post-hoc CIs / bootstrap
            "samples": samples,
        }
        print(f"  mean_p={mean_p:.3f} +/-{mp_ci:.3f} | "
              f"win_rate={win_rate:.3f} +/-{wr_ci:.3f} | ties={ties}/{n}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
