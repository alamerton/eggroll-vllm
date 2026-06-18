#!/usr/bin/env python3
"""Judged win-rate of trained qwen checkpoints vs the initial model.

For each run's latest checkpoint: unfuse the merged vLLM weights into
a transformers model (via merge_checkpoint.unfuse_vllm_to_transformers),
generate one summary per held-out post with the trained and the initial
model (same seed), and score both with the reward-model judge (soft
mode). Reports win_rate = fraction of posts where
P(trained > initial) > 0.5.

Held-out prompts are taken from beyond the training slice (training
used the first --train-slice rows of the corpus).

    python eval_qwen_winrate.py \
        --run-dir "$SCRATCH/for_es_lora/experiments/qwen06b_maxlot_seed0-*" \
        --run-dir "$SCRATCH/for_es_lora/experiments/qwen06b_ipo_seed0-*" \
        --n 50 --out winrates_qwen.json
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
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=64)
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
    seen = set()
    i = args.train_slice
    # The comparisons corpus repeats posts across rows; dedupe so the
    # N held-out prompts are N distinct posts.
    while len(prompts) < args.n and i < len(rows):
        p = rows[i]["prompt"]
        if p not in seen:
            seen.add(p)
            prompts.append(p)
        i += 1

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
                **ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                temperature=args.temperature, pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                skip_special_tokens=True)

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
            # A tie is NOT a loss. When the model barely moved, trained and
            # reference generations are identical -> equal judge reward ->
            # p == 0.5 exactly; counting that as a loss floors a non-moving
            # model near (1 - tie_rate) * 0.5 (~0.40 at tie_rate 0.2). Give
            # ties half credit; mean_p below is the tie-immune companion.
            if p > 0.5:
                wins += 1.0
            elif p == 0.5:
                wins += 0.5
                ties += 1
            if i < 3:
                samples.append({"prompt_tail": prompt[-120:],
                                "trained": trained_text, "initial": ref_text,
                                "p": round(p, 3)})
        base.load_state_dict(base_state)  # clean slate for next run
        results[os.path.basename(run_dir)] = {
            "checkpoint": os.path.basename(ckpt),
            "win_rate": wins / len(prompts),
            "mean_p": sum(probs) / len(probs),
            "ties": ties,
            "tie_rate": ties / len(prompts),
            "n": len(prompts),
            "samples": samples,
        }
        print(f"  win_rate={wins/len(prompts):.3f} mean_p={sum(probs)/len(probs):.3f} "
              f"ties={ties}/{len(prompts)}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
