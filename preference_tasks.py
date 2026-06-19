"""Preference-pair tasks for ES-LoRA.

Unlike the scalar-reward tasks in `tasks.py` (math, countdown), a
preference task has no ground-truth answer: each population member
generates a PAIR of responses per prompt (``samples_per_prompt=2``), a
pairwise judge (a HuggingFace scalar reward model) prefers one side,
and the fitness is built from the preference and, for the
maximal-lottery objective, from the sequence log-probs of the pair
under the member's policy and under the unperturbed base policy.

Objectives (``s = 2 p - 1`` with ``p = P(a preferred over b)``):

* ``naive``  -- ``f = s``. Pure preference signal.
* ``maxlot`` -- ``f = s * [pi(a) * pibar(b) - pi(b) * pibar(a)]``
  where ``pi`` is the member's policy and ``pibar`` the unperturbed
  base (the frozen current policy: ES-LoRA merges updates into the
  base weights each step, so scoring with no LoRA adapter is exactly
  pibar). Probabilities are ``exp`` of length-normalised sequence
  log-probs. At the regularised fixed point the policy is the maximal
  lottery of the judge's preference matrix; the one-term variant
  collapses to the Borda winner, hence the antisymmetric margin.

* ``ipo`` / ``dpo`` -- the reference-anchored objectives (Online-IPO and
  DPO) on length-normalised log-probs. They anchor on the FROZEN INITIAL
  policy, which ES-LoRA's merge-into-base update destroys -- so the
  engine actor keeps a frozen HF copy of the starting weights and
  teacher-force-scores the pair under it
  (``needs_ref_scores``; "just create a copy", per Bidipta).

The judge is loaded lazily and dropped from the pickled state, so
``ray.put(task)`` ships only the lightweight config to each engine and
every engine loads its own copy on first use.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

_JUDGE_MODES = ("soft", "hard", "bernoulli")
_OBJECTIVES = ("naive", "maxlot", "ipo", "dpo", "cross")


def _log_sigmoid(x: float) -> float:
    """Numerically stable log(sigmoid(x))."""
    if x >= 0:
        return -math.log1p(math.exp(-x))
    return x - math.log1p(math.exp(x))


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid; doesn't overflow for large |x|."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _uniform_from_key(key) -> float:
    """Deterministic uniform-in-[0, 1) draw seeded by an arbitrary key.

    BLAKE2b over ``repr(key)`` so the draw is stable across Python
    processes and Ray workers (unlike ``hash()``, which is
    PYTHONHASHSEED-dependent).
    """
    digest = hashlib.blake2b(repr(key).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / (1 << 64)


class TldrPreferenceTask:
    """Online TL;DR summarisation judged by a scalar reward model.

    Args:
        batch_size: prompts per training step.
        seed: RNG seed for prompt sampling.
        repo / split / dataset_size: HuggingFace preference corpus to
            draw prompts from (only the prompt column is used; the
            human-written summaries are not shown to the policy).
        prompt_column: prompt column name on the HF rows.
        prompt_suffix: appended to every prompt so the policy is cued to
            SUMMARISE rather than continue the post. The CarperAI prompts
            end at the post body with no cue, so the model just keeps
            writing the post; "\nTL;DR:" is the canonical completion
            format and the judge scores prompt+response, so the cue is
            seen consistently on both sides. Pass "" to disable.
        judge_model: HF repo id of the sequence-classification reward
            model used as the pairwise judge.
        judge_mode: "soft" (return sigmoid(r_a - r_b)), "hard"
            (argmax), or "bernoulli" (deterministic keyed coin flip --
            simulates a single noisy rater's click).
        judge_max_length: judge tokeniser truncation length.
        judge_device: where the judge runs. Default "cpu": the vLLM
            engine typically owns ~all GPU memory; pass "cuda" if the
            card has headroom.
        objective: "naive" or "maxlot" (see module docstring).
        rows: explicit list of dicts bypassing the HF load (tests).
        score_fn: optional ``str -> float`` scorer bypassing the HF
            judge load (tests).
    """

    is_preference = True

    def __init__(
        self,
        batch_size,
        seed,
        repo="CarperAI/openai_summarize_comparisons",
        split="train",
        dataset_size=None,
        eval_heldout_size=0,
        prompt_column="prompt",
        prompt_suffix="\nTL;DR:",
        judge_model="OpenAssistant/reward-model-deberta-v3-base",
        judge_mode="bernoulli",
        judge_max_length=512,
        judge_device="cpu",
        objective="maxlot",
        ipo_tau_inv=100.0,
        dpo_beta=0.1,
        pair_normalise=False,
        rows=None,
        score_fn=None,
    ):
        if judge_mode not in _JUDGE_MODES:
            raise ValueError(f"judge_mode must be one of {_JUDGE_MODES}; got {judge_mode!r}")
        if objective not in _OBJECTIVES:
            raise ValueError(f"objective must be one of {_OBJECTIVES}; got {objective!r}")
        if ipo_tau_inv <= 0.0:
            raise ValueError(f"ipo_tau_inv must be > 0, got {ipo_tau_inv}")
        if dpo_beta <= 0.0:
            raise ValueError(f"dpo_beta must be > 0, got {dpo_beta}")
        self.batch_size = batch_size
        self.judge_model = judge_model
        self.judge_mode = judge_mode
        self.judge_max_length = int(judge_max_length)
        self.judge_device = judge_device
        self.objective = objective
        self.ipo_tau_inv = float(ipo_tau_inv)
        self.dpo_beta = float(dpo_beta)
        # Within-pair softmax for the maxlot margin: raw
        # exp(length-normalised lp) is a confidence, not a probability,
        # and is gamed by deterministic repetition; the pair-restricted
        # probabilities cancel uniform confidence inflation.
        self.pair_normalise = bool(pair_normalise)
        self.needs_base_scores = objective == "maxlot"
        # ipo/dpo anchor on the FROZEN INITIAL policy; the engine's own
        # weights have ES updates merged in, so the actor scores these
        # with a frozen HF copy of the starting weights instead.
        self.needs_ref_scores = objective in ("ipo", "dpo")
        # "cross" is Roberto's prose scheme: one sample per antithetic
        # twin (a from +sigma, b from -sigma), one fitness per twin
        # pair, broadcast [f, -f]. The engine pairs the members.
        self.cross_pairs = objective == "cross"
        self.rng = np.random.default_rng(seed)

        self.eval_heldout_size = int(eval_heldout_size)
        heldout_dicts = []
        if rows is not None:
            row_dicts = [dict(r) for r in rows]
        else:
            from datasets import load_dataset

            hf = load_dataset(repo, split=split)
            train_end = min(dataset_size, len(hf)) if dataset_size is not None else len(hf)
            row_dicts = list(hf.select(range(train_end)))
            # Held-out eval set for the win-rate-vs-init curve: distinct prompts
            # BEYOND the training slice, excluding any whose text also appears in
            # training (the corpus repeats posts across rows). Fixed, so the
            # curve is comparable across steps.
            if self.eval_heldout_size > 0 and train_end < len(hf):
                seen = {r[prompt_column] for r in row_dicts}
                for j in range(train_end, len(hf)):
                    r = hf[j]
                    if r[prompt_column] not in seen:
                        seen.add(r[prompt_column])
                        heldout_dicts.append(r)
                        if len(heldout_dicts) >= self.eval_heldout_size:
                            break
        if not row_dicts:
            raise ValueError(f"loaded zero rows from {repo!r} split={split!r}")
        if prompt_column not in row_dicts[0]:
            raise ValueError(
                f"rows are missing column {prompt_column!r}; got keys "
                f"{list(row_dicts[0].keys())}"
            )
        self.prompt_suffix = str(prompt_suffix)
        self.prompts = [r[prompt_column] + self.prompt_suffix for r in row_dicts]
        self.heldout_prompts = [r[prompt_column] + self.prompt_suffix for r in heldout_dicts]

        # The judge is heavyweight and CUDA-bound; load lazily and keep
        # it out of the pickled state (see __getstate__).
        self._score_fn = score_fn
        self._injected_score_fn = score_fn is not None

    # ── Ray serialisation: ship config, not the judge ───────────────────
    def __getstate__(self):
        state = dict(self.__dict__)
        if not self._injected_score_fn:
            state["_score_fn"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    # ── Judge ────────────────────────────────────────────────────────────
    def _ensure_judge(self):
        if self._score_fn is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        device = self.judge_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)
        tokenizer = AutoTokenizer.from_pretrained(self.judge_model)
        model = AutoModelForSequenceClassification.from_pretrained(self.judge_model)
        model.to(device)
        model.eval()

        def hf_score(text):
            with torch.no_grad():
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.judge_max_length,
                ).to(device)
                logits = model(**inputs).logits
            # Scalar reward models output one logit; take the last
            # column to be robust to (1, 1) vs (1, K) shapes.
            return float(logits[0, -1].item())

        self._score_fn = hf_score

    def score_pair(self, prompt, response_a, response_b, key):
        """Return P(a preferred over b) per the configured judge mode."""
        self._ensure_judge()
        score_a = self._score_fn(prompt + response_a)
        score_b = self._score_fn(prompt + response_b)
        p = _sigmoid(score_a - score_b)
        if self.judge_mode == "soft":
            return p
        if self.judge_mode == "hard":
            return 1.0 if p > 0.5 else 0.0
        return 1.0 if _uniform_from_key(key) < p else 0.0

    def reward(self, text):
        """Raw judge reward for one (prompt+response) text; ensures the judge is
        loaded. Used by the in-loop win-rate-vs-init eval (runs on the engine)."""
        self._ensure_judge()
        return self._score_fn(text)

    # ── Task interface ───────────────────────────────────────────────────
    def get_batch(self):
        indices = self.rng.integers(0, len(self.prompts), size=self.batch_size)
        batch_prompts = [self.prompts[i] for i in indices]
        return batch_prompts, [None for _ in batch_prompts]

    def get_fitness_pairwise(
        self,
        prompt,
        responses,
        truncateds,
        gen_logprobs,
        gen_token_lens,
        base_logprobs,
        ref_logprobs,
        key,
    ):
        """Fitness for one (population member, prompt) response pair.

        Args:
            prompt: the prompt text.
            responses: ``[a, b]`` -- the member's two sampled responses.
            truncateds: per-response hit-the-length-limit flags
                (currently informational only).
            gen_logprobs: summed token log-probs of each response under
                the member's (LoRA-perturbed) policy.
            gen_token_lens: token count of each response (shared by the
                base/ref scores, which score the same tokens).
            base_logprobs: summed token log-probs under the unperturbed
                *current* base policy (maxlot's pibar), or None when
                ``needs_base_scores`` is False.
            ref_logprobs: summed token log-probs under the FROZEN
                INITIAL policy (ipo/dpo's reference), or None when
                ``needs_ref_scores`` is False.
            key: deterministic key for the bernoulli judge draw.

        Returns:
            ``(fitness, info)`` with per-pair diagnostics in ``info``.
        """
        response_a, response_b = responses
        p = self.score_pair(prompt, response_a, response_b, key)
        s = 2.0 * p - 1.0
        info = {"pref_p": p}
        if self.objective == "naive":
            return s, info

        len_a = max(int(gen_token_lens[0]), 1)
        len_b = max(int(gen_token_lens[1]), 1)
        lp_a = gen_logprobs[0] / len_a
        lp_b = gen_logprobs[1] / len_b

        if self.objective == "maxlot":
            base_lp_a = base_logprobs[0] / len_a
            base_lp_b = base_logprobs[1] / len_b
            if self.pair_normalise:
                # Within-pair win probabilities; margin = p(a) - pbar(a).
                p_a = _sigmoid(lp_a - lp_b)
                pbar_a = _sigmoid(base_lp_a - base_lp_b)
                margin = p_a - pbar_a
            else:
                margin = math.exp(lp_a + base_lp_b) - math.exp(lp_b + base_lp_a)
            info["margin"] = margin
            return s * margin, info

        if self.objective == "cross":
            raise RuntimeError(
                "cross objective uses get_fitness_cross (one sample per "
                "twin), not get_fitness_pairwise"
            )

        # ipo / dpo: reference-subtracted log-ratio, soft-p weighted over
        # both orientations (Online-IPO / DPO, length-normalised).
        ref_lp_a = ref_logprobs[0] / len_a
        ref_lp_b = ref_logprobs[1] / len_b
        log_ratio = (lp_a - lp_b) - (ref_lp_a - ref_lp_b)
        info["log_ratio"] = log_ratio
        if self.objective == "ipo":
            target = self.ipo_tau_inv / 2.0
            loss = p * (log_ratio - target) ** 2 + (1.0 - p) * (-log_ratio - target) ** 2
        else:  # dpo
            margin = self.dpo_beta * log_ratio
            loss = -p * _log_sigmoid(margin) - (1.0 - p) * _log_sigmoid(-margin)
        return -loss, info

    def get_fitness_cross(
        self,
        prompt,
        response_a,
        response_b,
        truncateds,
        lp_plus_a,
        len_a,
        lp_minus_b,
        len_b,
        key,
    ):
        """Fitness for one antithetic twin pair (cross objective).

        Roberto's prose scheme verbatim: a was generated by the +sigma
        twin, b by the -sigma twin; ``f = s * pi+(a) * pi-(b)`` with
        length-normalised log-probs from each twin's own generation.
        The caller broadcasts [f, -f] to the (+, -) twins.

        Returns ``(fitness, info)``.
        """
        p = self.score_pair(prompt, response_a, response_b, key)
        s = 2.0 * p - 1.0
        lp_a = lp_plus_a / max(int(len_a), 1)
        lp_b = lp_minus_b / max(int(len_b), 1)
        product = math.exp(lp_a + lp_b)
        return s * product, {"pref_p": p, "product": product}
