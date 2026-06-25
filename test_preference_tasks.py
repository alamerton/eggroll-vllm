"""Network-free unit tests for `preference_tasks`.

Run with ``pytest test_preference_tasks.py`` or plain
``python test_preference_tasks.py``. Uses injected ``rows`` and
``score_fn`` so no HuggingFace download, GPU, or vLLM is needed.
"""

from __future__ import annotations

import math
import pickle

from preference_tasks import TldrPreferenceTask

_ROWS = [{"prompt": f"POST {i}\nTL;DR:"} for i in range(8)]


def _task(objective="maxlot", judge_mode="soft", score_fn=None, seed=0):
    return TldrPreferenceTask(
        batch_size=4,
        seed=seed,
        rows=_ROWS,
        score_fn=score_fn or (lambda text: float(len(text))),
        judge_mode=judge_mode,
        objective=objective,
    )


def test_get_batch_shape_and_membership():
    task = _task()
    prompts, answers = task.get_batch()
    assert len(prompts) == 4 and len(answers) == 4
    assert all(a is None for a in answers)
    assert all(p in [r["prompt"] for r in _ROWS] for p in prompts)


def test_get_batch_is_seed_deterministic():
    assert _task(seed=7).get_batch()[0] == _task(seed=7).get_batch()[0]


def test_soft_judge_prefers_higher_reward():
    # Longer text scores higher under the stub scorer.
    task = _task(judge_mode="soft")
    p = task.score_pair("P", "a longer response", "ab", key=(0, 0, 0))
    assert p > 0.5


def test_naive_fitness_is_centred_preference():
    task = _task(objective="naive", judge_mode="hard")
    fit, info = task.get_fitness_pairwise(
        "P",
        ["a longer response", "ab"],
        [False, False],
        gen_logprobs=[0.0, 0.0],
        gen_token_lens=[4, 1],
        base_logprobs=None,
        ref_logprobs=None,
        key=(0, 0, 0),
    )
    assert fit == 1.0 and info["pref_p"] == 1.0
    assert task.needs_base_scores is False


def test_maxlot_fitness_matches_hand_computation():
    task = _task(objective="maxlot", judge_mode="hard")
    lp = [math.log(0.5) * 4, math.log(0.25) * 2]      # per-token 0.5 / 0.25
    base = [math.log(0.2) * 4, math.log(0.4) * 2]     # per-token 0.2 / 0.4
    fit, info = task.get_fitness_pairwise(
        "P",
        ["a longer response", "ab"],
        [False, False],
        gen_logprobs=lp,
        gen_token_lens=[4, 2],
        base_logprobs=base,
        ref_logprobs=None,
        key=(0, 0, 0),
    )
    margin = 0.5 * 0.4 - 0.25 * 0.2  # 0.15
    assert math.isclose(info["margin"], margin, rel_tol=1e-12)
    assert math.isclose(fit, margin, rel_tol=1e-12)  # s = +1 (a longer)


def test_maxlot_fitness_is_invariant_under_pair_relabeling():
    """Swapping (a, b) everywhere flips both s and the margin, so the
    member's fitness is unchanged -- which side is called 'a' is an
    arbitrary label (soft judge: p -> 1-p exactly)."""
    task = _task(objective="maxlot", judge_mode="soft")

    def fitness_and_margin(resps, lps, lens, base):
        fit, info = task.get_fitness_pairwise(
            "P", resps, [False, False], lps, lens, base, None, key=(0, 0, 0)
        )
        return fit, info["margin"]

    forward, margin_fwd = fitness_and_margin(
        ["a longer response", "ab"],
        [math.log(0.5) * 4, math.log(0.25) * 2],
        [4, 2],
        [math.log(0.2) * 4, math.log(0.4) * 2],
    )
    swapped, margin_swp = fitness_and_margin(
        ["ab", "a longer response"],
        [math.log(0.25) * 2, math.log(0.5) * 4],
        [2, 4],
        [math.log(0.4) * 2, math.log(0.2) * 4],
    )
    assert math.isclose(margin_fwd, -margin_swp, rel_tol=1e-12)  # antisymmetric
    assert math.isclose(forward, swapped, rel_tol=1e-9)  # label-invariant


def test_ipo_fitness_matches_hand_computation():
    task = TldrPreferenceTask(
        batch_size=2, seed=0, rows=_ROWS,
        score_fn=lambda text: float(len(text)),
        judge_mode="hard", objective="ipo", ipo_tau_inv=4.0,
    )
    assert task.needs_ref_scores and not task.needs_base_scores
    # Normalised: lp_a=log.5, lp_b=log.25; ref_a=log.25, ref_b=log.5
    # log_ratio = 2*log(2); p=1, target=2 -> loss=(2ln2-2)^2.
    fit, info = task.get_fitness_pairwise(
        "P",
        ["a longer response", "ab"],
        [False, False],
        gen_logprobs=[math.log(0.5) * 2, math.log(0.25) * 2],
        gen_token_lens=[2, 2],
        base_logprobs=None,
        ref_logprobs=[math.log(0.25) * 2, math.log(0.5) * 2],
        key=(0, 0, 0),
    )
    log_ratio = 2 * math.log(2)
    assert math.isclose(info["log_ratio"], log_ratio, rel_tol=1e-12)
    assert math.isclose(fit, -((log_ratio - 2.0) ** 2), rel_tol=1e-12)


def test_dpo_fitness_matches_hand_computation():
    task = TldrPreferenceTask(
        batch_size=2, seed=0, rows=_ROWS,
        score_fn=lambda text: float(len(text)),
        judge_mode="hard", objective="dpo", dpo_beta=1.0,
    )
    assert task.needs_ref_scores
    fit, info = task.get_fitness_pairwise(
        "P",
        ["a longer response", "ab"],
        [False, False],
        gen_logprobs=[math.log(0.5) * 2, math.log(0.25) * 2],
        gen_token_lens=[2, 2],
        base_logprobs=None,
        ref_logprobs=[math.log(0.25) * 2, math.log(0.5) * 2],
        key=(0, 0, 0),
    )
    log_ratio = 2 * math.log(2)
    expected_loss = math.log(1.0 + math.exp(-log_ratio))  # -log sigmoid(margin)
    assert math.isclose(fit, -expected_loss, rel_tol=1e-12)


def test_objective_flags():
    for objective, base, ref in (
        ("naive", False, False),
        ("maxlot", True, False),
        ("ipo", False, True),
        ("dpo", False, True),
    ):
        task = _task(objective=objective) if objective in ("naive", "maxlot") else \
            TldrPreferenceTask(batch_size=2, seed=0, rows=_ROWS,
                               score_fn=lambda t: 0.0, objective=objective)
        assert task.needs_base_scores is base, objective
        assert task.needs_ref_scores is ref, objective


def test_bernoulli_draw_is_key_deterministic():
    task = _task(judge_mode="bernoulli")
    draws = {task.score_pair("P", "aaa", "bb", key=(1, 2, 3)) for _ in range(5)}
    assert len(draws) == 1  # same key, same outcome
    assert draws.pop() in (0.0, 1.0)


def test_pickle_drops_lazy_judge_but_keeps_config():
    """ray.put round-trip must not ship a loaded judge."""
    task = TldrPreferenceTask(
        batch_size=2, seed=0, rows=_ROWS, judge_mode="soft", objective="maxlot"
    )
    clone = pickle.loads(pickle.dumps(task))
    assert clone._score_fn is None  # reloads lazily on the worker
    assert clone.judge_model == task.judge_model
    assert clone.prompts == task.prompts


def test_rejects_bad_modes():
    for kwargs in ({"judge_mode": "argmax"}, {"objective": "rlhf"}):
        try:
            _task(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"OK   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
