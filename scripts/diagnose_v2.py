"""
Exhaustive V2 training diagnostic.

Runs each component of the training pipeline independently and reports
exactly where the learning signal breaks down. Run this BEFORE training to
confirm the pipeline is healthy, or after a failed run to find the root cause.

Usage (Colab / terminal):
  python scripts/diagnose_v2.py --quick          # CPU-only, no model load
  python scripts/diagnose_v2.py --full           # loads model, runs real generation
  python scripts/diagnose_v2.py --full --seed 0  # reproducible
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = PASS if ok else FAIL
    print(f"  {tag} {name}")
    if detail:
        for line in textwrap.wrap(detail, 90):
            print(f"       {line}")
    results.append((name, ok, detail))


def section(title: str) -> None:
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


# ── TEST 1: environment sanity ────────────────────────────────────────────────

def test_env() -> dict | None:
    section("TEST 1 — Environment: can we build an episode?")
    try:
        from env_v2.exec_arena_env import CodeReviewArenaEnv
        env = CodeReviewArenaEnv(difficulty="easy", seed=42, keep_episode_dirs=False)
        obs = env.reset()
        gt = env.ground_truth()
        check("env.reset() succeeds", True)
        check("obs.prompt is non-empty", bool(obs.prompt and obs.prompt.strip()),
              f"prompt length: {len(obs.prompt)} chars")
        check("obs.pr_diff is non-empty", bool(obs.pr_diff and obs.pr_diff.strip()),
              f"diff length: {len(obs.pr_diff)} chars")
        check("ground_truth has verdict", "verdict" in gt, str(gt.keys()))
        check("ground_truth has root_cause", bool(gt.get("root_cause")),
              f"root_cause: {gt.get('root_cause')!r}")
        check("ground_truth has failure_path", bool(gt.get("failure_path")),
              f"failure_path: {gt.get('failure_path')}")
        print(f"\n  {INFO} Prompt preview (first 400 chars):")
        print(textwrap.indent(obs.prompt[:400], "    ") + "...")
        return {"obs": obs, "gt": gt, "env": env}
    except Exception as exc:
        check("env builds without exception", False, str(exc))
        return None


# ── TEST 2: prompt tokenization ───────────────────────────────────────────────

def test_tokenization(env_data: dict | None, tokenizer) -> None:
    section("TEST 2 — Tokenization: does the prompt fit?")
    if env_data is None:
        check("skipped (env failed)", False)
        return
    obs = env_data["obs"]
    tokens = tokenizer(obs.prompt, return_tensors="pt")
    n_tokens = tokens["input_ids"].shape[-1]
    check("prompt fits in 1024 tokens", n_tokens <= 1024,
          f"prompt token count: {n_tokens}")
    if n_tokens > 1024:
        print(f"  {WARN} Prompt will be TRUNCATED by TRL to 1024 tokens. "
              "Model loses context — increase --max_prompt_length.")


# ── TEST 3: raw model generation ─────────────────────────────────────────────

def test_generation(env_data: dict | None, model, tokenizer) -> list[str] | None:
    section("TEST 3 — Raw generation: what does the model actually output?")
    if env_data is None:
        check("skipped (env failed)", False)
        return None

    import torch
    obs = env_data["obs"]

    # Build the chat-format input the way TRL does
    messages = [{"role": "user", "content": obs.prompt}]
    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(model.device)
    except Exception:
        enc = tokenizer(obs.prompt, return_tensors="pt")
        prompt_ids = enc["input_ids"].to(model.device)

    prompt_len = prompt_ids.shape[-1]
    check("prompt encodes successfully", True, f"prompt tokens: {prompt_len}")

    completions: list[str] = []
    for trial in range(4):
        with torch.no_grad():
            out = model.generate(
                prompt_ids,
                max_new_tokens=384,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        new_tokens = out[0, prompt_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completions.append(text)
        n_new = len(new_tokens)
        hit_eos = n_new < 384
        print(f"\n  {INFO} Completion {trial+1} ({n_new} tokens, "
              f"{'terminated' if hit_eos else 'TRUNCATED at max'}):")
        print(textwrap.indent(text[:500], "    "))
        if len(text) > 500:
            print("    ... [truncated for display]")

    # Check if any completion terminated before max
    n_terminated = sum(1 for c in completions if len(tokenizer.encode(c)) < 384)
    check("≥1 completion terminates naturally (not truncated)",
          n_terminated > 0,
          f"{n_terminated}/4 completions terminated before 384 tokens. "
          "If 0: model never outputs EOS — all rewards will be MALFORMED.")

    return completions


# ── TEST 4: parser ────────────────────────────────────────────────────────────

def test_parser(completions: list[str] | None) -> None:
    section("TEST 4 — Parser: does parse_reviewer_output handle model output?")
    if not completions:
        check("skipped (no completions)", False)
        return

    from agents_v2.reviewer_io import parse_reviewer_output, MalformedPrediction

    n_ok = 0
    n_malformed = 0
    for i, text in enumerate(completions):
        pred = parse_reviewer_output(text)
        malformed = pred.is_malformed()
        if malformed:
            n_malformed += 1
        else:
            n_ok += 1
        status = "MalformedPrediction" if malformed else "ReviewerPrediction"
        reason = getattr(pred, "reason", "n/a") if malformed else "—"
        print(f"  {INFO} Completion {i+1}: {status}  reason={reason!r}  verdict={pred.verdict!r}")

    check("≥1 completion parses as ReviewerPrediction", n_ok > 0,
          f"{n_ok}/4 valid, {n_malformed}/4 malformed. "
          "If 0: model never produces parseable output → -0.5 on every sample → std=0.")


# ── TEST 5: reward variance ───────────────────────────────────────────────────

def test_reward_variance(completions: list[str] | None, env_data: dict | None) -> None:
    section("TEST 5 — Reward function: is there any variance?")
    if not completions or not env_data:
        check("skipped", False)
        return

    import statistics
    from agents_v2.reviewer_io import parse_reviewer_output
    from rewards_v2.causal_scorer import CausalScorer
    from env_v2.exec_engine import ExecutionResult, FailedTest

    gt = env_data["gt"]
    env = env_data["env"]
    er_dict = env._state.exec_result.as_dict()

    def _rehydrate(d: dict) -> ExecutionResult:
        failed = [
            FailedTest(
                nodeid=str(t.get("nodeid", "")), file=str(t.get("file", "")),
                line=int(t.get("line", 0) or 0), exception=str(t.get("exception", "")),
                message=str(t.get("message", "")), traceback=str(t.get("traceback", "")),
                call_chain=list(t.get("call_chain", []) or []),
            )
            for t in d.get("failed_tests", [])
        ]
        return ExecutionResult(
            returncode=int(d.get("returncode", 0)),
            duration_s=float(d.get("duration_s", 0.0)),
            passed=int(d.get("passed", 0)),
            failed=int(d.get("failed", 0)),
            errors=int(d.get("errors", 0)),
            failed_tests=failed,
            stdout_tail=str(d.get("stdout_tail", "")),
            stderr_tail=str(d.get("stderr_tail", "")),
            timed_out=bool(d.get("timed_out", False)),
            used_json_report=bool(d.get("used_json_report", False)),
        )

    scorer = CausalScorer()
    er = _rehydrate(er_dict)
    rewards = []
    for i, text in enumerate(completions):
        pred = parse_reviewer_output(text)
        reward, info = scorer.score(pred, gt, er, mutations=[], pr_diff=env_data["obs"].pr_diff)
        rewards.append(reward)
        comps = info.get("reward_components", {})
        print(f"  {INFO} Completion {i+1}: reward={reward:+.3f}  "
              f"root={comps.get('root_cause', 0):.2f}  "
              f"path={comps.get('failure_path', 0):.2f}  "
              f"verdict={comps.get('verdict', 0):.2f}  "
              f"outcome={info.get('episode_outcome')!r}")

    rvar = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
    rmean = statistics.mean(rewards) if rewards else 0.0
    check("reward std > 0.05 (GRPO needs variance)",
          rvar > 0.05,
          f"mean={rmean:+.3f}  std={rvar:.4f}. "
          "If std≈0: all completions get same reward → GRPO advantage=0 → loss=0 forever.")
    check("reward mean > -0.45 (not all malformed)",
          rmean > -0.45,
          f"mean={rmean:+.3f}. If mean≈-0.5: all completions hit MALFORMED_PENALTY.")


# ── TEST 6: GRPO advantage sanity ─────────────────────────────────────────────

def test_grpo_math() -> None:
    section("TEST 6 — GRPO math: what happens when all rewards are equal?")
    import statistics

    def grpo_advantage(rewards: list[float]) -> list[float]:
        mean = statistics.mean(rewards)
        std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        if std < 1e-8:
            return [0.0] * len(rewards)
        return [(r - mean) / std for r in rewards]

    cases = [
        ([-0.5, -0.5, -0.5, -0.5], "all same reward (your current situation)"),
        ([-0.5, -0.3, -0.1, 0.5],  "varied reward (healthy run)"),
        ([-0.5, -0.5, -0.5, 0.0],  "one slightly different"),
        ([1.0, 0.5, -0.5, -1.0],   "full range"),
    ]
    for rewards, label in cases:
        adv = grpo_advantage(rewards)
        std_r = statistics.stdev(rewards)
        std_a = statistics.stdev(adv) if len(adv) > 1 else 0.0
        ok = std_r > 0.05
        check(f"{label}: std_reward={std_r:.3f} → std_advantage={std_a:.3f}", ok)


# ── TEST 7: gold response roundtrip ───────────────────────────────────────────

def test_gold_roundtrip(env_data: dict | None) -> None:
    section("TEST 7 — Gold response: does our SFT warmup target score well?")
    if env_data is None:
        check("skipped (env failed)", False)
        return

    from agents_v2.reviewer_io import parse_reviewer_output
    from rewards_v2.causal_scorer import CausalScorer
    from env_v2.exec_engine import ExecutionResult, FailedTest

    gt = env_data["gt"]
    env = env_data["env"]
    er_dict = env._state.exec_result.as_dict()

    def _rehydrate(d):
        failed = [
            FailedTest(
                nodeid=str(t.get("nodeid", "")), file=str(t.get("file", "")),
                line=int(t.get("line", 0) or 0), exception=str(t.get("exception", "")),
                message=str(t.get("message", "")), traceback=str(t.get("traceback", "")),
                call_chain=list(t.get("call_chain", []) or []),
            )
            for t in d.get("failed_tests", [])
        ]
        return ExecutionResult(
            returncode=int(d.get("returncode", 0)), duration_s=float(d.get("duration_s", 0.0)),
            passed=int(d.get("passed", 0)), failed=int(d.get("failed", 0)),
            errors=int(d.get("errors", 0)), failed_tests=failed,
            stdout_tail=str(d.get("stdout_tail", "")), stderr_tail=str(d.get("stderr_tail", "")),
            timed_out=bool(d.get("timed_out", False)), used_json_report=bool(d.get("used_json_report", False)),
        )

    # Build gold response the way our SFT warmup does
    patterns = env_data.get("env").__class__  # just a placeholder
    root_cause = gt.get("root_cause", "")
    failure_path = gt.get("failure_path", [])
    symbol = root_cause.split("::")[-1] if "::" in root_cause else root_cause
    gold_json = json.dumps({
        "verdict": gt.get("verdict", "REQUEST_CHANGES"),
        "root_cause": root_cause,
        "failure_path": failure_path[:3],
        "confidence": 0.92,
        "reasoning": f"{symbol} is stale — will raise error at runtime.",
    })

    er = _rehydrate(er_dict)
    pred = parse_reviewer_output(gold_json)
    scorer = CausalScorer()
    reward, info = scorer.score(pred, gt, er, mutations=[], pr_diff=env_data["obs"].pr_diff)
    print(f"  {INFO} Gold JSON: {gold_json[:200]}")
    print(f"  {INFO} Parse result: {'ReviewerPrediction' if not pred.is_malformed() else 'MalformedPrediction'}")
    print(f"  {INFO} Reward: {reward:+.3f}  components: {info.get('reward_components', {})}")
    check("gold response parses correctly", not pred.is_malformed())
    check("gold response scores > 0.5",
          reward > 0.5,
          f"reward={reward:+.3f}. If <0.5: SFT warmup teaches a response that won't score well.")


# ── SUMMARY ───────────────────────────────────────────────────────────────────

def print_summary() -> None:
    section("SUMMARY")
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print(f"  {n_pass} passed  |  {n_fail} failed\n")
    if n_fail:
        print(f"  {FAIL} Failed checks:")
        for name, ok, detail in results:
            if not ok:
                print(f"    • {name}")
                if detail:
                    print(f"      → {detail[:120]}")
    else:
        print(f"  {PASS} All checks passed — training pipeline is healthy.")

    print(f"\n  Interpretation guide:")
    print("  • Test 3 clips_ratio=1.0  → completion length too short OR model in degenerate loop")
    print("  • Test 4 all malformed    → model doesn't know output format (SFT warmup needed)")
    print("  • Test 5 std=0            → all rewards identical → GRPO advantage=0 → loss=0 forever")
    print("  • Test 7 gold score<0.5   → SFT warmup target is wrong — fix _gold_response_v2()")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="No model load — env + parser + math only")
    p.add_argument("--full",  action="store_true", help="Load model and run real generation")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--seed",  type=int, default=42)
    args = p.parse_args()

    if not args.quick and not args.full:
        p.print_help()
        print("\nRun with --quick (no GPU) or --full (loads model).")
        sys.exit(0)

    env_data = test_env()
    test_grpo_math()
    test_gold_roundtrip(env_data)

    if args.full:
        print(f"\n{'─'*70}")
        print("  Loading model (this takes ~2 min on T4)...")
        from training.train import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model, backend="hf", seed=args.seed)
        test_tokenization(env_data, tokenizer)
        completions = test_generation(env_data, model, tokenizer)
        test_parser(completions)
        test_reward_variance(completions, env_data)
    else:
        print(f"\n  {INFO} --quick mode: skipping model load (Tests 2-5).")
        print(f"  {INFO} Run with --full on a GPU machine to test actual generation.")

    print_summary()


if __name__ == "__main__":
    main()
