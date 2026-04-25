"""V2 training driver: pre-generates a dataset of (prompt, ground_truth) rows
from :class:`env_v2.exec_arena_env.CodeReviewArenaEnv` and trains a Qwen
LoRA policy with TRL GRPO using :class:`rewards_v2.causal_scorer.CausalScorer`.

Design decisions
----------------
- Ground truth is captured at dataset-build time so the GRPO reward function
  is fast (no pytest in the inner training loop).
- The reward function parses each completion with the V2 reviewer parser,
  then scores against the cached ground truth + cached exec result.
- Per-component reward is logged to W&B keys ``reward/root_cause``,
  ``reward/failure_path``, ``reward/verdict``, ``reward/hallucination``,
  ``reward/calibration``, ``reward/total``.
- Reuses the existing dtype guards from the v1 trainer (Qwen + QLoRA on
  small GPUs).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

if __name__ == "__main__" and sys.platform == "win32" and not getattr(sys.flags, "utf8_mode", False):
    os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents_v2.generator_agent import ALL_PATTERNS  # noqa: E402
from env_v2.exec_arena_env import CodeReviewArenaEnv  # noqa: E402
from env_v2.exec_engine import ExecutionResult, FailedTest  # noqa: E402
from rewards_v2.causal_scorer import CausalScorer  # noqa: E402
from training_v2.curriculum import Curriculum  # noqa: E402
from utils_v2.metrics import MetricsLogger  # noqa: E402

log = logging.getLogger("codedrift_v2.train")

# Heldout split (mirrors plan section 9). Generator-only patterns at train time;
# the eval script restricts to held-out patterns.
TRAIN_PATTERNS = ["rename", "removal", "contract", "partial_rename", "null_missing"]
HELDOUT_PATTERNS = ["type_mismatch", "condition_flip", "off_by_one"]


def _serialize_failed_tests(tests: list[FailedTest]) -> list[dict[str, Any]]:
    return [
        {
            "nodeid": t.nodeid,
            "file": t.file,
            "line": t.line,
            "exception": t.exception,
            "message": t.message,
            "traceback": t.traceback,
            "call_chain": list(t.call_chain),
        }
        for t in tests
    ]


def _exec_result_from_dict(d: dict[str, Any]) -> ExecutionResult:
    failed = [
        FailedTest(
            nodeid=str(t.get("nodeid", "")),
            file=str(t.get("file", "")),
            line=int(t.get("line", 0) or 0),
            exception=str(t.get("exception", "")),
            message=str(t.get("message", "")),
            traceback=str(t.get("traceback", "")),
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


def build_dataset(
    n_episodes: int,
    seed: int,
    allowed_patterns: list[str],
    use_curriculum: bool,
) -> list[dict[str, Any]]:
    """Pre-generate training rows: (prompt, ground_truth, exec_result, pr_diff)."""
    env = CodeReviewArenaEnv(
        difficulty="easy",
        seed=seed,
        allowed_patterns=allowed_patterns,
        keep_episode_dirs=False,
    )
    curriculum = Curriculum(seed=seed, allowed_patterns=allowed_patterns) if use_curriculum else None
    rows: list[dict[str, Any]] = []
    for i in range(n_episodes):
        if curriculum is not None:
            pattern, difficulty, ep_seed, _ = curriculum.next_episode()
            env.difficulty = difficulty
            env.generator.rng.seed(ep_seed)
            obs = env.reset(forced_patterns=[pattern])
        else:
            obs = env.reset()
        gt = env.ground_truth()
        rows.append(
            {
                "prompt": obs.prompt,
                "pr_diff": obs.pr_diff,
                "ground_truth": gt,
                "exec_result": env._state.exec_result.as_dict(),  # type: ignore[union-attr]
                "n_bugs": obs.n_bugs,
                "patterns": gt.get("patterns", []),
            }
        )
        if curriculum is not None:
            curriculum.record_result(
                pattern=gt.get("patterns", ["?"])[0],
                difficulty=env.difficulty,
                seed=ep_seed,
                reward=0.0,
                root_cause_score=0.5,
            )
    return rows


# ── V2 output schema: JSON keys the model must produce ───────────────────────
# The V2 prompt asks for a JSON object, NOT KEY: value lines. The scaffold
# checks for JSON key strings (e.g. '"verdict"') not V1 markers ("VERDICT:").
_V2_JSON_KEYS = ('"verdict"', '"root_cause"', '"failure_path"', '"confidence"', '"reasoning"')

# Error types per bug pattern — used in gold responses
_V2_GOLD_ERRORS: dict[str, str] = {
    "rename":         "AttributeError",
    "partial_rename": "AttributeError",
    "removal":        "ModuleNotFoundError",
    "contract":       "TypeError",
    "null_missing":   "AttributeError",
    "type_mismatch":  "TypeError",
    "condition_flip": "AssertionError",
    "off_by_one":     "IndexError",
}


def _gold_response_v2(row: dict) -> str:
    """Gold JSON response for a V2 dataset row — used only during SFT warmup.

    Builds a schema-correct JSON string from the ground_truth dict so the SFT
    warmup teaches the model the JSON output format before GRPO starts. The
    content is grounded in ground_truth so it is genuinely correct, not random.
    """
    gt = row.get("ground_truth", {})
    verdict = gt.get("verdict", "REQUEST_CHANGES")
    root_cause = gt.get("root_cause", "")
    failure_path = gt.get("failure_path", [])
    patterns = row.get("patterns", [])
    pattern = patterns[0] if patterns else "rename"
    error = _V2_GOLD_ERRORS.get(pattern, "RuntimeError")
    symbol = root_cause.split("::")[-1] if "::" in root_cause else root_cause
    reasoning = (
        f"{symbol} is stale — calling it will raise {error} at runtime. "
        f"Update all call sites before merging."
        if root_cause
        else "No stale references detected. PR is consistent with the current codebase."
    )
    return json.dumps({
        "verdict": verdict,
        "root_cause": root_cause,
        "failure_path": failure_path[:3] if failure_path else [],
        "confidence": 0.92 if root_cause else 0.90,
        "reasoning": reasoning,
    })


def run_sft_warmup_v2(model, tokenizer, rows: list[dict], n_steps: int, lr: float = 2e-4) -> None:
    """SFT warmup for V2: teach JSON output format before GRPO.

    Without this, the base model (Qwen2.5-1.5B-Instruct) generates verbose
    natural language prose. parse_reviewer_output finds no JSON → MalformedPrediction
    → MALFORMED_PENALTY=-0.5 on every completion → std=0 → loss=0 forever.

    50 steps on gold examples is enough to collapse the model onto JSON output.
    GRPO then has signal to tune which JSON content is correct.
    """
    from datasets import Dataset
    import inspect

    log.info("SFT warmup V2: %d steps on %d rows", n_steps, len(rows))
    sft_data = []
    for row in rows[: min(len(rows), n_steps * 8)]:
        gold = _gold_response_v2(row)
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": gold},
        ]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = row["prompt"] + "\n\n" + gold
        sft_data.append({"text": text})

    if not sft_data:
        log.warning("SFT warmup V2 skipped — no rows available")
        return

    sft_dataset = Dataset.from_list(sft_data)
    try:
        from trl import SFTTrainer, SFTConfig
        sft_config = SFTConfig(
            max_steps=n_steps,
            learning_rate=lr,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            output_dir="/tmp/codedrift_v2_sft_warmup",
            logging_steps=max(1, n_steps // 5),
            report_to="none",
            max_seq_length=1536,
            dataset_text_field="text",
            save_strategy="no",
        )
        sft_kwargs = dict(model=model, args=sft_config, train_dataset=sft_dataset)
        sft_params = inspect.signature(SFTTrainer.__init__).parameters
        if "processing_class" in sft_params:
            sft_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in sft_params:
            sft_kwargs["tokenizer"] = tokenizer
        SFTTrainer(**sft_kwargs).train()
        log.info("SFT warmup V2 complete")
    except Exception as exc:
        log.warning("SFT warmup V2 failed (%s) — continuing without warmup", exc)


def _to_text(completion) -> str:
    """Extract plain text from whatever TRL version passes as a completion.

    TRL ≥0.15 with processing_class returns completions as message-dict lists
    [[{"role":"assistant","content":"..."}], ...]. Older TRL returns strings.
    Passing a raw dict to the scorer produces Python repr → no field markers
    found → MALFORMED_PENALTY (-0.5) on every sample → std=0 → zero gradient.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content", ""))
        return " ".join(
            str(m.get("content", m)) if isinstance(m, dict) else str(m)
            for m in completion
        )
    if isinstance(completion, dict):
        return str(completion.get("content", str(completion)))
    return str(completion)


def _format_scaffold(text: str) -> float:
    """Small additive bonus per JSON key present in the response (max +0.15).

    V2 expects a JSON object — check for '"verdict"', '"root_cause"' etc., NOT
    the V1 KEY:VALUE markers. Without this fix, the scaffold gave 0 for every
    V2 response even when the model was outputting partial JSON.

    +0.03 per key × 5 keys = +0.15 max. Moves malformed reward from −0.5 to
    −0.35 as the model learns to include more keys — giving GRPO a gradient
    to climb before content is correct.
    """
    return min(sum(0.03 for k in _V2_JSON_KEYS if k in text), 0.15)


def make_reward_fn(
    metrics: MetricsLogger,
    log_step: dict[str, int],
):
    import statistics as _stats

    scorer = CausalScorer()

    def reward_fn(
        prompts,
        completions,
        ground_truth=None,
        exec_result=None,
        pr_diff=None,
        patterns=None,
        **kwargs,
    ):
        from agents_v2.reviewer_io import parse_reviewer_output

        n = len(completions)
        rewards: list[float] = []
        for i, raw_completion in enumerate(completions):
            # Fix: extract plain text regardless of TRL completion format
            completion_text = _to_text(raw_completion)

            gt = ground_truth[i] if ground_truth is not None else {}
            er_dict = exec_result[i] if exec_result is not None else {}
            diff = pr_diff[i] if pr_diff is not None else ""
            pat = (patterns[i] if patterns is not None else []) or ["unknown"]
            er = _exec_result_from_dict(er_dict if isinstance(er_dict, dict) else {})
            prediction = parse_reviewer_output(completion_text)
            base_reward, info = scorer.score(
                prediction=prediction,
                ground_truth=gt if isinstance(gt, dict) else {},
                exec_result=er,
                mutations=[],
                pr_diff=diff,
            )
            # Format scaffold: gradient bridge from malformed → correct format
            scaffold = _format_scaffold(completion_text)
            reward = float(base_reward) + scaffold

            rewards.append(reward)
            comps = info.get("reward_components", {})
            metrics.log(
                step=log_step["step"],
                payload={
                    "reward_total": reward,
                    "reward_base": float(base_reward),
                    "reward_scaffold": scaffold,
                    "reward_root_cause": comps.get("root_cause", 0.0),
                    "reward_failure_path": comps.get("failure_path", 0.0),
                    "reward_verdict": comps.get("verdict", 0.0),
                    "reward_hallucination": comps.get("hallucination", 0.0),
                    "reward_calibration": comps.get("calibration", 0.0),
                    "outcome": info.get("episode_outcome", "unknown"),
                    "pattern": pat[0],
                    "root_cause_score": float(info.get("breakdown", {}).get("root_cause", 0.0)),
                },
            )
            log_step["step"] += 1

        # Variance diagnostic — low std = GRPO can't learn
        if len(rewards) > 1:
            rvar = _stats.stdev(rewards)
            rmean = _stats.mean(rewards)
            warn = " *** LOW VARIANCE — increase --max_completion_length or --num_generations ***" if rvar < 0.05 else ""
            log.info(
                "reward_fn n=%d mean=%.3f std=%.3f min=%.3f max=%.3f%s",
                n, rmean, rvar, min(rewards), max(rewards), warn,
            )

        return rewards

    return reward_fn


def train(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics_v2.jsonl"
    # TRL's GRPOTrainer already does report_to="wandb" when --wandb. A second
    # wandb.log(..., step=) from MetricsLogger used per-completion indices (0,1,2,…)
    # which violates monotonicity vs the trainer's global step — disable W&B here.
    # Per-component reward rows remain in metrics_v2.jsonl.
    metrics = MetricsLogger(
        output_path=metrics_path,
        use_wandb=False,
        extra={
            "model": args.model,
            "patterns": TRAIN_PATTERNS,
            "episodes": args.episodes,
            "steps": args.steps,
        },
    )

    print(f"\nCodeDrift V2 training\n  model: {args.model}\n  episodes: {args.episodes}\n  steps: {args.steps}")
    rows = build_dataset(
        n_episodes=args.episodes,
        seed=args.seed,
        allowed_patterns=TRAIN_PATTERNS,
        use_curriculum=not bool(args.no_curriculum),
    )
    eval_split_path = out_dir / "v2_dataset.jsonl"
    eval_split_path.write_text(
        "\n".join(json.dumps({k: r[k] for k in ("prompt", "patterns", "n_bugs")}) for r in rows),
        encoding="utf-8",
    )
    print(f"dataset: {len(rows)} episodes  written to {eval_split_path}")

    if args.dry_run:
        print("dry run — skipping TRL trainer instantiation")
        metrics.close()
        return

    # Lazy imports so the module stays importable on machines without GPU stack.
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    import inspect

    from training.train import load_model_and_tokenizer  # reuse v1 loader + dtype guards

    model, tokenizer = load_model_and_tokenizer(args.model, backend=args.backend, seed=args.seed)

    # SFT warmup: teach JSON output format before GRPO starts.
    # Without this, base model outputs verbose prose → parse_reviewer_output returns
    # MalformedPrediction → MALFORMED_PENALTY=-0.5 on every completion → std=0 → loss=0.
    if args.sft_warmup_steps > 0:
        print(f"\nRunning V2 SFT warmup ({args.sft_warmup_steps} steps)...")
        run_sft_warmup_v2(model, tokenizer, rows, n_steps=args.sft_warmup_steps)
    trainable_params = sum(p.numel() for p in model.parameters() if getattr(p, "requires_grad", False))
    if trainable_params <= 0:
        raise RuntimeError(
            "No trainable parameters detected (0). LoRA adapters were not attached or are frozen. "
            "Stopping early to avoid a wasted run. Try --backend hf and verify model.print_trainable_parameters()."
        )
    log.info("trainable_params=%s", trainable_params)
    dataset = Dataset.from_list(rows)

    log_step = {"step": 0}
    reward_fn = make_reward_fn(metrics, log_step)

    # Unsloth fast inference + GRPO generate() can mix bf16 activations with fp32
    # weights in attention -> "mat1 bfloat16 != mat2 float". hf backend is safe with bf16.
    use_bf16 = bool(args.bf16)
    if use_bf16 and args.backend == "unsloth":
        log.warning(
            "bf16 is disabled for GRPO when using --backend unsloth (mixed dtypes in "
            "Unsloth fast generate). Re-run with --backend hf if you need bf16."
        )
        use_bf16 = False

    config = GRPOConfig(
        output_dir=str(out_dir),
        max_steps=args.steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to="wandb" if args.wandb else "none",
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        bf16=use_bf16,
    )
    params = inspect.signature(GRPOTrainer.__init__).parameters
    trainer_kwargs = dict(model=model, reward_funcs=reward_fn, train_dataset=dataset, args=config)
    if "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)
    print("training V2...\n")
    trainer.train()
    final_dir = out_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"\nsaved adapter to {final_dir}")
    metrics.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CodeDrift V2 GRPO trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--backend", choices=["hf", "unsloth"], default="hf")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    # num_generations ≥ 4 required: 2 completions with identical reward → std=0 → zero gradient.
    p.add_argument("--num_generations", type=int, default=4, help="Rollouts per step — must be ≥4 for reward variance")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--logging_steps", type=int, default=5, help="Higher = less log I/O, faster")
    p.add_argument("--save_steps", type=int, default=200)
    # 1024 truncates prompts (~1600 tokens) — model never sees full diff+tests.
    # 2048 fits the full context with room to spare.
    p.add_argument("--max_prompt_length", type=int, default=2048)
    # 384 minimum: a full VERDICT+ROOT_CAUSE+FAILURE_PATH+CONFIDENCE+ISSUES+REASON
    # response needs 200-400 tokens. Below 256 every completion is truncated →
    # MALFORMED_PENALTY (-0.5) on all → std=0 → zero GRPO gradient.
    p.add_argument("--max_completion_length", type=int, default=384)
    # temperature 1.0: cold base model at 0.8 collapses to near-identical outputs → std=0.
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Use bf16 in GRPO (use with --backend hf; ignored for unsloth — see train_v2)",
    )
    p.add_argument("--output_dir", default="outputs/v2")
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument(
        "--no-curriculum",
        action="store_true",
        default=False,
        help="Build dataset with uniform random episodes (faster to generate)",
    )
    p.add_argument("--dry_run", action="store_true", default=False, help="Build dataset and exit")
    # SFT warmup: 0 = skip. Teach JSON format before GRPO so early rollouts
    # produce parseable output and reward variance is non-zero from step 1.
    p.add_argument("--sft_warmup_steps", type=int, default=50,
                   help="SFT format warmup steps before GRPO (0 to skip)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
