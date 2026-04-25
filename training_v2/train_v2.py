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


def make_reward_fn(
    metrics: MetricsLogger,
    log_step: dict[str, int],
):
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
        for i, completion in enumerate(completions):
            gt = ground_truth[i] if ground_truth is not None else {}
            er_dict = exec_result[i] if exec_result is not None else {}
            diff = pr_diff[i] if pr_diff is not None else ""
            pat = (patterns[i] if patterns is not None else []) or ["unknown"]
            er = _exec_result_from_dict(er_dict if isinstance(er_dict, dict) else {})
            prediction = parse_reviewer_output(completion)
            reward, info = scorer.score(
                prediction=prediction,
                ground_truth=gt if isinstance(gt, dict) else {},
                exec_result=er,
                mutations=[],
                pr_diff=diff,
            )
            rewards.append(float(reward))
            comps = info.get("reward_components", {})
            metrics.log(
                step=log_step["step"],
                payload={
                    "reward_total": float(reward),
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
    p.add_argument("--num_generations", type=int, default=4, help="Rollouts per step (lower = faster)")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--warmup_steps", type=int, default=10)
    p.add_argument("--logging_steps", type=int, default=5, help="Higher = less log I/O, faster")
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--max_completion_length", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
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
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
