"""V2 generalization eval — measure transfer to held-out bug patterns.

The trainer learns on TRAIN_PATTERNS. This script samples N episodes from
HELDOUT_PATTERNS and reports per-component reward + verdict accuracy. Use
with two policies (base vs LoRA) to compute a delta.

Modes:
- ``--policy oracle``: emits the ground-truth structured response. Useful as
  the upper bound for the scoring pipeline.
- ``--policy approve``: always APPROVE — naive baseline.
- ``--policy reject``: always REQUEST_CHANGES with empty root_cause — verdict
  baseline.
- ``--policy llm``: load a HF model (optionally with a LoRA adapter) and
  run greedy generation on each prompt. Falls back to ``approve`` if the
  HF stack is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Callable

if __name__ == "__main__" and sys.platform == "win32" and not getattr(sys.flags, "utf8_mode", False):
    os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents_v2.generator_agent import ALL_PATTERNS  # noqa: E402
from env_v2.exec_arena_env import CodeReviewArenaEnv  # noqa: E402
from utils_v2.metrics import MetricsLogger  # noqa: E402

TRAIN_PATTERNS = ["rename", "removal", "contract", "partial_rename", "null_missing"]
HELDOUT_PATTERNS = ["type_mismatch", "condition_flip", "off_by_one"]


def policy_approve(prompt: str, ground_truth: dict) -> str:
    return json.dumps(
        {"verdict": "APPROVE", "root_cause": "", "failure_path": [], "confidence": 0.5}
    )


def policy_reject(prompt: str, ground_truth: dict) -> str:
    return json.dumps(
        {
            "verdict": "REQUEST_CHANGES",
            "root_cause": "",
            "failure_path": [],
            "confidence": 0.5,
        }
    )


def policy_oracle(prompt: str, ground_truth: dict) -> str:
    return json.dumps(
        {
            "verdict": ground_truth.get("verdict", "REQUEST_CHANGES"),
            "root_cause": ground_truth.get("root_cause", ""),
            "failure_path": list(ground_truth.get("failure_path", [])),
            "confidence": 0.9,
            "reasoning": "oracle baseline",
        }
    )


def make_llm_policy(model_id: str, adapter_dir: str | None) -> Callable[[str, dict], str]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
        )
        if adapter_dir:
            from peft import PeftModel  # type: ignore

            model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()
    except Exception as exc:
        print(f"[warn] llm policy unavailable ({exc}); falling back to approve")
        return policy_approve

    def _policy(prompt: str, ground_truth: dict) -> str:
        import torch  # local

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text

    return _policy


POLICIES: dict[str, Callable[[str, dict], str]] = {
    "approve": policy_approve,
    "reject": policy_reject,
    "oracle": policy_oracle,
}


def evaluate(
    n_episodes: int,
    policy: Callable[[str, dict], str],
    seed: int,
    patterns: list[str],
    out_jsonl: Path | None,
) -> dict:
    env = CodeReviewArenaEnv(difficulty="easy", seed=seed, allowed_patterns=patterns)
    metrics = MetricsLogger(output_path=out_jsonl) if out_jsonl else None
    rewards: list[float] = []
    verdict_correct = 0
    root_correct = 0
    by_pattern: dict[str, list[float]] = {p: [] for p in patterns}
    for i in range(n_episodes):
        pat = patterns[i % len(patterns)]
        obs = env.reset(forced_patterns=[pat])
        gt = env.ground_truth()
        response = policy(obs.prompt, gt)
        _, reward, _, info = env.step(response)
        rewards.append(reward)
        components = info.get("reward_components", {})
        if info.get("pred_verdict") == info.get("gt_verdict"):
            verdict_correct += 1
        if info.get("breakdown", {}).get("root_cause", 0.0) >= 0.7:
            root_correct += 1
        by_pattern[pat].append(reward)
        if metrics is not None:
            metrics.log(
                step=i,
                payload={
                    "episode_id": obs.episode_id,
                    "pattern": pat,
                    "reward_total": reward,
                    "reward_root_cause": components.get("root_cause", 0.0),
                    "reward_failure_path": components.get("failure_path", 0.0),
                    "reward_verdict": components.get("verdict", 0.0),
                    "reward_hallucination": components.get("hallucination", 0.0),
                    "reward_calibration": components.get("calibration", 0.0),
                    "outcome": info.get("episode_outcome", "unknown"),
                    "root_cause_score": info.get("breakdown", {}).get("root_cause", 0.0),
                },
            )
    if metrics is not None:
        metrics.close()
    summary = {
        "n_episodes": n_episodes,
        "mean_reward": round(statistics.mean(rewards), 4) if rewards else 0.0,
        "verdict_accuracy": round(verdict_correct / n_episodes, 4) if n_episodes else 0.0,
        "root_cause_accuracy": round(root_correct / n_episodes, 4) if n_episodes else 0.0,
        "per_pattern_mean": {
            p: round(statistics.mean(v), 4) if v else 0.0 for p, v in by_pattern.items()
        },
    }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V2 generalization eval")
    p.add_argument("--policy", choices=list(POLICIES.keys()) + ["llm"], default="oracle")
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--patterns",
        choices=["heldout", "train", "all"],
        default="heldout",
        help="Which pattern set to sample from.",
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct", help="HF model id (llm only)")
    p.add_argument("--adapter", default="", help="LoRA adapter dir (llm only)")
    p.add_argument("--out", default="outputs/v2_eval.jsonl")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.patterns == "heldout":
        patterns = HELDOUT_PATTERNS
    elif args.patterns == "train":
        patterns = TRAIN_PATTERNS
    else:
        patterns = ALL_PATTERNS
    if args.policy == "llm":
        policy = make_llm_policy(args.model, args.adapter or None)
    else:
        policy = POLICIES[args.policy]
    summary = evaluate(
        n_episodes=args.episodes,
        policy=policy,
        seed=args.seed,
        patterns=patterns,
        out_jsonl=Path(args.out),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
