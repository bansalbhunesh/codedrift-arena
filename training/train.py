"""
CodeDrift Arena — GRPO Training Script
Model:   Qwen2.5-1.5B-Instruct  (fits T4 free Colab, ~8GB VRAM)
Trainer: HuggingFace TRL GRPOTrainer
Quant:   Unsloth 4-bit LoRA
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from codedrift.logutil import configure_logging
from env.codedrift_env import CodeDriftEnv
from integrations import config as cfg


def build_dataset(n_episodes: int, difficulty: str, personality: str, seed: int = 42):
    """
    Pre-generate episodes offline so training is fast.
    Each row: prompt + ground truth stale_refs (passed as kwargs to reward fn).
    """
    from datasets import Dataset

    rows = []
    env = CodeDriftEnv(difficulty=difficulty, personality=personality, seed=seed)

    for _ in range(n_episodes):
        obs = env.reset()

        serialized_actions = [
            {
                "drift_type": a.drift_type,
                "stale_ref": a.stale_ref,
                "current_ref": a.current_ref,
                "metadata": a.metadata,
            }
            for a in env.stale_actions
        ]

        rows.append(
            {
                "prompt": obs.prompt,
                "pr_diff": obs.pr_diff,
                "serialized_actions": json.dumps(serialized_actions),
                "n_stale_refs": obs.n_stale_refs,
            }
        )

    clean_env = CodeDriftEnv(difficulty="easy", personality="random")
    clean_pr = (
        "diff --git a/src/clean.py b/src/clean.py\n"
        "+++ b/src/clean.py\n"
        "+from models.user import User\n"
        "+user = User.get(user_id)\n"
        "+return user.to_dict()"
    )
    for _ in range(max(1, n_episodes // 5)):
        obs = clean_env.set_clean_episode(clean_pr)
        rows.append(
            {
                "prompt": obs.prompt,
                "pr_diff": obs.pr_diff,
                "serialized_actions": json.dumps([]),
                "n_stale_refs": 0,
            }
        )

    import random

    rng = random.Random(seed)
    rng.shuffle(rows)

    return Dataset.from_list(rows)


def make_reward_fn(difficulty: str):
    from agents.drift_agent import DriftAction
    from rewards.scorer import RewardScorer

    _ = difficulty  # reserved for curriculum / logging
    scorer = RewardScorer()
    log = logging.getLogger("codedrift.train.reward")

    def reward_fn(prompts, completions, serialized_actions=None, pr_diff=None, **kwargs):
        rewards = []
        n = len(completions)
        if serialized_actions is not None and len(serialized_actions) != n:
            raise ValueError(
                f"serialized_actions length {len(serialized_actions)} != completions {n}"
            )
        if pr_diff is not None and len(pr_diff) != n:
            raise ValueError(f"pr_diff length {len(pr_diff)} != completions {n}")
        for i, completion in enumerate(completions):
            try:
                raw = serialized_actions[i] if serialized_actions is not None else "[]"
                if isinstance(raw, str):
                    action_dicts = json.loads(raw)
                elif isinstance(raw, list):
                    action_dicts = raw
                else:
                    action_dicts = json.loads(str(raw))

                actions = [
                    DriftAction(
                        drift_type=d["drift_type"],
                        stale_ref=d["stale_ref"],
                        current_ref=d["current_ref"],
                        metadata=d.get("metadata") or {},
                    )
                    for d in action_dicts
                ]

                diff = pr_diff[i] if pr_diff is not None else ""
                reward, info = scorer.score(
                    agent_response=completion,
                    actions=actions,
                    pr_diff=diff,
                )
                log.debug(
                    "reward_fn row=%s reward=%.3f recall=%.2f outcome=%s verdict=%s",
                    i,
                    reward,
                    float(info.get("recall", 0.0)),
                    info.get("episode_outcome"),
                    info.get("verdict"),
                )
                rewards.append(float(reward))
            except Exception:
                log.exception("reward_fn_row_failed batch_index=%s", i)
                # Match max per-action miss penalty scale (~[-3, +3]) so curves stay interpretable.
                rewards.append(-1.0)

        assert len(rewards) == n, (
            f"reward_fn length mismatch: got {len(rewards)} rewards for batch size {n}"
        )
        return rewards

    return reward_fn


def _instantiate_grpo_trainer(GRPOTrainer, *, model, tokenizer, config, train_dataset, reward_fn):
    """Support TRL API changes (``args`` / ``processing_class`` vs legacy ``config`` / ``tokenizer``)."""
    import inspect

    params = inspect.signature(GRPOTrainer.__init__).parameters
    kwargs = dict(model=model, reward_funcs=reward_fn, train_dataset=train_dataset)
    if "args" in params:
        kwargs["args"] = config
    elif "config" in params:
        kwargs["config"] = config
    else:
        kwargs["args"] = config
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    else:
        raise TypeError(
            "GRPOTrainer has neither processing_class nor tokenizer; "
            "upgrade/downgrade trl or extend _instantiate_grpo_trainer."
        )
    return GRPOTrainer(**kwargs)


def train(args):
    configure_logging()
    try:
        from trl import GRPOConfig, GRPOTrainer
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: Install training deps: pip install -r requirements-train.txt")
        sys.exit(1)

    print("\nCodeDrift Arena Training")
    print(f"  Model:       {args.model}")
    print(f"  Difficulty:  {args.difficulty}")
    print(f"  Personality: {args.personality}")
    print(f"  Steps:       {args.steps}")
    print(f"  Episodes:    {args.episodes}")
    print(f"  HF dataset:  {cfg.HF_DATASET_REPO or '(unset — set CODEDRIFT_HF_DATASET_REPO)'}")
    print(f"  HF model:    {cfg.HF_MODEL_REPO or '(unset — set CODEDRIFT_HF_MODEL_REPO)'}")
    print(f"  W&B project: {cfg.WANDB_PROJECT or '(unset)'}\n")

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=True,
    )

    print("Building dataset...")
    dataset = build_dataset(
        n_episodes=args.episodes,
        difficulty=args.difficulty,
        personality=args.personality,
        seed=args.seed,
    )
    print(f"Dataset: {len(dataset)} episodes")

    reward_fn = make_reward_fn(args.difficulty)

    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        warmup_steps=20,
        logging_steps=10,
        save_steps=100,
        report_to="none",
        num_generations=4,
        max_prompt_length=1024,
        max_completion_length=256,
        temperature=0.8,
    )

    trainer = _instantiate_grpo_trainer(
        GRPOTrainer,
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=dataset,
        reward_fn=reward_fn,
    )

    print("Training...\n")
    trainer.train()

    out = Path(args.output_dir)
    model.save_pretrained(str(out / "final"))
    tokenizer.save_pretrained(str(out / "final"))
    print(f"\nSaved to {out / 'final'}")


def parse_args():
    p = argparse.ArgumentParser(description="CodeDrift Arena GRPO Training")
    p.add_argument("--model", default=cfg.DEFAULT_MODEL_ID)
    p.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"])
    p.add_argument("--personality", default="random", choices=["random", "subtle", "aggressive", "escalating"])
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=cfg.OUTPUT_DIR)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
