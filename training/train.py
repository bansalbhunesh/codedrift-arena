"""
CodeDrift Arena — GRPO Training Script
Model:   Qwen2.5-1.5B-Instruct  (fits T4 free Colab, ~8GB VRAM)
Trainer: HuggingFace TRL GRPOTrainer
Quant:   bitsandbytes 4-bit QLoRA
Backend: HF stack by default, optional Unsloth path via --backend unsloth
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

# Windows: default text encoding is often cp1252; TRL reads UTF-8 .jinja templates
# with pathlib and crashes with UnicodeDecodeError. Re-exec once with UTF-8 mode.
if __name__ == "__main__" and sys.platform == "win32" and not getattr(sys.flags, "utf8_mode", False):
    os.execv(sys.executable, [sys.executable, "-X", "utf8", *sys.argv])

sys.path.insert(0, str(Path(__file__).parent.parent))

from codedrift.logutil import configure_logging
from env.codedrift_env import CodeDriftEnv
from integrations import config as cfg


def _serialize_actions(actions) -> list[dict]:
    return [
        {
            "drift_type": a.drift_type,
            "stale_ref": a.stale_ref,
            "current_ref": a.current_ref,
            "metadata": a.metadata,
            "bug_pattern": getattr(a, "bug_pattern", ""),
        }
        for a in actions
    ]


def _row_primary_stale_key(row: dict) -> str:
    actions = json.loads(row.get("serialized_actions", "[]"))
    if not actions:
        return "clean:none"
    first = actions[0]
    stale_ref = str(first.get("stale_ref", "")).split("(")[0]
    return f"{first.get('drift_type', 'unknown')}:{stale_ref}"


def _stable_holdout_bucket(key: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def split_train_eval_rows(rows: list[dict], holdout_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Deterministically hold out stale keys for basic memorization checks."""
    train_rows: list[dict] = []
    eval_rows: list[dict] = []
    for row in rows:
        key = _row_primary_stale_key(row)
        if key == "clean:none":
            train_rows.append(row)
            continue
        bucket = _stable_holdout_bucket(key, seed)
        if bucket < holdout_fraction:
            eval_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, eval_rows


def build_dataset_rows(n_episodes: int, difficulty: str, personality: str, seed: int = 42) -> list[dict]:
    """
    Pre-generate episodes offline so training is fast.
    Each row: prompt + ground truth stale_refs (passed as kwargs to reward fn).
    """
    rows = []
    env = CodeDriftEnv(difficulty=difficulty, personality=personality, seed=seed)

    for _ in range(n_episodes):
        obs = env.reset()
        serialized_actions = _serialize_actions(env.stale_actions)
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
    return rows


def make_reward_fn(difficulty: str):
    from agents.drift_agent import DriftAction
    from rewards.scorer import RewardScorer

    _ = difficulty
    scorer = RewardScorer()
    log = logging.getLogger("codedrift.train.reward")

    def reward_fn(prompts, completions, serialized_actions=None, pr_diff=None, **kwargs):
        rewards = []
        n = len(completions)
        if serialized_actions is not None and len(serialized_actions) != n:
            raise ValueError(f"serialized_actions length {len(serialized_actions)} != completions {n}")
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
                        bug_pattern=d.get("bug_pattern") or "",
                    )
                    for d in action_dicts
                ]

                diff = pr_diff[i] if pr_diff is not None else ""
                reward, info = scorer.score(
                    agent_response=completion,
                    actions=actions,
                    pr_diff=diff,
                    failure_cascade=None,
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
                rewards.append(-1.0)

        if len(rewards) != n:
            log.error("reward_fn length mismatch: got %d rewards for batch size %d", len(rewards), n)
            if len(rewards) < n:
                rewards.extend([-1.0] * (n - len(rewards)))
            else:
                del rewards[n:]
        return rewards

    return reward_fn


def load_model_and_tokenizer(model_name: str, backend: str = "hf", seed: int = 42):
    """
    Load model with 4-bit QLoRA.
    - backend="hf": stable transformers + bitsandbytes + peft path
    - backend="unsloth": FastLanguageModel path when available
    """
    if backend == "unsloth":
        try:
            from unsloth import FastLanguageModel  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Unsloth backend requested but package is unavailable. "
                "Install requirements-train.txt (includes unsloth) or use --backend hf."
            ) from exc

        print("Loading model/tokenizer via Unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=1280,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=seed,
        )
        model.print_trainable_parameters()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return model, tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import torch

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # bf16 not supported on T4
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    # Defensive dtype alignment: in some Colab/Transformers stacks, lm_head may stay fp32
    # while hidden states are bf16/fp16 during generate(), causing linear dtype mismatch.
    if hasattr(model, "lm_head") and hasattr(model.lm_head, "weight"):
        model.lm_head = model.lm_head.to(dtype=torch.float16)
    elif hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        lm_head = getattr(model.base_model.model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            model.base_model.model.lm_head = lm_head.to(dtype=torch.float16)
            # Last-resort guard for mixed-precision autocast paths in some Colab stacks.
            # If hidden states become bf16 but lm_head stays fp16/fp32, cast inputs to
            # lm_head weight dtype right before linear projection.
            if not hasattr(model.base_model.model.lm_head, "_codedrift_safe_forward"):
                def _safe_lm_head_forward(x):
                    weight_dtype = model.base_model.model.lm_head.weight.dtype
                    if x.dtype != weight_dtype:
                        x = x.to(dtype=weight_dtype)
                    return torch.nn.functional.linear(
                        x,
                        model.base_model.model.lm_head.weight,
                        model.base_model.model.lm_head.bias,
                    )

                model.base_model.model.lm_head.forward = _safe_lm_head_forward
                model.base_model.model.lm_head._codedrift_safe_forward = True

    model.print_trainable_parameters()

    return model, tokenizer


def train(args):
    configure_logging()

    print("\nCodeDrift Arena Training")
    print(f"  Model:       {args.model}")
    print(f"  Difficulty:  {args.difficulty}")
    print(f"  Personality: {args.personality}")
    print(f"  Steps:       {args.steps}")
    print(f"  Episodes:    {args.episodes}")
    print(f"  Backend:     {args.backend}\n")

    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    import inspect

    model, tokenizer = load_model_and_tokenizer(args.model, backend=args.backend, seed=args.seed)

    print("Building dataset...")
    rows = build_dataset_rows(
        n_episodes=args.episodes,
        difficulty=args.difficulty,
        personality=args.personality,
        seed=args.seed,
    )
    train_rows, eval_rows = split_train_eval_rows(rows, args.heldout_fraction, args.seed)
    dataset = Dataset.from_list(train_rows)
    print(f"Dataset: train={len(train_rows)} rows, heldout_eval={len(eval_rows)} rows")

    if args.eval_split_out:
        out = Path(args.eval_split_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(eval_rows, indent=2), encoding="utf-8")
        print(f"Wrote heldout split to {out}")

    reward_fn = make_reward_fn(args.difficulty)

    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        warmup_steps=10,
        logging_steps=5,
        save_steps=200,
        report_to="wandb",
        num_generations=4,
        max_prompt_length=1024,
        max_completion_length=256,
        temperature=0.8,
        bf16=False,
        # fp16 disabled: GRPOTrainer generates rollouts outside AMP's GradScaler
        # context, causing "No inf checks recorded" assertion in PyTorch 2.x.
        # bitsandbytes already computes in float16 via bnb_4bit_compute_dtype.
    )

    # Safely instantiate GRPOTrainer across TRL versions
    params = inspect.signature(GRPOTrainer.__init__).parameters
    trainer_kwargs = dict(model=model, reward_funcs=reward_fn, train_dataset=dataset)
    trainer_kwargs["args"] = config
    if "processing_class" in params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**trainer_kwargs)

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
    p.add_argument(
        "--personality",
        default="random",
        choices=["random", "subtle", "aggressive", "escalating", "adaptive"],
    )
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--backend", choices=["hf", "unsloth"], default="hf")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=cfg.OUTPUT_DIR)
    p.add_argument("--heldout_fraction", type=float, default=0.2)
    p.add_argument(
        "--eval_split_out",
        default="",
        help="Optional path to write held-out eval rows as JSON.",
    )
    args = p.parse_args()
    if not 0.0 <= args.heldout_fraction < 1.0:
        p.error("--heldout_fraction must be in [0.0, 1.0).")
    return args


if __name__ == "__main__":
    train(parse_args())
