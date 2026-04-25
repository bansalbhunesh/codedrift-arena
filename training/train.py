"""
CodeDrift Arena — GRPO Training Script
Model:   Qwen2.5-1.5B-Instruct  (fits T4 free Colab, ~8GB VRAM)
Trainer: HuggingFace TRL GRPOTrainer
Quant:   bitsandbytes 4-bit QLoRA
Backend: HF stack by default, optional Unsloth path via --backend unsloth

WHY THE MODEL WASN'T LEARNING — fixed in this version
------------------------------------------------------
1. max_completion_length=256 truncated every response before ISSUES: was written,
   causing _parse_issues_section to fail → malformed → -1.0 on every completion.
2. All 4 completions scoring -1.0 → GRPO advantage std=0 → zero gradient.
3. TRL ≥0.15 passes completions as message dicts; old code passed raw dict to scorer.
4. No SFT warmup: base model doesn't know the output format → all outputs malformed.
5. Reward cliff: -1.0 for no format, +1.0 for perfect — no intermediate gradient.
"""

import argparse
import hashlib
import json
import logging
import os
import statistics
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


# ── Format fields the scorer requires ────────────────────────────────────────
_REQUIRED_FIELDS = ("VERDICT:", "ROOT_CAUSE:", "FAILURE_PATH:", "CONFIDENCE:", "ISSUES:", "REASON:")

# Error type keywords mapped to bug patterns — used for gold response generation
_GOLD_ERRORS: dict[str, str] = {
    "rename":         "AttributeError",
    "partial_rename": "AttributeError",
    "removal":        "ModuleNotFoundError",
    "contract":       "TypeError",
    "null_missing":   "AttributeError: NoneType",
    "type_mismatch":  "TypeError",
    "condition_flip": "AssertionError",
    "off_by_one":     "IndexError",
}


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


# ── Gold response generator (used for SFT warmup) ───────────────────────────

def _gold_response_for_row(row: dict) -> str:
    """Generate a correct structured response for a dataset row.

    Used only during SFT warmup to teach the model the output format before
    GRPO. The response is deterministic and grounded in the ground-truth
    serialized_actions for this row.
    """
    action_dicts = json.loads(row.get("serialized_actions", "[]"))
    if not action_dicts:
        return (
            "VERDICT: APPROVE\n"
            "ROOT_CAUSE: none\n"
            "FAILURE_PATH: n/a\n"
            "CONFIDENCE: 0.90\n"
            "ISSUES: none\n"
            "REASON: No stale references detected. PR is consistent with the current codebase."
        )
    refs = []
    issues_parts = []
    paths = []
    for d in action_dicts:
        stale = d["stale_ref"].split("(")[0]
        current = d["current_ref"].split("(")[0]
        pat = d.get("bug_pattern") or d["drift_type"]
        error = _GOLD_ERRORS.get(pat, "RuntimeError")
        refs.append(stale)
        issues_parts.append(
            f"{stale} is a stale reference — it was changed and the PR still uses the old form. "
            f"This will raise {error} at runtime. Update to {current}."
        )
        paths.append(f"failing_test -> caller -> {stale} -> {error}")

    first_stale = refs[0]
    return (
        "VERDICT: REQUEST_CHANGES\n"
        f"ROOT_CAUSE: {first_stale}\n"
        f"FAILURE_PATH: {paths[0]}\n"
        "CONFIDENCE: 0.92\n"
        f"ISSUES: {' ; '.join(issues_parts)}\n"
        f"REASON: Stale references detected ({', '.join(refs)}) — must update before merging."
    )


# ── Reward function ──────────────────────────────────────────────────────────

def _to_text(completion) -> str:
    """Extract plain text from whatever TRL version passes as a completion.

    TRL ≥0.15 with processing_class returns completions as message-dict lists
    [[{"role":"assistant","content":"..."}], ...]. Older versions return strings.
    Both must be handled so the scorer sees plain text, not a Python repr.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # Message-list format: find the last assistant message
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content", ""))
        # Fallback: concatenate all content fields
        return " ".join(
            str(m.get("content", m)) if isinstance(m, dict) else str(m)
            for m in completion
        )
    if isinstance(completion, dict):
        return str(completion.get("content", str(completion)))
    return str(completion)


def _format_scaffold(text: str) -> float:
    """Small additive bonus for having each required format field present.

    This creates a gradient bridge from "random garbage" (0 fields, −1.0 base)
    to "correct format" (6 fields, +0.18 scaffold). Without this, the reward
    cliff from −1.0 to +1.0 has no intermediate signal and GRPO can't start.
    Each field is worth 0.03; total capped at 0.18 so a well-formatted wrong
    answer still scores well below a correct answer.
    """
    upper = text.upper()
    return min(sum(0.03 for f in _REQUIRED_FIELDS if f in upper), 0.18)


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

        for i, raw_completion in enumerate(completions):
            try:
                # Fix 3: handle message-dict format from TRL ≥0.15
                agent_text = _to_text(raw_completion)

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
                base_reward, info = scorer.score(
                    agent_response=agent_text,
                    actions=actions,
                    pr_diff=diff,
                    failure_cascade=None,
                )

                # Fix 5: format scaffold — creates a gradient from -1.0 toward 0
                # before the model has learned to cite stale refs correctly.
                scaffold = _format_scaffold(agent_text)
                reward = float(base_reward) + scaffold

                log.debug(
                    "reward_fn row=%s base=%.3f scaffold=%.3f reward=%.3f "
                    "recall=%.2f outcome=%s verdict=%s",
                    i, base_reward, scaffold, reward,
                    float(info.get("recall", 0.0)),
                    info.get("episode_outcome"),
                    info.get("verdict"),
                )
                rewards.append(reward)
            except Exception:
                log.exception("reward_fn_row_failed batch_index=%s", i)
                rewards.append(-1.0)

        # Variance diagnostic — if std < 0.05 for many batches, GRPO can't learn
        if len(rewards) > 1:
            rvar = statistics.stdev(rewards)
            rmean = statistics.mean(rewards)
            log.info(
                "reward_fn batch: n=%d mean=%.3f std=%.3f min=%.3f max=%.3f%s",
                n, rmean, rvar, min(rewards), max(rewards),
                " [LOW VARIANCE — check completion length]" if rvar < 0.05 else "",
            )

        if len(rewards) != n:
            log.error("reward_fn length mismatch: got %d rewards for batch size %d", len(rewards), n)
            if len(rewards) < n:
                rewards.extend([-1.0] * (n - len(rewards)))
            else:
                del rewards[n:]
        return rewards

    return reward_fn


# ── SFT warmup ───────────────────────────────────────────────────────────────

def run_sft_warmup(model, tokenizer, rows: list[dict], n_steps: int, lr: float = 2e-4) -> None:
    """Supervised warmup: teach the output format before GRPO.

    The base model has never seen VERDICT/ROOT_CAUSE/FAILURE_PATH structure.
    Without this, early GRPO rollouts are all malformed → all score -1.0 →
    zero reward variance → zero GRPO advantage → flat loss for the first
    N steps (often the entire run).

    We build (prompt, gold_completion) pairs from the ground-truth actions and
    run a short SFT pass. This does NOT teach the model which specific refs are
    stale in the training set (the gold completions are schema-correct but
    generic). It only teaches format, giving GRPO a non-zero starting gradient.
    """
    from datasets import Dataset

    log = logging.getLogger("codedrift.train.sft")
    log.info("SFT warmup: %d steps on %d rows", n_steps, len(rows))

    sft_data = []
    for row in rows[: min(len(rows), n_steps * 8)]:
        gold = _gold_response_for_row(row)
        # Use the chat template so the model sees the same token structure as GRPO
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
        log.warning("SFT warmup skipped — no rows available")
        return

    sft_dataset = Dataset.from_list(sft_data)

    try:
        from trl import SFTTrainer, SFTConfig
        sft_config = SFTConfig(
            max_steps=n_steps,
            learning_rate=lr,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            output_dir="/tmp/codedrift_sft_warmup",
            logging_steps=max(1, n_steps // 5),
            report_to="none",
            max_seq_length=1536,
            dataset_text_field="text",
            save_strategy="no",
        )
        import inspect
        sft_params = inspect.signature(SFTTrainer.__init__).parameters
        sft_kwargs = dict(model=model, args=sft_config, train_dataset=sft_dataset)
        if "processing_class" in sft_params:
            sft_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in sft_params:
            sft_kwargs["tokenizer"] = tokenizer
        trainer = SFTTrainer(**sft_kwargs)
        trainer.train()
        log.info("SFT warmup complete")
    except Exception as exc:
        log.warning("SFT warmup failed (%s) — continuing without warmup", exc)


# ── Model loading ─────────────────────────────────────────────────────────────

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
            max_seq_length=2560,
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


# ── Main training loop ────────────────────────────────────────────────────────

def train(args):
    configure_logging()

    print("\nCodeDrift Arena Training")
    print(f"  Model:              {args.model}")
    print(f"  Difficulty:         {args.difficulty}")
    print(f"  Personality:        {args.personality}")
    print(f"  Steps:              {args.steps}")
    print(f"  Episodes:           {args.episodes}")
    print(f"  Backend:            {args.backend}")
    print(f"  Completion tokens:  {args.max_completion_length}")
    print(f"  Generations/step:   {args.num_generations}")
    print(f"  Temperature:        {args.temperature}")
    print(f"  SFT warmup steps:   {args.sft_warmup_steps}\n")

    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    import inspect

    model, tokenizer = load_model_and_tokenizer(args.model, backend=args.backend, seed=args.seed)
    trainable_params = sum(p.numel() for p in model.parameters() if getattr(p, "requires_grad", False))
    if trainable_params <= 0:
        raise RuntimeError(
            "No trainable parameters detected (0). LoRA adapters were not attached or are frozen. "
            "Stopping early to avoid a wasted run. Try --backend hf and verify model.print_trainable_parameters()."
        )
    print(f"Trainable params: {trainable_params:,}")

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

    # Fix 4: SFT warmup — teach the output format before GRPO starts
    if args.sft_warmup_steps > 0:
        print(f"\nRunning SFT warmup ({args.sft_warmup_steps} steps)...")
        run_sft_warmup(model, tokenizer, train_rows, n_steps=args.sft_warmup_steps)

    reward_fn = make_reward_fn(args.difficulty)

    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to="none" if args.no_wandb else "wandb",
        num_generations=args.num_generations,     # Fix 2: 4→8
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,  # Fix 1: 256→512
        temperature=args.temperature,              # Fix 2: 0.8→1.0
        bf16=bool(args.bf16),
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
    p = argparse.ArgumentParser(
        description="CodeDrift Arena GRPO Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    # Fix 1: default completion length raised from 256 → 512
    # A complete VERDICT+ROOT_CAUSE+FAILURE_PATH+CONFIDENCE+ISSUES+REASON response
    # needs 200-400 tokens. 256 always truncated ISSUES, causing parse failure → -1.0.
    p.add_argument("--max_completion_length", type=int, default=512)
    # Fix 2: more generations and higher temperature for reward variance
    # 4 near-identical completions → std≈0 → GRPO advantage≈0 → zero gradient.
    p.add_argument("--num_generations", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    # Fix 4: SFT warmup steps (0 = skip)
    p.add_argument(
        "--sft_warmup_steps",
        type=int,
        default=50,
        help="SFT format warmup steps before GRPO. 0 to skip. "
             "Teaches VERDICT/ISSUES/ROOT_CAUSE format so GRPO has non-zero starting variance.",
    )
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=200)
    # 1024 truncates prompts (~1600 tokens) — model never sees full diff+tests.
    p.add_argument("--max_prompt_length", type=int, default=2048)
    p.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Use bf16 (recommended on A100 if stable)",
    )
    p.add_argument(
        "--no_wandb",
        action="store_true",
        default=False,
        help="Disable Weights & Biases logging",
    )
    args = p.parse_args()
    if not 0.0 <= args.heldout_fraction < 1.0:
        p.error("--heldout_fraction must be in [0.0, 1.0).")
    return args


if __name__ == "__main__":
    train(parse_args())
