# Bug Code Arena — From Idea to Executable RL Environment

*A behind-the-scenes look at what we built, why it works, and what it actually trains.*

---

## The Problem We Set Out To Solve

Most code review benchmarks ask a model: "Does this PR *look* correct?"

That question has a fundamental flaw — it relies on a human deciding what "correct" looks like and labeling it. Labels are expensive, subjective, and go stale the moment the codebase changes.

We asked a different question:

> "Will this PR **break** production?"

And the answer doesn't need a human. It needs a test suite.

---

## The Core Insight

Every time a codebase evolves — a function is renamed, a module removed, a signature changed — the existing tests *already know* something is broken. They fail. Loudly. With a traceback that points exactly to the stale reference.

That failure is **ground truth**. Not a label. Not a heuristic. An **executable fact**.

We built an environment that generates this situation artificially and at scale, then trains a model to be the engineer who catches it in review — before it ships.

---

## How It Works

### Step 1 — Plant a bug

The **Drift Agent** introduces a realistic code mutation into the codebase:

- Rename `getUserData` → `fetchUserProfile`
- Remove `utils/legacy.py` entirely
- Change `createOrder(item, qty)` → `createOrder(item, qty, user_id)` (new required param)
- Flip `include_deleted=True` → `include_deleted=False` semantics
- Switch pagination from 1-based to 0-based indexing

The PR is opened against the *old* world. It still calls `getUserData`. It still imports `legacy.py`.

### Step 2 — Run the tests

The test suite executes. It fails. The failure log is:

```
AttributeError: module 'users' has no attribute 'getUserData'
  test_profile_page → get_user_dashboard → getUserData
```

This is the **oracle**. No human decided that. The Python runtime decided that.

### Step 3 — The reviewer reads the evidence

The model sees:
- The PR diff (what the developer changed)
- The failing test output (what broke)
- The current codebase (where things stand now)

It must produce a **structured bug report**:

```
VERDICT: REQUEST_CHANGES
ROOT_CAUSE: getUserData
FAILURE_PATH: test_profile_page → get_user_dashboard → getUserData → AttributeError
CONFIDENCE: 0.92
ISSUES: getUserData was renamed to fetchUserProfile. This PR still uses the old
        name — will raise AttributeError at runtime.
REASON: Stale reference detected. Must update before merging.
```

### Step 4 — Score the answer

The **Reward Scorer** checks nine distinct things:

| Component | What it checks |
|---|---|
| **Catch** | Did you name the right stale reference? |
| **Root cause** | Did you identify *which* symbol is stale? |
| **Failure path** | Did you trace the call chain from test to crash? |
| **Verdict** | Did you REQUEST_CHANGES (not APPROVE)? |
| **Confidence** | Is your confidence calibrated, not just maxed out? |
| **Error type named** | Did you say "AttributeError" / "TypeError" / etc.? |
| **Hard pattern bonus** | Did you catch a condition flip or off-by-one? (+0.25) |
| **Completeness** | Did you catch *all* stale refs, not just the first? (+0.20) |
| **Format complete** | Is the response structured and parseable? |

**Missed stale refs subtract from the reward.** Hallucinated symbols (citing a ref that doesn't exist) also penalize.

This gives a **rich gradient signal** to GRPO — not just "right or wrong" but "how right, and in what way."

### Step 5 — Learn

GRPO (Group Relative Policy Optimization) updates the model based on the relative quality of multiple sampled answers per episode. High-scoring answers pull the policy up. Low-scoring answers pull it down. Over 200+ steps on a T4 GPU, the model learns to reason causally about code — not to match keywords.

---

## What the Environment Actually Trains

This is not a vague "code understanding" environment. Here are the **specific, measurable capabilities** it trains:

### 1. Stale reference detection
The model learns to spot the exact symbol (function name, module path, call signature) that no longer exists in the current codebase — even when the PR *looks* syntactically valid.

### 2. Causal failure path tracing
Given a test failure, the model learns to trace *backwards*: which test triggered the crash → which intermediate caller invoked it → which stale symbol caused it. This is **dependency-aware reasoning**, not keyword search.

### 3. Error type identification
The model learns which failure mode corresponds to which bug pattern:
- Rename → `AttributeError` (symbol doesn't exist)
- Removal → `ModuleNotFoundError` (module was deleted)
- Contract change → `TypeError` (missing required argument)
- Null missing → `AttributeError` on `None`
- Type mismatch → `TypeError` at runtime
- Condition flip → `AssertionError` / wrong behavior
- Off-by-one → `IndexError` or wrong page returned

### 4. Verdict calibration
The model learns not just to say REQUEST_CHANGES, but to mean it: when there are no stale refs, APPROVE is the right answer and gets rewarded. The model must discriminate.

### 5. Confidence calibration
Overconfident wrong answers are penalized. Uncertain correct answers are rewarded less than confident correct ones. The model learns to express belief, not just output.

### 6. Anti-hallucination
Citing a symbol that doesn't appear in the diff or codebase costs reward. The model learns to stay grounded in the observable evidence.

---

## The Adversarial Loop

The **Drift Agent** is not passive. In `adaptive` mode, it tracks the reviewer's win rate over the last 5, 10, and 20 episodes and adjusts:

- If the reviewer is catching everything → switch to `subtle` bug patterns (deeper in the call stack)
- If the reviewer is struggling → stay on `aggressive` patterns (structural, obvious breaks)
- It can **target weak families** — if the model consistently misses `condition_flip`, the agent spawns more of them

This is the self-improvement loop: **the environment gets harder as the model gets better.**

Bug difficulty also scales with the explicit levels:
- **Easy** — 1 mutation per episode, rename/removal only
- **Medium** — 2 mutations, includes contract changes
- **Hard** — 3 mutations, all 8 bug patterns in play

---

## What the Demo Proves

The **Battle** tab is the clearest proof we can show in one click:

Same seed. Same mutation. Same PR. Same tests. Two reviewers.

**Junior (untrained):**
```
VERDICT: APPROVE
REASON: The code looks correct and follows existing patterns.
```
Reward: **−1.0** — Bug ships to production.

**Senior (GRPO-trained):**
```
VERDICT: REQUEST_CHANGES
ROOT_CAUSE: getUserData
FAILURE_PATH: test_profile_page → get_user_dashboard → getUserData → AttributeError
ISSUES: getUserData was renamed to fetchUserProfile. PR still uses old name.
```
Reward: **+2.05** — Bug caught. Merge blocked.

Delta: **+3.05** per episode. Across the **Gauntlet** (10 rounds): Senior wins 90%+.

---

## Theme Fit

| Theme | Fit | Why |
|---|---|---|
| **#4 — Self-Improvement** | **~45%** | Adaptive adversary, escalating curriculum, generator responds to reviewer win rate — the environment drives its own challenge level |
| **#3.1 — World Modeling (Professional Tasks)** | **~40%** | Real execution oracle (pytest), partially observable world (reviewer sees diff+tests, not the mutation), causal reward grounded in runtime facts |
| **#5 — Wild Card** | **~10%** | Executable ground truth for code review is a genuinely novel training signal — no human labels anywhere in the loop |
| **#1 — Multi-Agent** | **~5%** | Generator vs Reviewer is adversarial, but the "multi-agent" interaction is implicit, not the primary training story |

**Primary submission: Theme #4 (Self-Improvement)** — the core claim is that the environment *adapts to the model* and drives recursive capability growth in causal code debugging.

---

## Technical Stack

| Layer | What we used |
|---|---|
| **Model** | Qwen2.5-1.5B-Instruct (fits T4 free Colab, ~8GB VRAM) |
| **Training** | TRL `GRPOTrainer` + 4-bit QLoRA (bitsandbytes, PEFT) |
| **Environment** | Custom Python `Env` with OpenEnv-compatible API |
| **Execution oracle** | Subprocess `pytest` on AST-mutated mini-repos (V2) |
| **Reward** | 9-component causal scorer (V1: heuristic, V2: real execution) |
| **Demo** | Gradio 6 on Hugging Face Spaces (CPU-only, no GPU needed) |
| **Server** | FastAPI + OpenEnv manifest (`openenv.yaml`) |

---

## What's Next

- Run the full V2 training loop (real pytest oracle) and publish curves
- Upload a fine-tuned adapter to the Hub
- Add multi-turn episodes (reviewer asks clarifying questions)
- Extend to multi-file diffs and cross-module renames

---

## Try It

**Live demo:** [huggingface.co/spaces/Bhuneshlooper/CodeDrift](https://huggingface.co/spaces/Bhuneshlooper/CodeDrift)

**Source:** [github.com/bansalbhunesh/codedrift-arena](https://github.com/bansalbhunesh/codedrift-arena)

**Train it yourself:**
```bash
git clone https://github.com/bansalbhunesh/codedrift-arena
cd codedrift-arena
pip install -r requirements-train.txt
python training/train.py --difficulty easy --steps 200 --episodes 500
```

---

*Built for the OpenEnv Hackathon. The environment, scorer, and demo are fully open-source.*
