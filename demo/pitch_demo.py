"""
CodeDrift Arena — Pitch Demo
Run:  python demo/pitch_demo.py
"""

import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from codedrift.logutil import configure_logging
from agents.drift_agent import DriftAction
from env.codebase import build_base_codebase
from env.codedrift_env import CodeDriftEnv

SEPARATOR = "-" * 62

BASE_MODEL_RESPONSE = """\
VERDICT: APPROVE
ISSUES: none
REASON: Code looks clean, follows project patterns, imports are correct."""

TRAINED_MODEL_RESPONSE = """\
VERDICT: REQUEST_CHANGES
ISSUES: getUserData is not in the current codebase. It was renamed to
fetchUserData. The call on line +8 will raise a NameError at runtime.
REASON: Stale function reference detected. Update getUserData -> fetchUserData before merging."""


def _rename_pitch_state():
    base = build_base_codebase()
    drifted = copy.deepcopy(base)
    drifted.functions.pop("getUserData", None)
    drifted.functions["fetchUserData"] = "userId: str"
    actions = [
        DriftAction(
            drift_type="rename",
            stale_ref="getUserData",
            current_ref="fetchUserData",
            metadata={"signature": "userId: str"},
        )
    ]
    pr_diff = (
        "diff --git a/src/feature.py b/src/feature.py\n"
        "--- a/src/feature.py\n"
        "+++ b/src/feature.py\n"
        "@@ -1,10 +1,18 @@\n"
        "+from models.user import User\n"
        "+from models.order import Order\n"
        "+\n"
        "+def process_user_request(user_id: str, item: str):\n"
        "+    order = Order.find(user_id)\n"
        "+    if not order:\n"
        "+        raise ValueError('Order not found')\n"
        "+    data = getUserData(user_id)   # <- stale reference\n"
        "+    return {'status': 'ok', 'data': data}"
    )
    return base, drifted, actions, pr_diff


def run_demo(seed: int = 7):
    configure_logging()

    base, drifted, actions, pr_diff = _rename_pitch_state()

    env = CodeDriftEnv(difficulty="easy", personality="random", seed=seed)
    obs = env.inject_episode(drifted=drifted, actions=actions, pr_diff=pr_diff, base=base)

    print(f"\n{'=' * 62}")
    print("  CODEDRIFT ARENA - LIVE PITCH DEMO")
    print(f"{'=' * 62}\n")

    print("STEP 1: Drift agent mutates the codebase")
    print(f"{SEPARATOR}")
    print("  getUserData(userId)  ->  fetchUserData(userId)")
    print("  [Function renamed - codebase updated, PR not updated]\n")

    print("STEP 2: PR diff shown to reviewer agent")
    print(f"{SEPARATOR}")
    print(env.pr_diff)

    print(f"\nSTEP 3: Current codebase state (what agent sees)")
    print(f"{SEPARATOR}")
    for line in obs.codebase_context.splitlines()[:14]:
        if line.strip():
            print(f"  {line}")
    print("  ...")

    print(f"\n{'=' * 62}")
    print("  BEFORE TRAINING (base model)")
    print(f"{'=' * 62}")
    print(BASE_MODEL_RESPONSE)
    _, r_before, _, info_b = env.step(BASE_MODEL_RESPONSE)
    rb = float(info_b.get("recall", 0.0))
    print(
        f"\n  REWARD: {r_before:+.1f}  (ships broken code) | RECALL: {rb:.0%} | "
        f"OUTCOME: {info_b.get('episode_outcome')}"
    )

    env2 = CodeDriftEnv(difficulty="easy", seed=seed)
    env2.inject_episode(
        drifted=copy.deepcopy(drifted),
        actions=copy.deepcopy(actions),
        pr_diff=pr_diff,
        base=copy.deepcopy(base),
    )

    print(f"\n{'=' * 62}")
    print("  AFTER TRAINING")
    print(f"{'=' * 62}")
    print(TRAINED_MODEL_RESPONSE)
    _, r_after, _, info_a = env2.step(TRAINED_MODEL_RESPONSE)
    ra = float(info_a.get("recall", 0.0))
    print(
        f"\n  REWARD: {r_after:+.1f}  (catches the bug) | RECALL: {ra:.0%} | "
        f"OUTCOME: {info_a.get('episode_outcome')}"
    )

    print(f"\n{'=' * 62}")
    print(f"  IMPROVEMENT:  {r_before:+.1f}  ->  {r_after:+.1f}  (delta {r_after - r_before:+.1f})")
    print(f"{'=' * 62}\n")

    print("PITCH LINE:")
    print("  'We are not testing if an LLM can review code.'")
    print("  'We are training it to catch bugs in codebases that")
    print("   change underneath it - exactly what happens in production.'")
    print()


if __name__ == "__main__":
    run_demo()
