"""
CodeDrift Arena — Pitch Demo
Run:  python demo/pitch_demo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.drift_agent import DriftAction, DriftAgent
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


def run_demo(seed: int = 7):
    env = CodeDriftEnv(difficulty="easy", personality="random", seed=seed)
    env.drift_agent = DriftAgent(personality="random", seed=seed)
    env.reset()

    env._actions = [
        DriftAction(
            drift_type="rename",
            stale_ref="getUserData",
            current_ref="fetchUserData",
            metadata={"signature": "userId: str"},
        )
    ]
    env._drifted.functions.pop("getUserData", None)
    env._drifted.functions["fetchUserData"] = "userId: str"
    env._pr_diff = (
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
    env._build_obs()

    print(f"\n{'=' * 62}")
    print("  CODEDRIFT ARENA - LIVE PITCH DEMO")
    print(f"{'=' * 62}\n")

    print("STEP 1: Drift agent mutates the codebase")
    print(f"{SEPARATOR}")
    print("  getUserData(userId)  ->  fetchUserData(userId)")
    print("  [Function renamed - codebase updated, PR not updated]\n")

    print("STEP 2: PR diff shown to reviewer agent")
    print(f"{SEPARATOR}")
    print(env._pr_diff)

    print(f"\nSTEP 3: Current codebase state (what agent sees)")
    print(f"{SEPARATOR}")
    for name in sorted(env._drifted.functions.keys())[:6]:
        print(f"  def {name}({env._drifted.functions[name]})")
    print("  ...")

    print(f"\n{'=' * 62}")
    print("  BEFORE TRAINING (base model)")
    print(f"{'=' * 62}")
    print(BASE_MODEL_RESPONSE)
    _, r_before, _, _ = env.step(BASE_MODEL_RESPONSE)
    print(f"\n  REWARD: {r_before:+.1f}  (ships broken code)")

    env2 = CodeDriftEnv(difficulty="easy", seed=seed)
    env2._actions = env._actions
    env2._pr_diff = env._pr_diff
    env2._drifted = env._drifted

    print(f"\n{'=' * 62}")
    print("  AFTER TRAINING")
    print(f"{'=' * 62}")
    print(TRAINED_MODEL_RESPONSE)
    _, r_after, _, _ = env2.step(TRAINED_MODEL_RESPONSE)
    print(f"\n  REWARD: {r_after:+.1f}  (catches the bug)")

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
