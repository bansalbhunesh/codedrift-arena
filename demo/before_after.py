"""
CodeDrift Arena — Before/After Demo
Run:  python demo/before_after.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.codedrift_env import CodeDriftEnv

BEFORE_RESPONSE = """VERDICT: APPROVE
ISSUES: none
REASON: The code looks clean and follows existing patterns. Imports are correct."""

AFTER_RESPONSE = """VERDICT: REQUEST_CHANGES
ISSUES: getUserData is no longer defined in the current codebase. It was renamed to fetchUserData. Line calling getUserData(user_id) will raise a NameError at runtime.
REASON: The PR references a stale function name. Must be updated to fetchUserData before merging."""


def _actions_to_rows(actions):
    rows = []
    for a in actions:
        rows.append(
            {
                "type": a.drift_type,
                "stale": a.stale_ref,
                "current": a.current_ref if a.drift_type != "removal" else "[deleted]",
            }
        )
    return rows


def run_demo(seed: int = 42):
    """Uses a fixed rename drift so canned before/after responses match scoring."""

    env = CodeDriftEnv(difficulty="easy", seed=seed)
    env.reset()

    from agents.drift_agent import DriftAction

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
        "+from models.user import User\n"
        "+data = getUserData(user_id)  # stale\n"
        "+return data\n"
    )
    obs = env._build_obs()

    sep = "=" * 60

    print(f"\n{sep}")
    print("CODEDRIFT ARENA - BEFORE / AFTER DEMO")
    print(sep)

    print("\nPR DIFF SHOWN TO AGENT:")
    print(obs.pr_diff)

    print("\nCURRENT CODEBASE (what agent sees):")
    print(obs.codebase_context)

    print(f"\nGROUND TRUTH STALE REFS (hidden from agent):")
    for ref in _actions_to_rows(env.stale_actions):
        print(f"  [{ref['type'].upper()}] {ref['stale']} -> {ref.get('current', 'REMOVED')}")

    print(f"\n{'-' * 60}")
    print("BEFORE TRAINING (base model):")
    print(BEFORE_RESPONSE)
    _, reward_before, _, info_before = env.step(BEFORE_RESPONSE)
    print(f"\nREWARD: {reward_before:+.1f}")
    print(f"CAUGHT: {info_before['caught']} | MISSED: {info_before['missed']}")

    env2 = CodeDriftEnv(difficulty="easy", seed=seed)
    env2.reset()
    env2._actions = list(env._actions)
    env2._drifted = env._drifted
    env2._pr_diff = env._pr_diff

    print(f"\n{'-' * 60}")
    print("AFTER TRAINING:")
    print(AFTER_RESPONSE)
    _, reward_after, _, info_after = env2.step(AFTER_RESPONSE)
    print(f"\nREWARD: {reward_after:+.1f}")
    print(f"CAUGHT: {info_after['caught']} | MISSED: {info_after['missed']}")

    print(f"\n{'-' * 60}")
    delta = reward_after - reward_before
    print(f"IMPROVEMENT: {reward_before:+.1f} -> {reward_after:+.1f}  (delta {delta:+.1f})")
    print(sep)


if __name__ == "__main__":
    run_demo()
