"""
CodeDrift Arena — Before/After Demo
Run:  python demo/before_after.py

Shows a **funny toxic-approve** beat, then four vignettes building to a **multi-drift finale**:
rename → removal → contract → **rename+contract (hero ending)**.
"""

import copy
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from codedrift.logutil import configure_logging
from agents.drift_agent import DriftAction
from env.codebase import build_base_codebase
from env.codedrift_env import CodeDriftEnv

FUNNY_RESPONSE = """VERDICT: APPROVE
ISSUES: none
REASON: Everything looks good 👍 Ship it."""

BEFORE_RESPONSE = """VERDICT: APPROVE
ISSUES: none
REASON: The code looks clean and follows existing patterns. Imports are correct."""

AFTER_RESPONSE = """VERDICT: REQUEST_CHANGES
ISSUES: getUserData is no longer defined in the current codebase. It was renamed to fetchUserData. Line calling getUserData(user_id) will raise a NameError at runtime.
REASON: The PR references a stale function name. Must be updated to fetchUserData before merging."""

BEFORE_CONTRACT = """VERDICT: APPROVE
ISSUES: none
REASON: Looks consistent with the order service."""

AFTER_CONTRACT = """VERDICT: REQUEST_CHANGES
ISSUES: createOrder is still invoked with only item and qty; the current API requires userId. The call createOrder(item, qty) is stale.
REASON: PR must pass userId to match the codebase contract."""

BEFORE_REMOVAL = """VERDICT: APPROVE
ISSUES: none
REASON: Helper import looks fine for this release."""

AFTER_REMOVAL = """VERDICT: REQUEST_CHANGES
ISSUES: The PR imports from utils.legacy, but utils/legacy.py is not in the current codebase file list; that module was removed.
REASON: Stale import from a deleted path; drop or replace before merge."""

BEFORE_MULTI = """VERDICT: APPROVE
ISSUES: none
REASON: Ship it; the diff matches our usual patterns."""

AFTER_MULTI = """VERDICT: REQUEST_CHANGES
ISSUES: getUserData is stale (renamed to fetchUserData). createOrder(item, qty) is stale; current API requires userId with item and qty.
REASON: Two independent drift issues; fix both call sites before merge."""


def print_judge_panel(phase: str, reward: float, info: dict) -> None:
    """Explicit BEFORE/AFTER style block for terminal demos."""
    print(f"\n[{phase}]")
    em = (info.get("judge_emoji") or "⚪").strip()
    strip = info.get("metric_strip") or f"reward={reward:+.2f}"
    print(f"{em} {strip}")
    if info.get("judge_summary"):
        print(f"-> {info['judge_summary']}")
    if info.get("judge_why_matters"):
        print(f"💡 {info['judge_why_matters']}")
    if info.get("confidence_strip"):
        print(info["confidence_strip"])


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


def _rename_demo_state():
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
        ),
    ]
    pr_diff = (
        "diff --git a/src/feature.py b/src/feature.py\n"
        "--- a/src/feature.py\n"
        "+++ b/src/feature.py\n"
        "+from models.user import User\n"
        "+data = getUserData(user_id)  # stale\n"
        "+return data\n"
    )
    return base, drifted, actions, pr_diff


def _contract_demo_state():
    base = build_base_codebase()
    drifted = copy.deepcopy(base)
    # Match a plausible tree where rename already landed (avoids showing stale getUserData
    # next to an upgraded createOrder signature in the same snapshot).
    drifted.functions.pop("getUserData", None)
    drifted.functions["fetchUserData"] = "userId: str"
    drifted.api_signatures["createOrder"] = ["item", "qty", "userId"]
    actions = [
        DriftAction(
            drift_type="contract",
            stale_ref="createOrder(item, qty)",
            current_ref="createOrder(item, qty, userId)",
            metadata={
                "function": "createOrder",
                "old_params": ["item", "qty"],
                "new_params": ["item", "qty", "userId"],
            },
        ),
    ]
    pr_diff = (
        "diff --git a/src/orders.py b/src/orders.py\n"
        "+++ b/src/orders.py\n"
        "+# PR still uses old two-argument call\n"
        "+result = createOrder(item, qty)  # stale: missing userId\n"
    )
    return base, drifted, actions, pr_diff


def _removal_demo_state():
    base = build_base_codebase()
    drifted = copy.deepcopy(base)
    stale_path = "utils/legacy.py"
    if stale_path not in drifted.files:
        raise RuntimeError(f"expected {stale_path!r} in base codebase files")
    drifted.files.remove(stale_path)
    actions = [
        DriftAction(
            drift_type="removal",
            stale_ref=stale_path,
            current_ref="[deleted]",
            metadata={"module": "utils.legacy"},
        ),
    ]
    pr_diff = (
        "diff --git a/src/helpers.py b/src/helpers.py\n"
        "+++ b/src/helpers.py\n"
        "+from utils.legacy import format_date  # stale: module deleted\n"
    )
    return base, drifted, actions, pr_diff


def _multi_drift_demo_state():
    """Rename + contract in one episode (two ground-truth stale refs)."""
    base = build_base_codebase()
    drifted = copy.deepcopy(base)
    drifted.functions.pop("getUserData", None)
    drifted.functions["fetchUserData"] = "userId: str"
    drifted.api_signatures["createOrder"] = ["item", "qty", "userId"]
    actions = [
        DriftAction(
            drift_type="rename",
            stale_ref="getUserData",
            current_ref="fetchUserData",
            metadata={"signature": "userId: str"},
        ),
        DriftAction(
            drift_type="contract",
            stale_ref="createOrder(item, qty)",
            current_ref="createOrder(item, qty, userId)",
            metadata={
                "function": "createOrder",
                "old_params": ["item", "qty"],
                "new_params": ["item", "qty", "userId"],
            },
        ),
    ]
    pr_diff = (
        "diff --git a/src/service.py b/src/service.py\n"
        "+++ b/src/service.py\n"
        "+data = getUserData(user_id)\n"
        "+order = createOrder(item, qty)\n"
    )
    return base, drifted, actions, pr_diff


def _print_before_after_scenario(
    *,
    title: str,
    base,
    drifted,
    actions,
    pr_diff: str,
    before_text: str,
    after_text: str,
) -> None:
    env = CodeDriftEnv(difficulty="easy")
    obs = env.inject_episode(drifted=drifted, actions=actions, pr_diff=pr_diff, base=base)

    sep = "=" * 60
    print(f"\n{sep}")
    print(title)
    print(sep)

    print("\nPR DIFF SHOWN TO AGENT:")
    print(obs.pr_diff)

    print("\nCURRENT CODEBASE (what agent sees):")
    print(obs.codebase_context)

    print("\nGROUND TRUTH STALE REFS (hidden from agent):")
    for ref in _actions_to_rows(env.stale_actions):
        print(f"  [{ref['type'].upper()}] {ref['stale']} -> {ref.get('current', 'REMOVED')}")

    print(f"\n{'-' * 60}")
    print("BEFORE TRAINING (base model):")
    print(before_text)
    _, reward_before, _, info_before = env.step(before_text)
    rb = float(info_before.get("recall", 0.0))
    print(
        f"CAUGHT: {info_before['caught']} | MISSED: {info_before['missed']} | "
        f"RECALL: {rb:.0%} | OUTCOME: {info_before.get('episode_outcome')}"
    )
    print_judge_panel("BEFORE TRAINING", reward_before, info_before)

    env2 = CodeDriftEnv(difficulty="easy")
    env2.inject_episode(
        drifted=copy.deepcopy(drifted),
        actions=copy.deepcopy(actions),
        pr_diff=pr_diff,
        base=copy.deepcopy(base),
    )

    print(f"\n{'-' * 60}")
    print("AFTER TRAINING:")
    print(after_text)
    _, reward_after, _, info_after = env2.step(after_text)
    ra = float(info_after.get("recall", 0.0))
    print(
        f"CAUGHT: {info_after['caught']} | MISSED: {info_after['missed']} | "
        f"RECALL: {ra:.0%} | OUTCOME: {info_after.get('episode_outcome')}"
    )
    print_judge_panel("AFTER TRAINING", reward_after, info_after)

    print(f"\n{'-' * 60}")
    delta = reward_after - reward_before
    print(f"IMPROVEMENT: {reward_before:+.1f} -> {reward_after:+.1f}  (delta {delta:+.1f})")
    print(sep)


def funny_failure_interlude() -> None:
    """Toxic positivity on a drifted rename — memorable demo beat."""
    sep = "-" * 60
    print(f"\n{sep}")
    print("INTERLUDE: Toxic positivity (same drift as Scenario 1)")
    print(sep)
    base, drifted, actions, pr_diff = _rename_demo_state()
    env = CodeDriftEnv(difficulty="easy")
    env.inject_episode(drifted=drifted, actions=actions, pr_diff=pr_diff, base=base)
    print(FUNNY_RESPONSE)
    _, r, _, info = env.step(FUNNY_RESPONSE)
    print(f"\nREWARD: {r:+.1f}  (narrator: this is what we train *away* from)")
    print_judge_panel("INTERLUDE (toxic positivity result)", r, info)


def run_demo(seed: int = 42):
    _ = seed  # reserved for future stochastic episodes
    configure_logging()

    print("\n" + "=" * 60)
    print("CODEDRIFT ARENA - BEFORE / AFTER DEMO (multi-scenario)")
    print("=" * 60)

    funny_failure_interlude()

    b1, d1, a1, p1 = _rename_demo_state()
    _print_before_after_scenario(
        title="SCENARIO 1: Function rename (stale symbol in PR)",
        base=b1,
        drifted=d1,
        actions=a1,
        pr_diff=p1,
        before_text=BEFORE_RESPONSE,
        after_text=AFTER_RESPONSE,
    )

    b3, d3, a3, p3 = _removal_demo_state()
    _print_before_after_scenario(
        title="SCENARIO 2: Deleted module (stale import path)",
        base=b3,
        drifted=d3,
        actions=a3,
        pr_diff=p3,
        before_text=BEFORE_REMOVAL,
        after_text=AFTER_REMOVAL,
    )

    b2, d2, a2, p2 = _contract_demo_state()
    _print_before_after_scenario(
        title="SCENARIO 3: API contract drift (stale call arity)",
        base=b2,
        drifted=d2,
        actions=a2,
        pr_diff=p2,
        before_text=BEFORE_CONTRACT,
        after_text=AFTER_CONTRACT,
    )

    b4, d4, a4, p4 = _multi_drift_demo_state()
    _print_before_after_scenario(
        title="SCENARIO 4 (FINALE): Two drifts at once (rename + contract)",
        base=b4,
        drifted=d4,
        actions=a4,
        pr_diff=p4,
        before_text=BEFORE_MULTI,
        after_text=AFTER_MULTI,
    )


if __name__ == "__main__":
    run_demo()
