"""
CodeDrift Arena - Pitch Demo
Run:  python demo/pitch_demo.py

Three short vignettes (rename, contract, removal); same before/after reward arc as demo/before_after.py.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

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


def _load_before_after_module() -> Any:
    """Load sibling ``before_after.py`` for shared scenario builders (no package required)."""
    path = Path(__file__).resolve().parent / "before_after.py"
    spec = importlib.util.spec_from_file_location("codedrift_before_after_demo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scenario module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def _run_pitch_scenario(
    *,
    title: str,
    drift_blurb: str,
    base,
    drifted,
    actions,
    pr_diff: str,
    before: str,
    after: str,
    seed: int,
) -> None:
    env = CodeDriftEnv(difficulty="easy", personality="random", seed=seed)
    obs = env.inject_episode(drifted=drifted, actions=actions, pr_diff=pr_diff, base=base)

    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}\n")

    print("STEP 1: Drift agent mutates the codebase")
    print(f"{SEPARATOR}")
    print(drift_blurb)
    print()

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
    print(before)
    _, r_before, _, info_b = env.step(before)
    rb = float(info_b.get("recall", 0.0))
    print(
        f"\n  REWARD: {r_before:+.1f}  (ships broken code) | RECALL: {rb:.0%} | "
        f"OUTCOME: {info_b.get('episode_outcome')}"
    )
    if info_b.get("metric_strip"):
        print(f"  METRICS: {info_b['metric_strip']}")

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
    print(after)
    _, r_after, _, info_a = env2.step(after)
    ra = float(info_a.get("recall", 0.0))
    print(
        f"\n  REWARD: {r_after:+.1f}  (catches the bug) | RECALL: {ra:.0%} | "
        f"OUTCOME: {info_a.get('episode_outcome')}"
    )
    if info_a.get("metric_strip"):
        print(f"  METRICS: {info_a['metric_strip']}")

    print(f"\n{'=' * 62}")
    print(f"  IMPROVEMENT:  {r_before:+.1f}  ->  {r_after:+.1f}  (delta {r_after - r_before:+.1f})")
    print(f"{'=' * 62}\n")


def run_demo(seed: int = 7, scenario: str = "all") -> None:
    configure_logging()
    ba = _load_before_after_module()

    scenarios: list[tuple[str, str, tuple, str, str]] = []

    if scenario in ("all", "rename"):
        b, d, a, p = _rename_pitch_state()
        scenarios.append(
            (
                "SCENARIO 1 - Rename (stale function in PR)",
                "  getUserData(userId)  ->  fetchUserData(userId)\n"
                "  [Function renamed - codebase updated, PR not updated]",
                (b, d, a, p),
                BASE_MODEL_RESPONSE,
                TRAINED_MODEL_RESPONSE,
            )
        )
    if scenario in ("all", "contract"):
        b, d, a, p = ba._contract_demo_state()
        scenarios.append(
            (
                "SCENARIO 2 - Contract (stale call arity)",
                "  createOrder now requires userId\n"
                "  [API signature upgraded - PR still uses old arity]",
                (b, d, a, p),
                ba.BEFORE_CONTRACT,
                ba.AFTER_CONTRACT,
            )
        )
    if scenario in ("all", "removal"):
        b, d, a, p = ba._removal_demo_state()
        scenarios.append(
            (
                "SCENARIO 3 - Removal (dead import path)",
                "  utils/legacy.py removed from the repo\n"
                "  [File deleted - PR still imports it]",
                (b, d, a, p),
                ba.BEFORE_REMOVAL,
                ba.AFTER_REMOVAL,
            )
        )
    if scenario in ("all", "multi"):
        b, d, a, p = ba._multi_drift_demo_state()
        scenarios.append(
            (
                "SCENARIO 4 - Two drifts (rename + contract)",
                "  Two independent problems in one PR\n"
                "  [Stale function name AND stale API arity]",
                (b, d, a, p),
                ba.BEFORE_MULTI,
                ba.AFTER_MULTI,
            )
        )

    if not scenarios:
        raise SystemExit(f"unknown scenario: {scenario!r} (use all|rename|contract|removal|multi)")

    print(f"\n{'#' * 62}")
    print("  CODEDRIFT ARENA - LIVE PITCH DEMO")
    print(f"{'#' * 62}")

    for title, blurb, tup, before, after in scenarios:
        b, d, a, p = tup
        _run_pitch_scenario(
            title=title,
            drift_blurb=blurb,
            base=b,
            drifted=d,
            actions=a,
            pr_diff=p,
            before=before,
            after=after,
            seed=seed,
        )

    print("PITCH LINE:")
    print("  'We are not testing if an LLM can review code.'")
    print("  'We are training it to catch bugs in codebases that")
    print("   change underneath it - exactly what happens in production.'")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="CodeDrift Arena pitch demo")
    p.add_argument(
        "--scenario",
        choices=["all", "rename", "contract", "removal", "multi"],
        default="all",
        help="Run one drift family, multi-drift, or all (default: all).",
    )
    p.add_argument("--seed", type=int, default=7, help="RNG seed for env construction.")
    args = p.parse_args()
    run_demo(seed=args.seed, scenario=args.scenario)


if __name__ == "__main__":
    main()
