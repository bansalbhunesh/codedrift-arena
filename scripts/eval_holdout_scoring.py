#!/usr/bin/env python3
"""
Deterministic holdout / OOD sanity check for the reward function (no GPU model).

Reads JSON rows produced by ``training/train.py --eval_split_out ...`` and
reports mean reward for a naive APPROVE vs a template REQUEST_CHANGES that
cites every serialized stale action by bare symbol name.

Usage:
  python training/train.py --episodes 50 --eval_split_out eval_rows.json
  python scripts/eval_holdout_scoring.py eval_rows.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.drift_agent import DriftAction
from rewards.scorer import RewardScorer


def _load_actions(raw: object) -> list[DriftAction]:
    if isinstance(raw, str):
        rows = json.loads(raw)
    else:
        rows = raw
    return [
        DriftAction(
            drift_type=str(d["drift_type"]),
            stale_ref=str(d["stale_ref"]),
            current_ref=str(d["current_ref"]),
            metadata=dict(d.get("metadata") or {}),
        )
        for d in rows
    ]


def naive_response() -> str:
    return "VERDICT: APPROVE\nISSUES: none\nREASON: Looks fine.\n"


def oracle_response(actions: list[DriftAction]) -> str:
    parts = []
    for a in actions:
        bare = a.stale_ref.split("(")[0]
        parts.append(bare)
    issues = ", ".join(parts) if parts else "none"
    return f"VERDICT: REQUEST_CHANGES\nISSUES: {issues}\nREASON: Stale references must be fixed.\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Score held-out rows with naive vs oracle reviews")
    p.add_argument("eval_json", type=Path, help="JSON file of dataset rows (with serialized_actions)")
    args = p.parse_args()
    data = json.loads(args.eval_json.read_text(encoding="utf-8"))
    scorer = RewardScorer()
    naive_r, oracle_r = [], []
    for row in data:
        actions = _load_actions(row.get("serialized_actions", "[]"))
        diff = str(row.get("pr_diff", ""))
        r1, _ = scorer.score(naive_response(), actions, diff)
        r2, _ = scorer.score(oracle_response(actions), actions, diff)
        naive_r.append(float(r1))
        oracle_r.append(float(r2))
    n = len(data)
    if not n:
        print("No rows in file.")
        return
    mn = sum(naive_r) / n
    mo = sum(oracle_r) / n
    print(f"rows={n}")
    print(f"mean_reward naive_approve: {mn:+.3f}")
    print(f"mean_reward oracle_catch: {mo:+.3f}")
    print("If naive is near oracle on drifted rows, the rubric or rows may be degenerate.")


if __name__ == "__main__":
    main()
