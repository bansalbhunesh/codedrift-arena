"""CodeDriftEnv lifecycle and inject_episode validation."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.drift_agent import DriftAction
from env.codebase import build_base_codebase
from env.codedrift_env import CodeDriftEnv


class TestCodeDriftEnv(unittest.TestCase):
    def test_second_step_raises_until_reset(self) -> None:
        env = CodeDriftEnv(difficulty="easy", seed=0)
        env.reset()
        env.step("VERDICT: APPROVE\nISSUES: none\nREASON: x.\n")
        with self.assertRaises(RuntimeError):
            env.step("VERDICT: APPROVE\nISSUES: none\nREASON: again.\n")

    def test_inject_rename_mismatch_raises(self) -> None:
        base = build_base_codebase()
        drifted = copy.deepcopy(base)
        drifted.functions.pop("getUserData", None)
        drifted.functions["fetchUserData"] = "userId: str"
        acts = [
            DriftAction(
                drift_type="rename",
                stale_ref="getUserData",
                current_ref="fetchUserData",
                metadata={},
            ),
        ]
        env = CodeDriftEnv()
        with self.assertRaises(ValueError):
            env.inject_episode(
                drifted=drifted,
                actions=acts,
                pr_diff="+no stale token here\n",
                base=base,
            )


if __name__ == "__main__":
    unittest.main()
