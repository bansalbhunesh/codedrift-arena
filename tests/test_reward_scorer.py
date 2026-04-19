"""RewardScorer regression tests (no pytest required)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.drift_agent import DriftAction
from rewards.scorer import RewardScorer


class TestRewardScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.s = RewardScorer()

    def test_clean_pr_approve(self) -> None:
        r, info = self.s.score(
            "VERDICT: APPROVE\nISSUES: none\nREASON: ok.\n",
            [],
            "",
        )
        self.assertAlmostEqual(r, self.s.R_CORRECT_APPROVE)
        self.assertEqual(info["episode_outcome"], "correct_approve")

    def test_clean_pr_reject(self) -> None:
        r, info = self.s.score(
            "VERDICT: REQUEST_CHANGES\nISSUES: something\nREASON: no.\n",
            [],
            "",
        )
        self.assertAlmostEqual(r, self.s.R_FALSE_REJECTION)

    def test_rename_caught_only_if_stale_name_mentioned(self) -> None:
        """Regression: citing only the new symbol must not count as catching drift."""
        a = DriftAction(
            drift_type="rename",
            stale_ref="getUserData",
            current_ref="fetchUserData",
            metadata={},
        )
        r, info = self.s.score(
            "VERDICT: REQUEST_CHANGES\nISSUES: switch to fetchUserData\nREASON: rename.\n",
            [a],
            "",
        )
        self.assertIn("missed", info)
        self.assertEqual(info["missed"], ["rename:getUserData"])

        r2, info2 = self.s.score(
            "VERDICT: REQUEST_CHANGES\nISSUES: getUserData is stale; use fetchUserData\nREASON: ok.\n",
            [a],
            "",
        )
        self.assertEqual(info2["caught"], ["rename:getUserData"])
        self.assertAlmostEqual(r2, self.s.R_CAUGHT_STALE)

    def test_step_before_reset_raises(self) -> None:
        from env.codedrift_env import CodeDriftEnv

        env = CodeDriftEnv()
        with self.assertRaises(RuntimeError):
            env.step("VERDICT: APPROVE\nISSUES: none\nREASON: x\n")

    def test_contract_requires_old_params_in_issues(self) -> None:
        a = DriftAction(
            drift_type="contract",
            stale_ref="createOrder(item, qty)",
            current_ref="createOrder(item, qty, userId)",
            metadata={"function": "createOrder", "old_params": ["item", "qty"], "new_params": ["item", "qty", "userId"]},
        )
        r, info = self.s.score(
            "VERDICT: REQUEST_CHANGES\nISSUES: createOrder signature changed\nREASON: createOrder(item, qty) is stale\n",
            [a],
            "",
        )
        self.assertEqual(info["missed"], ["contract:createOrder"])

        r2, info2 = self.s.score(
            "VERDICT: REQUEST_CHANGES\nISSUES: PR still calls createOrder(item, qty) without userId\nREASON: ok.\n",
            [a],
            "",
        )
        self.assertEqual(info2["caught"], ["contract:createOrder"])
        self.assertAlmostEqual(r2, self.s.R_CAUGHT_STALE)

    def test_inject_episode_then_step(self) -> None:
        import copy

        from env.codebase import build_base_codebase
        from env.codedrift_env import CodeDriftEnv

        base = build_base_codebase()
        d = copy.deepcopy(base)
        d.functions.pop("getUserData", None)
        d.functions["fetchUserData"] = "userId: str"
        acts = [
            DriftAction(
                drift_type="rename",
                stale_ref="getUserData",
                current_ref="fetchUserData",
                metadata={},
            ),
        ]
        env = CodeDriftEnv()
        env.inject_episode(drifted=d, actions=acts, pr_diff="+stale", base=base)
        obs, r, done, _ = env.step(
            "VERDICT: REQUEST_CHANGES\nISSUES: getUserData\nREASON: x.\n"
        )
        self.assertTrue(done)
        self.assertEqual(obs.episode_step, 0)
        self.assertGreater(r, 0.0)


if __name__ == "__main__":
    unittest.main()
