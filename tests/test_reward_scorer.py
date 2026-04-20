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
        self.assertIn("metric_strip", info)
        self.assertEqual(info.get("diff_grounded_count"), 0)
        self.assertEqual(info.get("judge_emoji"), "🟢")
        self.assertIn("correctly approved", (info.get("judge_summary") or "").lower())
        self.assertIn("false alarms", (info.get("judge_why_matters") or "").lower())
        self.assertIn("confidence:", (info.get("confidence_strip") or "").lower())
        self.assertIn("SUCCESS", info.get("judge_keyword_line") or "")

    def test_toxic_positivity_approve_on_drift_is_red(self) -> None:
        """Drifted PR + APPROVE / ISSUES none should read as a clear failure to judges."""
        a = DriftAction(
            drift_type="rename",
            stale_ref="getUserData",
            current_ref="fetchUserData",
            metadata={},
        )
        toxic = (
            "VERDICT: APPROVE\nISSUES: none\n"
            "REASON: Everything looks good 👍 Ship it.\n"
        )
        r, info = self.s.score(toxic, [a], "+getUserData(x)\n")
        self.assertLess(r, 0.0)
        self.assertEqual(info.get("judge_emoji"), "🔴")
        self.assertIn("missed", (info.get("judge_summary") or "").lower())
        self.assertIn("FAILURE: missed schema drift", info.get("judge_keyword_line") or "")

    def test_perfect_multi_drift_keyword(self) -> None:
        """Finale-style episode with two actions gets the multi-drift SUCCESS headline."""
        a1 = DriftAction(
            drift_type="rename",
            stale_ref="getUserData",
            current_ref="fetchUserData",
            metadata={},
        )
        a2 = DriftAction(
            drift_type="contract",
            stale_ref="createOrder(item, qty)",
            current_ref="createOrder(item, qty, userId)",
            metadata={
                "function": "createOrder",
                "old_params": ["item", "qty"],
                "new_params": ["item", "qty", "userId"],
            },
        )
        pr = "+getUserData(x)\n+createOrder(item, qty)\n"
        text = (
            "VERDICT: REQUEST_CHANGES\n"
            "ISSUES: getUserData stale; createOrder(item, qty) missing userId with item and qty.\n"
            "REASON: fix both.\n"
        )
        r, info = self.s.score(text, [a1, a2], pr)
        self.assertAlmostEqual(r, 2.0)
        self.assertIn("multiple drifts blocked", info.get("judge_keyword_line") or "")

    def test_diff_grounding_and_metric_strip_on_drift(self) -> None:
        a = DriftAction(
            drift_type="rename",
            stale_ref="getUserData",
            current_ref="fetchUserData",
            metadata={},
        )
        pr = "+x = getUserData(uid)\n"
        r, info = self.s.score(
            "VERDICT: REQUEST_CHANGES\nISSUES: getUserData is stale\nREASON: ok.\n",
            [a],
            pr,
        )
        self.assertGreater(r, 0.0)
        self.assertEqual(len(info["diff_grounding"]), 1)
        self.assertTrue(info["diff_grounding"][0]["stale_token_in_pr_diff"])
        self.assertEqual(info.get("diff_grounded_count"), 1)
        self.assertIn("grounded_in_diff=1/1", info.get("metric_strip", ""))
        self.assertEqual(info.get("judge_emoji"), "🟢")
        self.assertIn("every injected", (info.get("judge_summary") or "").lower())
        self.assertIn("production bug", (info.get("judge_why_matters") or "").lower())
        self.assertIn("HIGH", (info.get("confidence_strip") or ""))
        self.assertIn("SUCCESS", info.get("judge_keyword_line") or "")

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

    def test_rename_substring_false_positive(self) -> None:
        """ISSUES must cite the stale identifier as a token, not as a substring."""
        a = DriftAction(
            drift_type="rename",
            stale_ref="getUserData",
            current_ref="fetchUserData",
            metadata={},
        )
        r, info = self.s.score(
            "VERDICT: REQUEST_CHANGES\nISSUES: getUserDatab is mentioned\nREASON: x.\n",
            [a],
            "",
        )
        self.assertEqual(info["missed"], ["rename:getUserData"])
        self.assertLess(r, 0)

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

    def test_contract_param_substring_not_counted(self) -> None:
        """Short param names must appear as tokens, not as substrings of other words."""
        a = DriftAction(
            drift_type="contract",
            stale_ref="createOrder(item, qty)",
            current_ref="createOrder(item, qty, userId)",
            metadata={
                "function": "createOrder",
                "old_params": ["item", "qty"],
                "new_params": ["item", "qty", "userId"],
            },
        )
        r, info = self.s.score(
            "VERDICT: REQUEST_CHANGES\n"
            "ISSUES: createOrder lineitem wrong\n"
            "REASON: x.\n",
            [a],
            "",
        )
        self.assertEqual(info["missed"], ["contract:createOrder"])

    def test_contract_natural_language_params(self) -> None:
        """ISSUES may name old params in prose without comma-separated call syntax."""
        a = DriftAction(
            drift_type="contract",
            stale_ref="createOrder(item, qty)",
            current_ref="createOrder(item, qty, userId)",
            metadata={
                "function": "createOrder",
                "old_params": ["item", "qty"],
                "new_params": ["item", "qty", "userId"],
            },
        )
        r, info = self.s.score(
            "VERDICT: REQUEST_CHANGES\n"
            "ISSUES: createOrder is called with wrong parameters item and qty without userId\n"
            "REASON: ok.\n",
            [a],
            "",
        )
        self.assertEqual(info["caught"], ["contract:createOrder"])
        self.assertAlmostEqual(r, self.s.R_CAUGHT_STALE)

    def test_spurious_stale_identifier_penalty(self) -> None:
        """Keyword-dumping stale catalog symbols should incur a deterministic penalty."""
        a = DriftAction(
            drift_type="rename",
            stale_ref="getUserData",
            current_ref="fetchUserData",
            metadata={},
        )
        focused, _ = self.s.score(
            "VERDICT: REQUEST_CHANGES\nISSUES: getUserData is stale\nREASON: x.\n",
            [a],
            "",
        )
        spammy, info_spam = self.s.score(
            (
                "VERDICT: REQUEST_CHANGES\n"
                "ISSUES: getUserData is stale; also createOrder sendEmail authenticate "
                "deleteRecord validateInput parseResponse checkPermission\n"
                "REASON: x.\n"
            ),
            [a],
            "",
        )
        self.assertLess(spammy, focused)
        self.assertGreater(len(info_spam.get("spurious_mentions", [])), 0)
        self.assertIn("spurious_stale_mentions", info_spam.get("breakdown", {}))

    def test_ungrounded_catch_gets_reduced_credit(self) -> None:
        """Catches without stale token evidence in diff should receive scaled reward."""
        a = DriftAction(
            drift_type="rename",
            stale_ref="getUserData",
            current_ref="fetchUserData",
            metadata={},
        )
        text = "VERDICT: REQUEST_CHANGES\nISSUES: getUserData is stale\nREASON: x.\n"
        grounded, info_g = self.s.score(text, [a], "+x = getUserData(uid)\n")
        ungrounded, info_u = self.s.score(text, [a], "+x = some_other_call(uid)\n")

        self.assertAlmostEqual(grounded, self.s.R_CAUGHT_STALE)
        self.assertAlmostEqual(
            ungrounded,
            self.s.R_CAUGHT_STALE * self.s.R_UNGROUNDED_CATCH_SCALE,
        )
        self.assertGreater(grounded, ungrounded)
        self.assertEqual(info_g.get("diff_grounded_count"), 1)
        self.assertEqual(info_u.get("diff_grounded_count"), 0)
        self.assertIn("caught_ungrounded_rename:getUserData", info_u.get("breakdown", {}))

    def test_no_spurious_penalty_for_expected_removal_module(self) -> None:
        """Removal drifts may reference stale file path or dotted module without penalty."""
        a = DriftAction(
            drift_type="removal",
            stale_ref="services/v1_client.py",
            current_ref="[deleted]",
            metadata={"module": "services.v1_client"},
        )
        r, info = self.s.score(
            (
                "VERDICT: REQUEST_CHANGES\n"
                "ISSUES: stale import from services.v1_client and deleted file services/v1_client.py\n"
                "REASON: x.\n"
            ),
            [a],
            "",
        )
        self.assertAlmostEqual(r, self.s.R_CAUGHT_STALE)
        self.assertEqual(info.get("spurious_mentions"), [])

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
        env.inject_episode(
            drifted=d,
            actions=acts,
            pr_diff="+data = getUserData(user_id)\n",
            base=base,
        )
        obs, r, done, _ = env.step(
            "VERDICT: REQUEST_CHANGES\nISSUES: getUserData\nREASON: x.\n"
        )
        self.assertTrue(done)
        self.assertEqual(obs.episode_step, 0)
        self.assertGreater(r, 0.0)

    def test_no_issues_section_no_mention_credit(self) -> None:
        """Stale symbol only in REASON must not count (malformed ISSUES)."""
        a = DriftAction(
            drift_type="rename",
            stale_ref="getUserData",
            current_ref="fetchUserData",
            metadata={},
        )
        r, info = self.s.score(
            "VERDICT: REQUEST_CHANGES\nREASON: getUserData is wrong\n",
            [a],
            "",
        )
        self.assertEqual(info["missed"], ["rename:getUserData"])
        self.assertTrue(info.get("malformed_issues"))


if __name__ == "__main__":
    unittest.main()
