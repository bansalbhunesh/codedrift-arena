"""DriftAgent behaviour (personality / difficulty invariants)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.drift_agent import DriftAgent


class TestDriftAgent(unittest.TestCase):
    def test_aggressive_easy_returns_one_drift_type(self) -> None:
        """Aggressive must respect difficulty cap (easy => one planned drift type)."""
        for seed in range(50):
            agent = DriftAgent(personality="aggressive", seed=seed)
            picked = agent._pick_drift_types("easy")
            self.assertEqual(len(picked), 1, msg=f"seed={seed} picked={picked!r}")

    def test_subtle_medium_unique_drift_types(self) -> None:
        """Subtle must not pick the same drift type twice on medium (base_count=2)."""
        for seed in range(400):
            agent = DriftAgent(personality="subtle", seed=seed)
            picked = agent._pick_drift_types("medium")
            self.assertEqual(len(picked), 2, msg=f"seed={seed}")
            self.assertEqual(len(picked), len(set(picked)), msg=f"seed={seed} picked={picked!r}")

    def test_subtle_hard_at_most_three_types(self) -> None:
        for seed in range(200):
            agent = DriftAgent(personality="subtle", seed=seed)
            picked = agent._pick_drift_types("hard")
            self.assertLessEqual(len(picked), 3)
            self.assertEqual(len(picked), len(set(picked)), msg=f"seed={seed} picked={picked!r}")

    def test_adaptive_mode_thresholds(self) -> None:
        agent = DriftAgent(personality="adaptive", seed=0)
        # Warmup path.
        self.assertEqual(agent._adaptive_mode(), "random")

        # Strong reviewer => aggressive.
        for _ in range(9):
            agent.record_reviewer_result(True)
        for _ in range(1):
            agent.record_reviewer_result(False)
        self.assertEqual(agent._adaptive_mode(), "aggressive")

        # Mid reviewer => subtle.
        agent = DriftAgent(personality="adaptive", seed=1)
        pattern = [True, True, True, True, True, True, False, False, False, False]  # 60%
        for v in pattern:
            agent.record_reviewer_result(v)
        self.assertEqual(agent._adaptive_mode(), "subtle")

        # Struggling reviewer => random.
        agent = DriftAgent(personality="adaptive", seed=2)
        pattern = [True, False, False, False, False, False, False, False, False, False]  # 10%
        for v in pattern:
            agent.record_reviewer_result(v)
        self.assertEqual(agent._adaptive_mode(), "random")

    def test_adaptive_pick_respects_difficulty_cap(self) -> None:
        agent = DriftAgent(personality="adaptive", seed=3)
        for _ in range(10):
            agent.record_reviewer_result(True)  # drive aggressive branch
        picked = agent._pick_drift_types("easy")
        self.assertEqual(len(picked), 1)
        self.assertIn(picked[0], {"rename", "removal", "contract"})


if __name__ == "__main__":
    unittest.main()
