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


if __name__ == "__main__":
    unittest.main()
