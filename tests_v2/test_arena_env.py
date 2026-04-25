"""Full V2 arena env contract: reset+step, structured ground truth, replay protection."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env_v2.exec_arena_env import CodeReviewArenaEnv


class TestArenaEnv(unittest.TestCase):
    def test_reset_returns_consistent_observation(self) -> None:
        env = CodeReviewArenaEnv(difficulty="easy", seed=101)
        obs = env.reset(forced_patterns=["contract"])
        self.assertTrue(obs.episode_id)
        self.assertEqual(obs.episode_step, 0)
        self.assertEqual(obs.n_bugs, 1)
        self.assertIn("PR DIFF", obs.prompt)
        self.assertIn("PYTEST OUTPUT", obs.prompt)

    def test_perfect_response_yields_high_reward(self) -> None:
        env = CodeReviewArenaEnv(difficulty="easy", seed=102)
        obs = env.reset(forced_patterns=["contract"])
        gt = env.ground_truth()
        response = json.dumps(
            {
                "verdict": gt["verdict"],
                "root_cause": gt["root_cause"],
                "failure_path": gt["failure_path"],
                "confidence": 0.85,
            }
        )
        _, reward, done, info = env.step(response)
        self.assertTrue(done)
        self.assertGreater(reward, 1.5)
        self.assertEqual(info["episode_outcome"], "perfect")

    def test_step_after_consume_raises(self) -> None:
        env = CodeReviewArenaEnv(difficulty="easy", seed=103)
        env.reset(forced_patterns=["contract"])
        env.step("{}")
        with self.assertRaises(RuntimeError):
            env.step("{}")

    def test_malformed_step_returns_negative_reward(self) -> None:
        env = CodeReviewArenaEnv(difficulty="easy", seed=104)
        env.reset(forced_patterns=["contract"])
        _, reward, _, info = env.step("not json at all")
        self.assertLess(reward, 0.0)
        self.assertEqual(info["episode_outcome"], "malformed")


if __name__ == "__main__":
    unittest.main()
