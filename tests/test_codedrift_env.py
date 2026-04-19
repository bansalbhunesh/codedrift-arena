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
    def test_episode_id_on_reset_and_step_info(self) -> None:
        env = CodeDriftEnv(difficulty="easy", seed=0)
        env.reset()
        self.assertEqual(len(env.episode_id), 12)
        _obs, _r, _d, info = env.step("VERDICT: APPROVE\nISSUES: none\nREASON: x.\n")
        self.assertEqual(info.get("episode_id"), env.episode_id)

    def test_step_accepts_none_response(self) -> None:
        env = CodeDriftEnv(difficulty="easy", seed=1)
        env.reset()
        _obs, _r, _d, _info = env.step(None)  # type: ignore[arg-type]
        self.assertEqual(_info.get("verdict"), "REQUEST_CHANGES")

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
