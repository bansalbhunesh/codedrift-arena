"""Sanity-check V2 end-to-end: reset, score perfect prediction, score garbage."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env_v2.exec_arena_env import CodeReviewArenaEnv


def main() -> int:
    env = CodeReviewArenaEnv(difficulty="easy", seed=42)
    obs = env.reset(forced_patterns=["contract"])
    gt = env.ground_truth()
    print("episode_id:", obs.episode_id)
    print("ground truth:", json.dumps(gt, indent=2))

    perfect = json.dumps(
        {
            "verdict": gt["verdict"],
            "root_cause": gt["root_cause"],
            "failure_path": gt["failure_path"],
            "confidence": 0.9,
            "reasoning": "createOrder gained a required userId param.",
        }
    )
    _, reward, done, info = env.step(perfect)
    print("perfect reward:", reward, "outcome:", info.get("episode_outcome"))
    print("components:", info.get("reward_components"))

    env2 = CodeReviewArenaEnv(difficulty="easy", seed=42)
    env2.reset(forced_patterns=["contract"])
    _, r_bad, _, info_bad = env2.step("totally not json at all")
    print("malformed reward:", r_bad, "outcome:", info_bad.get("episode_outcome"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
