"""
Smoke test: core env + optional OpenEnv bridge.

  python scripts/smoke_env.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def smoke_core() -> bool:
    from env.codedrift_env import CodeDriftEnv

    env = CodeDriftEnv(difficulty="easy", personality="random", seed=0)
    obs = env.reset()
    assert obs.prompt and obs.pr_diff
    _, reward, done, info = env.step(
        "VERDICT: REQUEST_CHANGES\nISSUES: stale refs\nREASON: test.\n"
    )
    assert done is True
    assert isinstance(reward, float)
    assert "episode_outcome" in info
    print("[ok] CodeDriftEnv reset/step")
    return True


def smoke_openenv() -> bool:
    try:
        from integrations.codedrift_openenv import OPENENV_AVAILABLE, CodeDriftOpenEnvironment
        from openenv.core.env_server.types import Action
    except ImportError as e:
        print("[skip] OpenEnv:", e)
        return True

    if not OPENENV_AVAILABLE:
        print("[skip] openenv-core not installed")
        return True

    env = CodeDriftOpenEnvironment(seed=1)
    obs = env.reset(seed=1, episode_id="smoke-1")
    assert obs.metadata.get("prompt")
    act = Action(metadata={"agent_response": "VERDICT: APPROVE\nISSUES: none\nREASON: x.\n"})
    obs2 = env.step(act)
    assert obs2.done is True
    assert obs2.reward is not None or obs2.metadata.get("scorer_info") is not None
    print("[ok] CodeDriftOpenEnvironment reset/step")
    return True


def main() -> int:
    from codedrift.logutil import configure_logging

    configure_logging()
    smoke_core()
    smoke_openenv()
    print("Smoke finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
