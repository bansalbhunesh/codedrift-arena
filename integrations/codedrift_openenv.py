"""
OpenEnv bridge for CodeDrift Arena.

Install optional dependency: ``pip install openenv-core``
"""

from __future__ import annotations

from typing import Any, Optional

from env.codedrift_env import CodeDriftEnv

try:
    from openenv.core.env_server.http_server import create_fastapi_app
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import Action, EnvironmentMetadata, Observation, State

    OPENENV_AVAILABLE = True
except ImportError:  # pragma: no cover
    OPENENV_AVAILABLE = False
    Environment = None  # type: ignore[misc, assignment]
    Action = Observation = State = EnvironmentMetadata = None  # type: ignore
    create_fastapi_app = None  # type: ignore

__all__ = ["OPENENV_AVAILABLE", "CodeDriftOpenEnvironment", "build_openenv_app"]


def _pack_inner_obs(inner_obs) -> dict[str, Any]:
    return {
        "prompt": inner_obs.prompt,
        "pr_diff": inner_obs.pr_diff,
        "codebase_context": inner_obs.codebase_context,
        "episode_step": inner_obs.episode_step,
        "n_stale_refs": inner_obs.n_stale_refs,
    }


if OPENENV_AVAILABLE:

    class CodeDriftOpenEnvironment(Environment):  # type: ignore[misc, valid-type]
        """
        Wraps :class:`CodeDriftEnv` behind OpenEnv's ``Environment`` API.

        Pass the model's review in ``Action(metadata={"agent_response": "..."})``.
        """

        SUPPORTS_CONCURRENT_SESSIONS = False

        def __init__(
            self,
            difficulty: str = "easy",
            personality: str = "random",
            seed: Optional[int] = None,
            transform: Any = None,
            rubric: Any = None,
        ):
            super().__init__(transform=transform, rubric=rubric)
            self._difficulty = difficulty
            self._personality = personality
            self._seed = seed
            self._inner = CodeDriftEnv(difficulty=difficulty, personality=personality, seed=seed)
            self._state = State(episode_id=None, step_count=0)

        def get_metadata(self) -> EnvironmentMetadata:
            from . import config as cfg

            return EnvironmentMetadata(
                name="codedrift-arena",
                description="Single-step PR review RL env under adversarial codebase drift.",
                version="0.1.0",
                author=None,
                readme_content=None,
                documentation_url=cfg.HF_SPACE_URL or None,
            )

        def reset(
            self,
            seed: Optional[int] = None,
            episode_id: Optional[str] = None,
            **kwargs: Any,
        ) -> Observation:
            if seed is not None:
                self._inner = CodeDriftEnv(
                    difficulty=self._difficulty,
                    personality=self._personality,
                    seed=seed,
                )
            inner_obs = self._inner.reset()
            self._state = State(episode_id=episode_id, step_count=0)
            return Observation(
                done=False,
                reward=None,
                metadata=_pack_inner_obs(inner_obs),
            )

        def step(
            self,
            action: Action,
            timeout_s: Optional[float] = None,
            **kwargs: Any,
        ) -> Observation:
            text = str((action.metadata or {}).get("agent_response", "")).strip()
            _inner_obs, reward, done, info = self._inner.step(text)
            self._state = State(
                episode_id=self._state.episode_id,
                step_count=self._state.step_count + 1,
            )
            meta = _pack_inner_obs(_inner_obs)
            meta["scorer_info"] = info
            return Observation(done=done, reward=reward, metadata=meta)

        @property
        def state(self) -> State:
            return self._state

    def build_openenv_app():
        from openenv.core.env_server.types import Action as ActT
        from openenv.core.env_server.types import Observation as ObsT

        return create_fastapi_app(
            lambda: CodeDriftOpenEnvironment(),
            ActT,
            ObsT,
        )

else:

    class CodeDriftOpenEnvironment:  # type: ignore[no-redef]
        """Stub: install ``openenv-core`` for the real OpenEnv server class."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "CodeDriftOpenEnvironment requires openenv-core. "
                "Install with: pip install 'openenv-core'"
            )

    def build_openenv_app():  # type: ignore[no-redef]
        raise ImportError("Install openenv-core: pip install 'openenv-core'")
