"""OpenEnv bridge for the V2 Executable Code Review Arena.

Mirrors :mod:`integrations.codedrift_openenv` but wraps
:class:`env_v2.exec_arena_env.CodeReviewArenaEnv`. Use ``build_openenv_app_v2``
to mount a separate FastAPI app (e.g. behind ``/api/v2`` if you proxy it).

V1 routes are untouched.
"""

from __future__ import annotations

from typing import Any, Optional

from env_v2.exec_arena_env import CodeReviewArenaEnv

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


__all__ = [
    "OPENENV_AVAILABLE",
    "CodeReviewObservation",
    "CodeReviewArenaOpenEnv",
    "build_openenv_app_v2",
]


def _pack(inner_obs, env: Optional[CodeReviewArenaEnv]) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "prompt": inner_obs.prompt,
        "pr_diff": inner_obs.pr_diff,
        "repo_snapshot": inner_obs.repo_snapshot,
        "pytest_output": inner_obs.pytest_output,
        "failing_tests": list(inner_obs.failing_tests),
        "episode_step": inner_obs.episode_step,
        "n_bugs": inner_obs.n_bugs,
        "episode_id": inner_obs.episode_id,
    }
    if env is not None and env.episode_id:
        meta["episode_id"] = env.episode_id
    return meta


if OPENENV_AVAILABLE:

    class CodeReviewObservation(Observation):  # type: ignore[misc, valid-type]
        prompt: str = ""
        pr_diff: str = ""
        repo_snapshot: str = ""
        pytest_output: str = ""
        failing_tests: list = []
        episode_step: int = 0
        n_bugs: int = 0
        episode_id: str | None = None
        scorer_info: dict[str, Any] | None = None

    class CodeReviewArenaOpenEnv(Environment):  # type: ignore[misc, valid-type]
        SUPPORTS_CONCURRENT_SESSIONS = False

        def __init__(
            self,
            difficulty: str = "easy",
            seed: Optional[int] = None,
            allowed_patterns: Optional[list[str]] = None,
            transform: Any = None,
            rubric: Any = None,
        ):
            super().__init__(transform=transform, rubric=rubric)
            self._difficulty = difficulty
            self._seed = seed
            self._allowed = allowed_patterns
            self._inner = CodeReviewArenaEnv(
                difficulty=difficulty, seed=seed, allowed_patterns=allowed_patterns
            )
            self._state = State(episode_id=None, step_count=0)

        def get_metadata(self) -> EnvironmentMetadata:
            return EnvironmentMetadata(
                name="codedrift-arena-v2",
                description="Executable code review arena: real pytest ground truth + causal reward.",
                version="0.2.0",
                author=None,
                readme_content=None,
                documentation_url=None,
            )

        def reset(
            self,
            seed: Optional[int] = None,
            episode_id: Optional[str] = None,
            **kwargs: Any,
        ) -> CodeReviewObservation:
            if seed is not None:
                self._inner = CodeReviewArenaEnv(
                    difficulty=self._difficulty,
                    seed=seed,
                    allowed_patterns=self._allowed,
                )
            inner_obs = self._inner.reset()
            self._state = State(episode_id=episode_id or inner_obs.episode_id, step_count=0)
            meta = _pack(inner_obs, self._inner)
            return CodeReviewObservation(
                done=False,
                reward=None,
                metadata=meta,
                prompt=inner_obs.prompt,
                pr_diff=inner_obs.pr_diff,
                repo_snapshot=inner_obs.repo_snapshot,
                pytest_output=inner_obs.pytest_output,
                failing_tests=list(inner_obs.failing_tests),
                episode_step=inner_obs.episode_step,
                n_bugs=inner_obs.n_bugs,
                episode_id=inner_obs.episode_id,
                scorer_info=None,
            )

        def step(
            self,
            action: Action,
            timeout_s: Optional[float] = None,
            **kwargs: Any,
        ) -> CodeReviewObservation:
            text = str((action.metadata or {}).get("agent_response", "")).strip()
            inner_obs, reward, done, info = self._inner.step(text)
            self._state = State(
                episode_id=self._state.episode_id,
                step_count=self._state.step_count + 1,
            )
            meta = _pack(inner_obs, self._inner)
            meta["scorer_info"] = info
            return CodeReviewObservation(
                done=done,
                reward=reward,
                metadata=meta,
                prompt=inner_obs.prompt,
                pr_diff=inner_obs.pr_diff,
                repo_snapshot=inner_obs.repo_snapshot,
                pytest_output=inner_obs.pytest_output,
                failing_tests=list(inner_obs.failing_tests),
                episode_step=inner_obs.episode_step,
                n_bugs=inner_obs.n_bugs,
                episode_id=inner_obs.episode_id,
                scorer_info=info,
            )

        @property
        def state(self) -> State:
            return self._state

    def build_openenv_app_v2():
        from openenv.core.env_server.types import Action as ActT

        return create_fastapi_app(
            lambda: CodeReviewArenaOpenEnv(),
            ActT,
            CodeReviewObservation,
        )

else:
    CodeReviewObservation = None  # type: ignore[misc, assignment]

    class CodeReviewArenaOpenEnv:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError(
                "CodeReviewArenaOpenEnv requires openenv-core. Install with: pip install 'openenv-core'"
            )

    def build_openenv_app_v2():  # type: ignore[no-redef]
        raise ImportError("Install openenv-core: pip install 'openenv-core'")
