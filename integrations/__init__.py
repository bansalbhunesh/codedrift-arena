"""Integrations (OpenEnv bridge, deployment config)."""

from typing import Any

__all__ = [
    "OPENENV_AVAILABLE",
    "CodeDriftObservation",
    "CodeDriftOpenEnvironment",
    "build_openenv_app",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from integrations import codedrift_openenv as m

        return getattr(m, name)
    raise AttributeError(name)
