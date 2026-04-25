"""Executable Code Review Arena (V2) — single-step OpenEnv-shaped environment.

Episode flow:
  1. Materialize a fresh copy of ``env_v2/base_repo`` into a temp directory.
  2. Generator agent applies bug-pattern mutations (real AST rewrites).
  3. Pytest runs and produces structured failure ground truth.
  4. Reviewer (LLM) is given prompt, diff, snapshot, pytest output.
  5. ``step()`` parses the structured response, scores it via the causal
     scorer, and returns ``(obs, reward, done, info)``.

The contract intentionally mirrors :class:`env.codedrift_env.CodeDriftEnv`
so the same OpenEnv server / training rigs can host both v1 and v2.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from agents_v2.generator_agent import (
    ALL_PATTERNS,
    DIFFICULTY_TO_COUNT,
    GenerationOutcome,
    GeneratorAgent,
    MutationResult,
)
from agents_v2.prompts import build_review_prompt
from agents_v2.reviewer_io import parse_reviewer_output
from env_v2.base_repo_loader import copy_base_repo, list_source_files
from env_v2.exec_engine import ExecutionResult, FailedTest, run_pytest

logger = logging.getLogger(__name__)


@dataclass
class ObservationV2:
    prompt: str
    pr_diff: str
    repo_snapshot: str
    pytest_output: str
    failing_tests: list[str]
    episode_step: int
    n_bugs: int
    episode_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "pr_diff": self.pr_diff,
            "repo_snapshot": self.repo_snapshot,
            "pytest_output": self.pytest_output,
            "failing_tests": list(self.failing_tests),
            "episode_step": self.episode_step,
            "n_bugs": self.n_bugs,
            "episode_id": self.episode_id,
        }


@dataclass
class _EpisodeState:
    repo_dir: Path
    outcome: GenerationOutcome
    exec_result: ExecutionResult
    obs: ObservationV2
    ready: bool = True


class CodeReviewArenaEnv:
    """V2 environment with real test execution as ground truth."""

    def __init__(
        self,
        difficulty: str = "easy",
        seed: Optional[int] = None,
        allowed_patterns: Optional[list[str]] = None,
        timeout_s: int = 30,
        keep_episode_dirs: bool = False,
        scorer: Any = None,
    ):
        if difficulty not in DIFFICULTY_TO_COUNT:
            raise ValueError(f"unknown difficulty {difficulty!r}")
        self.difficulty = difficulty
        self.seed = seed
        self.allowed_patterns = list(allowed_patterns) if allowed_patterns else list(ALL_PATTERNS)
        self.timeout_s = int(timeout_s)
        self.keep_episode_dirs = bool(keep_episode_dirs)
        self.generator = GeneratorAgent(seed=seed, allowed_patterns=self.allowed_patterns)

        # Lazy-imported to avoid circular dep at module load.
        if scorer is None:
            from rewards_v2.causal_scorer import CausalScorer

            scorer = CausalScorer()
        self.scorer = scorer
        self._state: Optional[_EpisodeState] = None

    # ── Public API ────────────────────────────────────────────────────────

    def reset(
        self,
        forced_patterns: Optional[list[str]] = None,
        difficulty_override: Optional[str] = None,
    ) -> ObservationV2:
        self._cleanup_previous()
        difficulty = difficulty_override or self.difficulty
        episode_id = uuid.uuid4().hex[:12]
        td = Path(tempfile.mkdtemp(prefix=f"v2_ep_{episode_id}_"))
        repo_dir = copy_base_repo(td / "repo")
        outcome = self.generator.generate(
            repo_dir, difficulty=difficulty, forced_patterns=forced_patterns
        )
        started = time.perf_counter()
        exec_result = run_pytest(repo_dir, timeout_s=self.timeout_s)
        logger.info(
            "v2_episode_reset id=%s difficulty=%s mutations=%s failed=%s exec_s=%.2f",
            episode_id,
            difficulty,
            len(outcome.mutations),
            exec_result.failed,
            time.perf_counter() - started,
        )

        snapshot = self._format_snapshot(repo_dir)
        prompt = build_review_prompt(
            pr_diff=outcome.pr_diff,
            repo_snapshot=snapshot,
            pytest_output=exec_result.stdout_tail,
            failing_test_ids=[t.nodeid for t in exec_result.failed_tests],
        )
        obs = ObservationV2(
            prompt=prompt,
            pr_diff=outcome.pr_diff,
            repo_snapshot=snapshot,
            pytest_output=exec_result.stdout_tail,
            failing_tests=[t.nodeid for t in exec_result.failed_tests],
            episode_step=0,
            n_bugs=len(outcome.mutations),
            episode_id=episode_id,
        )
        self._state = _EpisodeState(
            repo_dir=repo_dir, outcome=outcome, exec_result=exec_result, obs=obs, ready=True
        )
        return obs

    def step(self, agent_response: str) -> tuple[ObservationV2, float, bool, dict[str, Any]]:
        if self._state is None or not self._state.ready:
            raise RuntimeError("Call reset() before step() (episode already consumed).")
        prediction = parse_reviewer_output(agent_response)
        ground_truth = self._build_ground_truth()
        reward, info = self.scorer.score(
            prediction=prediction,
            ground_truth=ground_truth,
            exec_result=self._state.exec_result,
            mutations=self._state.outcome.mutations,
        )
        info.setdefault("episode_id", self._state.obs.episode_id)
        info.setdefault("difficulty", self.difficulty)
        info.setdefault("n_bugs", self._state.obs.n_bugs)
        info.setdefault("requested_patterns", list(self._state.outcome.requested))
        info.setdefault("failing_tests", list(self._state.obs.failing_tests))
        self._state.ready = False
        return self._state.obs, float(reward), True, info

    def render(self) -> str:
        if self._state is None:
            return "v2 arena: no active episode"
        muts = "\n".join(
            f"  [{m.pattern}] {m.root_cause} {m.detail}" for m in self._state.outcome.mutations
        )
        return (
            f"=== V2 Arena Episode {self._state.obs.episode_id} ===\n"
            f"difficulty={self.difficulty} n_bugs={self._state.obs.n_bugs}\n"
            f"mutations:\n{muts}\n"
            f"failing_tests={self._state.obs.failing_tests}\n"
        )

    # ── Properties / helpers ─────────────────────────────────────────────

    @property
    def is_ready_for_step(self) -> bool:
        return self._state is not None and self._state.ready

    @property
    def episode_id(self) -> str:
        return self._state.obs.episode_id if self._state else ""

    def ground_truth(self) -> dict[str, Any]:
        return self._build_ground_truth()

    def debug_snapshot(self) -> dict[str, Any]:
        if self._state is None:
            return {"episode_ready": False}
        return {
            "episode_id": self._state.obs.episode_id,
            "difficulty": self.difficulty,
            "n_bugs": self._state.obs.n_bugs,
            "mutations": [asdict(m) for m in self._state.outcome.mutations],
            "failing_tests": list(self._state.obs.failing_tests),
            "ready": self._state.ready,
        }

    def _build_ground_truth(self) -> dict[str, Any]:
        assert self._state is not None
        muts = self._state.outcome.mutations
        exec_result = self._state.exec_result
        primary = muts[0] if muts else None
        return {
            "verdict": "REQUEST_CHANGES" if exec_result.failed + exec_result.errors > 0 else "APPROVE",
            "root_cause": primary.root_cause if primary else "",
            "root_cause_symbols": [m.root_cause for m in muts],
            "patterns": [m.pattern for m in muts],
            "failure_path": _build_failure_path(muts, exec_result.failed_tests),
            "failing_tests": [t.nodeid for t in exec_result.failed_tests],
        }

    def _format_snapshot(self, repo_dir: Path) -> str:
        lines = ["=== src/ ==="]
        for src in list_source_files(repo_dir):
            text = src.read_text(encoding="utf-8")
            rel = str(src.relative_to(repo_dir)).replace("\\", "/")
            lines.append(f"--- {rel} ---")
            lines.append(text)
        return "\n".join(lines)

    def _cleanup_previous(self) -> None:
        if self._state is None or self.keep_episode_dirs:
            return
        try:
            shutil.rmtree(self._state.repo_dir.parent, ignore_errors=True)
        except Exception:
            logger.exception("v2_cleanup_failed dir=%s", self._state.repo_dir)


def _build_failure_path(
    mutations: list[MutationResult], failed_tests: list[FailedTest]
) -> list[str]:
    """Synthesize a canonical ground-truth path: test -> intermediate -> root."""
    if not mutations:
        return []
    primary = mutations[0]
    if not failed_tests:
        return [primary.root_cause]
    test_id = failed_tests[0].nodeid
    return [test_id, primary.symbol, primary.root_cause]
