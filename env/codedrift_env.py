"""
CodeDriftEnv — main RL environment.

Compatible with OpenEnv interface (reset/step/render).
The orchestrator agent (code reviewer) is TRAINED.
The drift agent is FROZEN — it acts adversarially but receives no gradient.

Episode flow:
  1. Base codebase generated
  2. DriftAgent.act() mutates codebase → stale refs recorded
  3. PRDiffGenerator embeds stale refs into realistic PR diff
  4. Reviewer agent receives: current codebase + PR diff
  5. Reviewer responds: VERDICT + ISSUES + REASON
  6. RewardScorer grades the response deterministically
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Optional

from agents.drift_agent import DriftAction, DriftAgent
from env.codebase import CodebaseState, build_base_codebase
from env.pr_generator import PRDiffGenerator
from rewards.scorer import RewardScorer

DIFFICULTIES = frozenset({"easy", "medium", "hard"})
PERSONALITIES = frozenset({"random", "subtle", "aggressive", "escalating"})

logger = logging.getLogger(__name__)

REVIEWER_SYSTEM_PROMPT = """You are a senior software engineer doing a code review.

You have been given:
1. The CURRENT state of the codebase (functions, files, APIs)
2. A PR diff proposing new changes

Your job: carefully check whether the PR references anything that no longer
exists or has changed in the current codebase (renamed functions, deleted files,
updated API signatures).

Respond EXACTLY in this format — no other text:

VERDICT: [APPROVE | REQUEST_CHANGES]
ISSUES: [describe every stale reference found, or write 'none']
REASON: [one sentence explaining your verdict]"""


@dataclass
class Observation:
    prompt: str  # full prompt ready to feed to the LLM
    pr_diff: str  # raw diff text
    codebase_context: str  # formatted codebase state
    episode_step: int
    n_stale_refs: int  # count (not content — agent doesn't see this)


class CodeDriftEnv:
    """
    Main environment class. Instantiate once and call reset() per episode.

    Use :meth:`set_clean_episode` or :meth:`inject_episode` for scripted rows
    (training / demos) instead of assigning private attributes.
    """

    def __init__(
        self,
        difficulty: str = "easy",
        personality: str = "random",
        seed: Optional[int] = None,
    ):
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {sorted(DIFFICULTIES)}, got {difficulty!r}")
        if personality not in PERSONALITIES:
            raise ValueError(f"personality must be one of {sorted(PERSONALITIES)}, got {personality!r}")
        self.difficulty = difficulty
        self.drift_agent = DriftAgent(personality=personality, seed=seed)
        self.pr_gen = PRDiffGenerator(seed=seed)
        self.scorer = RewardScorer()

        self._base: Optional[CodebaseState] = None
        self._drifted: Optional[CodebaseState] = None
        self._actions: list[DriftAction] = []
        self._pr_diff: str = ""
        self._step: int = 0
        self._episode_ready: bool = False
        self._cached_reset_obs: Optional[Observation] = None

    def reset(self) -> Observation:
        """Start a new episode. Returns initial observation."""
        self._base = build_base_codebase()
        self._drifted, self._actions = self.drift_agent.act(self._base, self.difficulty)
        self._pr_diff = self.pr_gen.generate(self._actions)
        self._step = 0
        self._episode_ready = True
        self._cached_reset_obs = self._build_obs()
        logger.info(
            "episode_reset difficulty=%s personality=%s n_stale=%s",
            self.difficulty,
            self.drift_agent.personality,
            len(self._actions),
        )
        return self._cached_reset_obs

    def set_clean_episode(self, pr_diff: str) -> Observation:
        """
        Scripted **clean PR** row: no drift, canonical codebase, custom diff text.

        Sets invariants and ``_episode_ready`` so :meth:`step` is valid.
        """
        self._base = build_base_codebase()
        self._drifted = self._base.clone()
        self._actions = []
        self._pr_diff = pr_diff
        self._step = 0
        self._episode_ready = True
        self._cached_reset_obs = self._build_obs()
        logger.info("set_clean_episode n_stale=0")
        return self._cached_reset_obs

    def inject_episode(
        self,
        *,
        drifted: CodebaseState,
        actions: list[DriftAction],
        pr_diff: str,
        base: Optional[CodebaseState] = None,
    ) -> Observation:
        """
        Scripted episode (demos / tests): supply drifted state, ground-truth
        actions, and diff. Does not run DriftAgent.

        ``base`` defaults to a fresh :func:`build_base_codebase` (for context only).
        """
        self._base = copy.deepcopy(base) if base is not None else build_base_codebase()
        self._drifted = copy.deepcopy(drifted)
        self._actions = copy.deepcopy(actions)
        self._pr_diff = pr_diff
        self._step = 0
        self._episode_ready = True
        self._cached_reset_obs = self._build_obs()
        logger.info("inject_episode n_stale=%s", len(self._actions))
        return self._cached_reset_obs

    def step(self, agent_response: str) -> tuple[Observation, float, bool, dict]:
        """
        Process agent response and return (obs, reward, done, info).

        Single-step env: ``obs`` is the **same** observation as after
        :meth:`reset` / inject (episode_step stays 0) so callers are not
        misled by a post-step rebuild.
        """
        if not self._episode_ready or self._drifted is None:
            raise RuntimeError("Call reset(), set_clean_episode(), or inject_episode() before step().")
        if self._cached_reset_obs is None:
            raise RuntimeError("Internal error: missing cached reset observation.")

        reward, info = self.scorer.score(
            agent_response=agent_response,
            actions=self._actions,
            pr_diff=self._pr_diff,
        )
        self._step += 1
        done = True
        logger.info(
            "episode_step reward=%.3f outcome=%s verdict=%s",
            reward,
            info.get("episode_outcome"),
            info.get("verdict"),
        )
        return self._cached_reset_obs, reward, done, info

    def render(self) -> str:
        """Human-readable episode summary — useful for debugging."""
        lines = [
            "-- CodeDrift Arena ---------------------------------",
            f"Difficulty:  {self.difficulty}",
            f"Drift agent: {self.drift_agent.describe()}",
            f"Stale refs:  {len(self._actions)}",
        ]
        for a in self._actions:
            lines.append(f"  [{a.drift_type}] {a.stale_ref} -> {a.current_ref}")
        lines.append("PR diff:")
        lines.append(self._pr_diff)
        lines.append("Codebase (drifted):")
        lines.append(self._format_codebase(self._drifted))
        return "\n".join(lines)

    @property
    def stale_actions(self) -> list[DriftAction]:
        """Ground truth stale refs for this episode (used by reward fn)."""
        return self._actions

    @property
    def pr_diff(self) -> str:
        return self._pr_diff

    def _build_obs(self) -> Observation:
        ctx = self._format_codebase(self._drifted)
        prompt = self._build_prompt(ctx)
        return Observation(
            prompt=prompt,
            pr_diff=self._pr_diff,
            codebase_context=ctx,
            episode_step=self._step,
            n_stale_refs=len(self._actions),
        )

    def _format_codebase(self, cb: Optional[CodebaseState]) -> str:
        if not cb:
            return ""
        lines = ["=== CURRENT CODEBASE ==="]
        lines.append("Available functions:")
        for name, sig in sorted(cb.functions.items()):
            lines.append(f"  def {name}({sig})")
        lines.append("Available files:")
        for f in sorted(cb.files):
            lines.append(f"  {f}")
        lines.append("API signatures:")
        for name, params in sorted(cb.api_signatures.items()):
            lines.append(f"  {name}({', '.join(params)})")
        return "\n".join(lines)

    def _build_prompt(self, codebase_ctx: str) -> str:
        return (
            f"{REVIEWER_SYSTEM_PROMPT}\n\n"
            f"{codebase_ctx}\n\n"
            f"=== PR DIFF ===\n{self._pr_diff}\n\n"
            f"Your review:"
        )
