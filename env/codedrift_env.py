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
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from agents.drift_agent import DriftAction, DriftAgent
from codedrift.constants import DIFFICULTIES, PERSONALITIES
from env.codebase import CodebaseState, build_base_codebase
from env.pr_generator import PRDiffGenerator
from rewards.scorer import RewardScorer

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
        self._episode_id: str = ""

    def _new_episode_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def reset(self) -> Observation:
        """Start a new episode. Returns initial observation."""
        self._episode_id = self._new_episode_id()
        self._base = build_base_codebase()
        self._drifted, self._actions = self.drift_agent.act(self._base, self.difficulty)
        self._pr_diff = self.pr_gen.generate(self._actions)
        self._step = 0
        self._episode_ready = True
        self._cached_reset_obs = self._build_obs()
        logger.info(
            "episode_reset episode_id=%s difficulty=%s personality=%s n_stale=%s",
            self._episode_id,
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
        self._episode_id = self._new_episode_id()
        self._base = build_base_codebase()
        self._drifted = self._base.clone()
        self._actions = []
        self._pr_diff = pr_diff
        self._step = 0
        self._episode_ready = True
        self._cached_reset_obs = self._build_obs()
        logger.info("set_clean_episode episode_id=%s n_stale=0", self._episode_id)
        return self._cached_reset_obs

    def inject_episode(
        self,
        *,
        drifted: CodebaseState,
        actions: list[DriftAction],
        pr_diff: str,
        base: Optional[CodebaseState] = None,
        validate: bool = True,
    ) -> Observation:
        """
        Scripted episode (demos / tests): supply drifted state, ground-truth
        actions, and diff. Does not run DriftAgent.

        ``base`` defaults to a fresh :func:`build_base_codebase` (for context only).
        When ``validate`` is True, checks that actions match ``drifted`` and that
        ``pr_diff`` plausibly contains stale references.
        """
        drifted = copy.deepcopy(drifted)
        act_list = copy.deepcopy(actions)
        if validate and act_list:
            _validate_injected_episode(drifted, act_list, pr_diff)
        self._episode_id = self._new_episode_id()
        self._base = copy.deepcopy(base) if base is not None else build_base_codebase()
        self._drifted = drifted
        self._actions = act_list
        self._pr_diff = pr_diff
        self._step = 0
        self._episode_ready = True
        self._cached_reset_obs = self._build_obs()
        logger.info("inject_episode episode_id=%s n_stale=%s", self._episode_id, len(self._actions))
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

        if agent_response is None:
            agent_response = ""

        reward, info = self.scorer.score(
            agent_response=agent_response,
            actions=self._actions,
            pr_diff=self._pr_diff,
        )
        info.setdefault("episode_id", self._episode_id)
        self._step += 1
        done = True
        logger.info(
            "episode_step episode_id=%s reward=%.3f outcome=%s verdict=%s",
            self._episode_id,
            reward,
            info.get("episode_outcome"),
            info.get("verdict"),
        )
        # Optional self-play curriculum signal for DriftAgent(personality="adaptive").
        reviewer_won = bool(info.get("episode_outcome") == "perfect")
        self.drift_agent.record_reviewer_result(reviewer_won)
        self._episode_ready = False
        return self._cached_reset_obs, reward, done, info

    def render(self) -> str:
        """Human-readable episode summary — useful for debugging."""
        lines = [
            "-- CodeDrift Arena ---------------------------------",
            f"Episode id:  {self._episode_id or '(none)'}",
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

    @property
    def episode_id(self) -> str:
        return self._episode_id

    @property
    def is_ready_for_step(self) -> bool:
        """Whether the current episode can accept exactly one ``step`` call."""
        return self._episode_ready

    def debug_snapshot(self) -> dict[str, Any]:
        """Compact state for demos / remote debugging (no large blobs)."""
        return {
            "episode_id": self._episode_id,
            "episode_ready": self.is_ready_for_step,
            "difficulty": self.difficulty,
            "n_stale_refs": len(self._actions),
            "drift_version": getattr(self._drifted, "version", None),
            "step_counter": self._step,
        }

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


def _validate_injected_episode(drifted: CodebaseState, actions: list[DriftAction], pr_diff: str) -> None:
    """Raise ValueError if scripted episode is internally inconsistent."""
    if not pr_diff.strip():
        raise ValueError("inject_episode: pr_diff must be non-empty")

    for a in actions:
        if a.drift_type == "rename":
            if a.stale_ref in drifted.functions:
                raise ValueError(
                    f"rename invariant: stale_ref {a.stale_ref!r} must not appear in drifted.functions"
                )
            if a.current_ref not in drifted.functions:
                raise ValueError(
                    f"rename invariant: current_ref {a.current_ref!r} must appear in drifted.functions"
                )
            bare = a.stale_ref.split("(")[0]
            if bare not in pr_diff and a.stale_ref not in pr_diff:
                raise ValueError(f"rename invariant: stale symbol {bare!r} not found in pr_diff")
        elif a.drift_type == "removal":
            if a.stale_ref in drifted.files:
                raise ValueError(f"removal invariant: path {a.stale_ref!r} must not be in drifted.files")
            mod = a.metadata.get("module", "")
            in_diff = a.stale_ref in pr_diff
            if mod:
                in_diff = in_diff or mod in pr_diff or mod.replace(".", "/") in pr_diff
            if not in_diff:
                raise ValueError("removal invariant: stale path or module not found in pr_diff")
        elif a.drift_type == "contract":
            fn = a.metadata.get("function")
            new_params = a.metadata.get("new_params")
            if not fn or new_params is None:
                raise ValueError("contract action requires metadata.function and new_params")
            if drifted.api_signatures.get(fn) != list(new_params):
                raise ValueError(
                    f"contract invariant: drifted.api_signatures[{fn!r}] must equal action new_params"
                )
            if fn not in pr_diff and a.stale_ref not in pr_diff:
                raise ValueError(f"contract invariant: function or stale_ref not found in pr_diff")
        else:
            raise ValueError(f"unknown drift_type: {a.drift_type!r}")
