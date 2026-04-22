"""
DriftAgent — the adversarial agent that introduces codebase drift.

This agent is FROZEN during training (no gradient updates).
Its job is to mutate the codebase in ways that could fool the reviewer.

Personality modes control HOW it drifts:
  - subtle:      prefers contract changes (hardest to spot)
  - aggressive:  applies all 3 drift types every episode
  - random:      uniform random selection (default, good for training)
  - escalating:  starts easy, gets harder as episode count increases
  - adaptive:    adjusts strategy from reviewer win-rate (self-play curriculum)

This framing earns the Fleet AI bonus prize:
  "We trained an oversight agent (reviewer) to monitor the behavior
   of another AI agent (drift agent) operating in a shared codebase."
"""

import copy
import logging
import random
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

from codedrift.constants import DIFFICULTIES, PERSONALITIES

# ── Drift catalogue ───────────────────────────────────────────────────────────

FUNCTION_RENAMES = [
    ("getUserData", "fetchUserData"),
    ("createOrder", "submitOrder"),
    ("deleteRecord", "removeRecord"),
    ("sendNotification", "dispatchAlert"),
    ("validateInput", "sanitizeInput"),
    ("parseResponse", "deserializeResponse"),
    ("loadConfig", "readConfig"),
    ("checkPermission", "verifyAccess"),
]

REMOVABLE_FILES = [
    "utils/legacy.py",
    "helpers/deprecated.py",
    "lib/old_auth.py",
    "services/v1_client.py",
    "adapters/xml_parser.py",
    "connectors/soap_bridge.py",
]

API_CONTRACT_CHANGES = [
    {
        "function": "createOrder",
        "old_params": ["item", "qty"],
        "new_params": ["item", "qty", "userId"],
        "reason": "userId made mandatory for audit trail",
    },
    {
        "function": "sendEmail",
        "old_params": ["to", "subject", "body"],
        "new_params": ["to", "subject", "body", "template_id"],
        "reason": "template_id required for tracking",
    },
    {
        "function": "authenticate",
        "old_params": ["username", "password"],
        "new_params": ["username", "password", "mfa_token"],
        "reason": "MFA now mandatory",
    },
    {
        "function": "fetchUserData",
        "old_params": ["userId"],
        "new_params": ["userId", "include_deleted"],
        "reason": "soft-delete support added",
    },
    {
        "function": "submitOrder",
        "old_params": ["item", "qty", "userId"],
        "new_params": ["item", "qty", "userId", "warehouse_id"],
        "reason": "multi-warehouse rollout",
    },
]

@dataclass(repr=False)
class DriftAction:
    """Structured output of one drift agent action."""

    drift_type: str  # rename | removal | contract | none
    stale_ref: str  # what the PR will incorrectly use
    current_ref: str  # what the codebase now has
    metadata: dict[str, Any]  # extra info for reward scorer

    def __repr__(self) -> str:
        return f"DriftAction({self.drift_type!r}, {self.stale_ref!r} -> {self.current_ref!r})"


class DriftAgent:
    """
    Adversarial agent that introduces schema drift into the codebase.
    Frozen during training — acts as a challenging, evolving opponent
    for the code reviewer agent.
    """

    def __init__(self, personality: str = "random", seed: Optional[int] = None):
        if personality not in PERSONALITIES:
            raise ValueError(f"personality must be one of {sorted(PERSONALITIES)}, got {personality!r}")
        self.personality = personality
        self.episode_count = 0
        self.rng = random.Random(seed)
        self._reviewer_wins: list[bool] = []

    def act(self, codebase, difficulty: str = "easy") -> tuple:
        """
        Mutates codebase and returns (drifted_codebase, list[DriftAction]).
        The reviewer must catch every DriftAction in the returned list.
        """
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {sorted(DIFFICULTIES)}, got {difficulty!r}")
        drifted = copy.deepcopy(codebase)
        actions = []

        drift_types = self._pick_drift_types(difficulty)

        for dtype in drift_types:
            action = self._apply_drift(drifted, dtype)
            if action:
                actions.append(action)
                drifted.version += 1

        if len(actions) < len(drift_types):
            logger.warning(
                "drift_agent_partial_apply personality=%s difficulty=%s planned=%s applied=%s "
                "(catalog exhausted or invariants prevented some drifts)",
                self.personality,
                difficulty,
                len(drift_types),
                len(actions),
            )

        self.episode_count += 1
        return drifted, actions

    # ── Private ───────────────────────────────────────────────────────────────

    def _pick_drift_types(self, difficulty: str) -> list[str]:
        base_count = {"easy": 1, "medium": 2, "hard": 3}[difficulty]

        if self.personality == "subtle":
            # Bias toward contract in the pool, but sample *without replacement* so
            # easy=1 / medium=2 / hard=3 never repeats the same drift type in one episode.
            pool = ["contract", "contract", "rename", "removal"]
            unique_pool = list(dict.fromkeys(pool))
            k = min(base_count, len(unique_pool))
            return self.rng.sample(unique_pool, k=k)

        elif self.personality == "aggressive":
            pool = ["rename", "removal", "contract"]
            k = min(len(pool), base_count)
            return self.rng.sample(pool, k=k)

        elif self.personality == "escalating":
            level = min(self.episode_count // 10, 2)
            actual_count = level + 1
            pool = ["rename", "removal", "contract"]
            return self.rng.sample(pool, k=min(actual_count, len(pool)))

        elif self.personality == "adaptive":
            mode = self._adaptive_mode()
            if mode == "aggressive":
                pool = ["rename", "removal", "contract"]
                return self.rng.sample(pool, k=min(base_count, len(pool)))
            if mode == "subtle":
                pool = ["contract", "contract", "rename", "removal"]
                unique_pool = list(dict.fromkeys(pool))
                k = min(base_count, len(unique_pool))
                return self.rng.sample(unique_pool, k=k)
            # warmup / struggling reviewer -> random
            pool = ["rename", "removal", "contract"]
            return self.rng.sample(pool, k=min(base_count, len(pool)))

        else:  # random
            pool = ["rename", "removal", "contract"]
            return self.rng.sample(pool, k=min(base_count, len(pool)))

    def _apply_drift(self, drifted, dtype: str) -> Optional[DriftAction]:
        if dtype == "rename":
            return self._do_rename(drifted)
        elif dtype == "removal":
            return self._do_removal(drifted)
        elif dtype == "contract":
            return self._do_contract(drifted)
        return None

    def _do_rename(self, drifted) -> Optional[DriftAction]:
        candidates = [(o, n) for o, n in FUNCTION_RENAMES if o in drifted.functions]
        if not candidates:
            return None
        old_name, new_name = self.rng.choice(candidates)
        sig = drifted.functions.pop(old_name)
        drifted.functions[new_name] = sig
        return DriftAction(
            drift_type="rename",
            stale_ref=old_name,
            current_ref=new_name,
            metadata={"signature": sig},
        )

    def _do_removal(self, drifted) -> Optional[DriftAction]:
        removable = [f for f in REMOVABLE_FILES if f in drifted.files]
        if not removable:
            return None
        removed = self.rng.choice(removable)
        drifted.files.remove(removed)
        return DriftAction(
            drift_type="removal",
            stale_ref=removed,
            current_ref="[deleted]",
            metadata={"module": removed.replace("/", ".").replace(".py", "")},
        )

    def _do_contract(self, drifted) -> Optional[DriftAction]:
        candidates = [
            c
            for c in API_CONTRACT_CHANGES
            if c["function"] in drifted.api_signatures and drifted.api_signatures[c["function"]] == c["old_params"]
        ]
        if not candidates:
            return None
        change = self.rng.choice(candidates)
        fn = change["function"]
        drifted.api_signatures[fn] = change["new_params"]
        return DriftAction(
            drift_type="contract",
            stale_ref=f"{fn}({', '.join(change['old_params'])})",
            current_ref=f"{fn}({', '.join(change['new_params'])})",
            metadata={
                "function": fn,
                "old_params": change["old_params"],
                "new_params": change["new_params"],
                "reason": change.get("reason", ""),
            },
        )

    def describe(self) -> str:
        return f"DriftAgent(personality={self.personality}, episodes_run={self.episode_count})"

    def record_reviewer_result(self, reviewer_won: bool) -> None:
        """Track reviewer success for adaptive curriculum mode."""
        self._reviewer_wins.append(bool(reviewer_won))
        # Keep only recent history used by _adaptive_mode().
        if len(self._reviewer_wins) > 100:
            self._reviewer_wins = self._reviewer_wins[-100:]

    def recent_win_rate(self, window: int = 10) -> float:
        """Reviewer win rate over the most recent ``window`` episodes."""
        if window <= 0 or not self._reviewer_wins:
            return 0.0
        recent = self._reviewer_wins[-window:]
        return sum(1 for x in recent if x) / len(recent)

    def _adaptive_mode(self) -> str:
        """
        Choose strategy from reviewer performance.

        - Warmup (<5 episodes): random
        - win_rate > 0.8: aggressive
        - win_rate > 0.5: subtle
        - else: random
        """
        if len(self._reviewer_wins) < 5:
            return "random"
        wr = self.recent_win_rate(window=10)
        if wr > 0.8:
            return "aggressive"
        if wr > 0.5:
            return "subtle"
        return "random"
