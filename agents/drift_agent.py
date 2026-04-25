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
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

from codedrift.constants import DIFFICULTIES, PERSONALITIES
from env.bug_patterns import (
    CONDITION_FLIP_CASES,
    NULL_MISSING_CASES,
    OFF_BY_ONE_CASES,
    PARTIAL_RENAME_CASES,
    TYPE_MISMATCH_CASES,
)

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

    drift_type: str        # underlying mutation class: rename | removal | contract
    stale_ref: str         # what the PR will incorrectly use
    current_ref: str       # what the codebase now has
    metadata: dict[str, Any]  # extra info for reward scorer
    bug_pattern: str = field(default="")  # semantic pattern name (new; empty = legacy)

    def __repr__(self) -> str:
        pat = f" [{self.bug_pattern}]" if self.bug_pattern else ""
        return f"DriftAction({self.drift_type!r}{pat}, {self.stale_ref!r} -> {self.current_ref!r})"


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
        self._type_wins: dict[str, list[bool]] = {}  # per drift-type win history

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
            # Random, but bias toward the pattern the reviewer fails on most.
            pool = [
                "rename", "removal", "contract",
                "partial_rename", "null_missing", "type_mismatch",
                "condition_flip", "off_by_one",
            ]
            weakest = self.weakest_type()
            if weakest and weakest in pool and base_count == 1:
                # 65% chance to target the reviewer's known weak spot
                if self.rng.random() < 0.65:
                    return [weakest]
            return self.rng.sample(pool, k=min(base_count, len(pool)))

        else:  # random — include new realistic patterns
            pool = [
                "rename", "removal", "contract",
                "partial_rename", "null_missing", "type_mismatch",
                "condition_flip", "off_by_one",
            ]
            return self.rng.sample(pool, k=min(base_count, len(pool)))

    def _apply_drift(self, drifted, dtype: str) -> Optional[DriftAction]:
        """Dispatch to the correct mutation method by pattern/drift name."""
        if dtype == "partial_rename":
            return self._do_partial_rename(drifted)
        elif dtype == "null_missing":
            return self._do_null_missing(drifted)
        elif dtype == "type_mismatch":
            return self._do_type_mismatch(drifted)
        elif dtype == "condition_flip":
            return self._do_condition_flip(drifted)
        elif dtype == "off_by_one":
            return self._do_off_by_one(drifted)
        elif dtype == "rename":
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

    # ── New realistic bug patterns ────────────────────────────────────────────

    def _do_partial_rename(self, drifted) -> Optional[DriftAction]:
        """Rename a function, but the PR diff will show one stale call remaining."""
        candidates = [(c["old_name"], c["new_name"], c) for c in PARTIAL_RENAME_CASES if c["old_name"] in drifted.functions]
        if not candidates:
            # Fall back to standard rename if no partial-rename candidates available
            return self._do_rename(drifted)
        old_name, new_name, case = self.rng.choice(candidates)
        sig = drifted.functions.pop(old_name)
        drifted.functions[new_name] = sig
        return DriftAction(
            drift_type="rename",
            stale_ref=old_name,
            current_ref=new_name,
            metadata={
                "signature": sig,
                "stale_context": case["stale_context"],
                "fresh_context": case["fresh_context"],
            },
            bug_pattern="partial_rename",
        )

    def _do_null_missing(self, drifted) -> Optional[DriftAction]:
        """Change a function's return type to Optional; PR won't guard against None."""
        candidates = [c for c in NULL_MISSING_CASES if c["function"] in drifted.functions and c["function"] not in drifted.nullable_returns]
        if not candidates:
            # Fall back to contract change if no null-missing candidates
            return self._do_contract(drifted)
        case = self.rng.choice(candidates)
        fn = case["function"]
        drifted.nullable_returns.add(fn)
        return DriftAction(
            drift_type="contract",
            stale_ref=f"{fn}().{case['nullable_attribute']}",
            current_ref=f"({fn}() or {{}}).{case['nullable_attribute']}",
            metadata={
                "function": fn,
                "nullable_attribute": case["nullable_attribute"],
                "old_return": case["old_return"],
                "new_return": case["new_return"],
                "reason": case["reason"],
                "test_suite": case["test_suite"],
                "test_name": case["test_name"],
                "caller": case["caller"],
            },
            bug_pattern="null_missing",
        )

    def _do_type_mismatch(self, drifted) -> Optional[DriftAction]:
        """Change a parameter's type; PR still passes the old type."""
        candidates = [
            c for c in TYPE_MISMATCH_CASES
            if c["function"] in drifted.functions
            and drifted.param_types.get(c["function"], {}).get(c["param"]) == c["old_type"]
        ]
        if not candidates:
            return self._do_contract(drifted)
        case = self.rng.choice(candidates)
        fn, param = case["function"], case["param"]
        drifted.param_types.setdefault(fn, {})[param] = case["new_type"]
        return DriftAction(
            drift_type="contract",
            stale_ref=f"{fn}({param}={case['old_example']})",
            current_ref=f"{fn}({param}={case['new_example']!r})",
            metadata={
                "function": fn,
                "param": param,
                "old_type": case["old_type"],
                "new_type": case["new_type"],
                "old_example": case["old_example"],
                "new_example": case["new_example"],
                "reason": case["reason"],
                "test_suite": case["test_suite"],
                "test_name": case["test_name"],
                "caller": case["caller"],
            },
            bug_pattern="type_mismatch",
        )

    def _do_condition_flip(self, drifted) -> Optional[DriftAction]:
        """Invert a boolean parameter's semantics; PR passes the now-wrong value."""
        candidates = [c for c in CONDITION_FLIP_CASES if c["function"] in drifted.functions]
        if not candidates:
            return self._do_contract(drifted)
        case = self.rng.choice(candidates)
        fn = case["function"]
        return DriftAction(
            drift_type="contract",
            stale_ref=f"{fn}({case['param']}={case['old_value']})",
            current_ref=f"{fn}({case['param']}={case['new_correct_value']})",
            metadata={
                "function": fn,
                "param": case["param"],
                "old_value": case["old_value"],
                "new_correct_value": case["new_correct_value"],
                "old_semantics": case["old_semantics"],
                "new_semantics": case["new_semantics"],
                "reason": case["reason"],
                "test_suite": case["test_suite"],
                "test_name": case["test_name"],
                "caller": case["caller"],
            },
            bug_pattern="condition_flip",
        )

    def _do_off_by_one(self, drifted) -> Optional[DriftAction]:
        """Change index convention (1-based → 0-based); PR uses old offset."""
        candidates = [c for c in OFF_BY_ONE_CASES if c["function"] in drifted.functions]
        if not candidates:
            return self._do_contract(drifted)
        case = self.rng.choice(candidates)
        fn = case["function"]
        return DriftAction(
            drift_type="contract",
            stale_ref=case["old_call"],
            current_ref=case["new_correct_call"],
            metadata={
                "function": fn,
                "param": case["param"],
                "old_convention": case["old_convention"],
                "new_convention": case["new_convention"],
                "old_call": case["old_call"],
                "new_correct_call": case["new_correct_call"],
                "reason": case["reason"],
                "test_suite": case["test_suite"],
                "test_name": case["test_name"],
                "caller": case["caller"],
            },
            bug_pattern="off_by_one",
        )

    # Backward-compatible aliases for legacy smoke scripts that call _apply_*.
    # Keep these thin wrappers so external judge scripts don't break.
    def _apply_partial_rename(self, drifted) -> Optional[DriftAction]:
        return self._do_partial_rename(drifted)

    def _apply_null_missing(self, drifted) -> Optional[DriftAction]:
        return self._do_null_missing(drifted)

    def _apply_type_mismatch(self, drifted) -> Optional[DriftAction]:
        return self._do_type_mismatch(drifted)

    def _apply_condition_flip(self, drifted) -> Optional[DriftAction]:
        return self._do_condition_flip(drifted)

    def _apply_off_by_one(self, drifted) -> Optional[DriftAction]:
        return self._do_off_by_one(drifted)

    def describe(self) -> str:
        return f"DriftAgent(personality={self.personality}, episodes_run={self.episode_count})"

    def record_reviewer_result(self, reviewer_won: bool, drift_types: list[str] | None = None) -> None:
        """Track reviewer success for adaptive curriculum mode."""
        self._reviewer_wins.append(bool(reviewer_won))
        if len(self._reviewer_wins) > 100:
            self._reviewer_wins = self._reviewer_wins[-100:]

        # Per-type tracking powers the targeted adversary strategy.
        if drift_types:
            for dtype in drift_types:
                wins = self._type_wins.setdefault(dtype, [])
                wins.append(bool(reviewer_won))
                if len(wins) > 50:
                    self._type_wins[dtype] = wins[-50:]

    def failure_type_win_rates(self, window: int = 10) -> dict[str, float]:
        """Reviewer win rate per drift type over the recent ``window``."""
        result: dict[str, float] = {}
        for dtype, wins in self._type_wins.items():
            recent = wins[-window:] if wins else []
            result[dtype] = sum(1 for w in recent if w) / len(recent) if recent else 0.5
        return result

    def weakest_type(self) -> str | None:
        """Drift type the reviewer fails on most (returns None if insufficient data)."""
        rates = {k: v for k, v in self.failure_type_win_rates().items() if len(self._type_wins.get(k, [])) >= 3}
        if not rates:
            return None
        return min(rates, key=rates.__getitem__)

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

    def adaptive_snapshot(self) -> dict[str, Any]:
        """
        UI-friendly state for the "Adversary Brain" panel.

        Returns empty dict for non-adaptive personalities.
        """
        if self.personality != "adaptive":
            return {}
        per_mode = {
            "random": self.recent_win_rate(window=5),
            "subtle": self.recent_win_rate(window=10),
            "aggressive": self.recent_win_rate(window=20),
        }
        stage = self._adaptive_mode()
        type_rates = self.failure_type_win_rates()
        return {
            "enabled": True,
            "stage": stage,
            "episodes_run": self.episode_count,
            "history_len": len(self._reviewer_wins),
            "recent_win_rate_5": self.recent_win_rate(window=5),
            "recent_win_rate_10": self.recent_win_rate(window=10),
            "recent_win_rate_20": self.recent_win_rate(window=20),
            "mode_scores": per_mode,
            "type_win_rates": type_rates,
            "weakest_type": self.weakest_type(),
        }
