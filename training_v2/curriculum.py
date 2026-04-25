"""Curriculum + per-pattern accuracy tracker for V2 self-improvement.

Two responsibilities:
- Track rolling reviewer accuracy per bug pattern (root_cause_score >= 0.7).
- Pick the next episode's pattern + difficulty using:
    * weighted-by-weakness softmax sampling, plus
    * occasional replay of the hardest past episode (probability ``p_replay``).

Difficulty auto-promotes ``easy -> medium -> hard`` once rolling root-cause
accuracy crosses 0.7 over the recent window, and demotes if it falls below
0.4.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from agents_v2.generator_agent import ALL_PATTERNS
from training_v2.replay import ReplayBuffer, ReplayItem

DIFFICULTY_ORDER = ["easy", "medium", "hard"]
DEFAULT_WINDOW = 50
PROMOTE_THRESHOLD = 0.7
DEMOTE_THRESHOLD = 0.4


@dataclass
class _PatternStats:
    history: deque = field(default_factory=lambda: deque(maxlen=20))

    def record(self, root_cause_score: float) -> None:
        self.history.append(1.0 if root_cause_score >= 0.7 else 0.0)

    def accuracy(self) -> float:
        if not self.history:
            return 0.5  # neutral prior so untried patterns get tried.
        return sum(self.history) / len(self.history)


class Curriculum:
    """Selects (pattern, difficulty) for the next episode."""

    def __init__(
        self,
        seed: Optional[int] = None,
        allowed_patterns: Optional[list[str]] = None,
        replay_capacity: int = 256,
        p_replay: float = 0.3,
        window: int = DEFAULT_WINDOW,
    ):
        self.rng = random.Random(seed)
        self.allowed = list(allowed_patterns) if allowed_patterns else list(ALL_PATTERNS)
        self.p_replay = float(p_replay)
        self.window = int(window)
        self.replay = ReplayBuffer(capacity=replay_capacity)
        self._stats: dict[str, _PatternStats] = {p: _PatternStats() for p in self.allowed}
        self._global_history: deque = deque(maxlen=self.window)
        self._difficulty: str = "easy"
        self._next_seed: int = self.rng.randint(0, 2**31 - 1)

    @property
    def difficulty(self) -> str:
        return self._difficulty

    def accuracy_per_pattern(self) -> dict[str, float]:
        return {p: round(self._stats[p].accuracy(), 4) for p in self.allowed}

    def global_accuracy(self) -> float:
        if not self._global_history:
            return 0.0
        return sum(self._global_history) / len(self._global_history)

    def next_episode(self) -> tuple[str, str, int, bool]:
        """Return (pattern, difficulty, seed, is_replay)."""
        if self.replay.items() and self.rng.random() < self.p_replay:
            hardest = self.replay.hardest(n=1)
            if hardest:
                item = hardest[0]
                return item.pattern, item.difficulty, item.seed, True
        pattern = self._sample_weakest_pattern()
        seed = self._next_seed
        self._next_seed = self.rng.randint(0, 2**31 - 1)
        return pattern, self._difficulty, seed, False

    def record_result(
        self,
        pattern: str,
        difficulty: str,
        seed: int,
        reward: float,
        root_cause_score: float,
    ) -> ReplayItem | None:
        if pattern in self._stats:
            self._stats[pattern].record(root_cause_score)
        self._global_history.append(1.0 if root_cause_score >= 0.7 else 0.0)
        self._maybe_adjust_difficulty()
        # Only replay genuinely-hard episodes.
        if root_cause_score < 0.5:
            return self.replay.add(
                pattern=pattern,
                seed=int(seed),
                difficulty=difficulty,
                reward=float(reward),
                root_cause_score=float(root_cause_score),
            )
        return None

    def _sample_weakest_pattern(self) -> str:
        # Weight = 1 - accuracy (clipped); softmax over weights.
        weights = []
        for p in self.allowed:
            acc = self._stats[p].accuracy()
            weights.append(max(0.05, 1.0 - acc))
        total = sum(math.exp(w * 2.0) for w in weights)
        r = self.rng.random() * total
        cum = 0.0
        for p, w in zip(self.allowed, weights):
            cum += math.exp(w * 2.0)
            if r <= cum:
                return p
        return self.allowed[-1]

    def _maybe_adjust_difficulty(self) -> None:
        if len(self._global_history) < self.window:
            return
        acc = self.global_accuracy()
        idx = DIFFICULTY_ORDER.index(self._difficulty)
        if acc >= PROMOTE_THRESHOLD and idx < len(DIFFICULTY_ORDER) - 1:
            self._difficulty = DIFFICULTY_ORDER[idx + 1]
            self._global_history.clear()
        elif acc <= DEMOTE_THRESHOLD and idx > 0:
            self._difficulty = DIFFICULTY_ORDER[idx - 1]
            self._global_history.clear()

    def snapshot(self) -> dict:
        return {
            "difficulty": self._difficulty,
            "global_accuracy": round(self.global_accuracy(), 4),
            "per_pattern_accuracy": self.accuracy_per_pattern(),
            "replay_size": len(self.replay),
        }
