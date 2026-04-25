"""Bounded replay buffer of failure cases for curriculum sampling."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional


@dataclass
class ReplayItem:
    pattern: str
    seed: int
    difficulty: str
    reward: float
    root_cause_score: float
    fingerprint: str = ""

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "reward": self.reward,
            "root_cause_score": self.root_cause_score,
            "fingerprint": self.fingerprint,
        }


class ReplayBuffer:
    """Capped FIFO buffer of episodes; dedupes by mutation fingerprint."""

    def __init__(self, capacity: int = 256):
        self.capacity = int(capacity)
        self._items: deque[ReplayItem] = deque(maxlen=self.capacity)
        self._seen: set[str] = set()

    def __len__(self) -> int:  # noqa: D401
        return len(self._items)

    def add(
        self,
        pattern: str,
        seed: int,
        difficulty: str,
        reward: float,
        root_cause_score: float,
    ) -> Optional[ReplayItem]:
        fp = _fingerprint(pattern, seed, difficulty)
        if fp in self._seen:
            return None
        item = ReplayItem(
            pattern=pattern,
            seed=int(seed),
            difficulty=difficulty,
            reward=float(reward),
            root_cause_score=float(root_cause_score),
            fingerprint=fp,
        )
        if len(self._items) == self._items.maxlen:
            evicted = self._items.popleft()
            self._seen.discard(evicted.fingerprint)
        self._items.append(item)
        self._seen.add(fp)
        return item

    def hardest(self, n: int = 1) -> list[ReplayItem]:
        if not self._items:
            return []
        return sorted(self._items, key=lambda it: it.reward)[:n]

    def items(self) -> Iterable[ReplayItem]:
        return tuple(self._items)


def _fingerprint(pattern: str, seed: int, difficulty: str) -> str:
    payload = json.dumps([pattern, int(seed), difficulty], sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
