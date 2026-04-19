"""Shared configuration values (no imports from env or agents — avoids cycles)."""

from __future__ import annotations

DIFFICULTIES = frozenset({"easy", "medium", "hard"})
PERSONALITIES = frozenset({"random", "subtle", "aggressive", "escalating"})
