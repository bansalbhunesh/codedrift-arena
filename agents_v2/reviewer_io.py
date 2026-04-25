"""Strict structured-output parser for the V2 reviewer.

Accepts either:
- a JSON object matching the schema in :mod:`agents_v2.prompts`, or
- a fenced JSON block, or
- ``KEY: value`` lines (legacy v1 style) — these are normalized into the
  same dataclass.

Malformed inputs become :class:`MalformedPrediction` so the scorer can
penalize without crashing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Union

VALID_VERDICTS = {"APPROVE", "REQUEST_CHANGES"}


@dataclass
class ReviewerPrediction:
    verdict: str
    root_cause: str
    failure_path: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""
    raw: str = ""

    def is_malformed(self) -> bool:
        return False


@dataclass
class MalformedPrediction:
    reason: str
    raw: str = ""
    verdict: str = "REQUEST_CHANGES"
    root_cause: str = ""
    failure_path: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reasoning: str = ""

    def is_malformed(self) -> bool:
        return True


PredictionLike = Union[ReviewerPrediction, MalformedPrediction]


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_BARE_JSON_RE = re.compile(r"(\{[\s\S]*\})")
_KEY_LINE_RE = re.compile(r"^\s*([A-Za-z_]+)\s*:\s*(.*)$")


def _coerce_confidence(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(1.0, x))


def _coerce_failure_path(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        # Allow ``a -> b -> c`` or comma-separated fallback.
        parts = re.split(r"->|,", value)
        return [p.strip() for p in parts if p.strip()]
    return [str(value).strip()]


def _build_from_dict(data: dict, raw: str) -> PredictionLike:
    if not isinstance(data, dict):
        return MalformedPrediction(reason="not a JSON object", raw=raw)
    verdict = str(data.get("verdict", "")).strip().upper()
    if verdict not in VALID_VERDICTS:
        # Tolerate "REQUEST CHANGES" / "REJECT" by mapping.
        if verdict in {"REJECT", "REQUEST CHANGES", "BLOCK"}:
            verdict = "REQUEST_CHANGES"
        elif verdict in {"OK", "LGTM"}:
            verdict = "APPROVE"
        else:
            return MalformedPrediction(
                reason=f"invalid verdict {verdict!r}", raw=raw, verdict="REQUEST_CHANGES"
            )
    return ReviewerPrediction(
        verdict=verdict,
        root_cause=str(data.get("root_cause", "")).strip(),
        failure_path=_coerce_failure_path(data.get("failure_path")),
        confidence=_coerce_confidence(data.get("confidence", 0.0)),
        reasoning=str(data.get("reasoning", "")).strip(),
        raw=raw,
    )


def _parse_key_value(text: str, raw: str) -> PredictionLike:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        m = _KEY_LINE_RE.match(line)
        if not m:
            continue
        key = m.group(1).strip().lower()
        val = m.group(2).strip()
        fields[key] = val
    if "verdict" not in fields:
        return MalformedPrediction(reason="no VERDICT field", raw=raw)
    return _build_from_dict(
        {
            "verdict": fields.get("verdict", ""),
            "root_cause": fields.get("root_cause", ""),
            "failure_path": fields.get("failure_path", ""),
            "confidence": fields.get("confidence", "0"),
            "reasoning": fields.get("reasoning") or fields.get("reason", ""),
        },
        raw=raw,
    )


def parse_reviewer_output(text: Optional[str]) -> PredictionLike:
    if text is None:
        return MalformedPrediction(reason="empty output", raw="")
    raw = str(text)
    stripped = raw.strip()
    if not stripped:
        return MalformedPrediction(reason="empty output", raw=raw)

    fenced = _FENCED_JSON_RE.search(stripped)
    candidate: Optional[str] = fenced.group(1) if fenced else None
    if candidate is None and stripped.startswith("{"):
        candidate = stripped
    if candidate is None:
        m = _BARE_JSON_RE.search(stripped)
        if m:
            candidate = m.group(1)

    if candidate is not None:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return _build_from_dict(data, raw=raw)

    # Fallback to KEY: value parser (v1 legacy style).
    return _parse_key_value(stripped, raw=raw)
