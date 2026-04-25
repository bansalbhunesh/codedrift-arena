"""Multi-component causal reward for V2.

Components, each in [0, 1] before weighting:
- ``root_cause_score``: graded match between predicted root cause and any of
  the ground-truth mutation symbols (full credit for full ``file::symbol``,
  partial for symbol-only or file-only).
- ``failure_path_score``: ordered Jaccard-like overlap between predicted
  ``failure_path`` tokens and the canonical ground-truth chain.
- ``verdict_score``: 1 if predicted verdict matches the ground-truth one
  (REQUEST_CHANGES iff any test failed/errored), else 0.
- ``calibration_score``: 1 - Brier(confidence, root_cause_correct).
- ``hallucination_penalty``: 1 if any predicted symbol does not appear in
  the actual repo or diff, else 0 (subtracted from total).

Weighted total:
  R = 1.0 * root_cause + 0.6 * failure_path + 0.2 * verdict
      - 0.4 * hallucination - 0.2 * calibration_error
Bounded to [-1.0, +2.0].

Malformed predictions (missing schema fields) get a fixed penalty of -0.5
plus a small hallucination penalty so models cannot "win" by emitting noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents_v2.reviewer_io import MalformedPrediction, PredictionLike, ReviewerPrediction
from env_v2.exec_engine import ExecutionResult

W_ROOT = 1.0
W_PATH = 0.6
W_VERDICT = 0.2
W_HALLUC = 0.4
W_CAL = 0.2
MIN_REWARD = -1.0
MAX_REWARD = 2.0
MALFORMED_PENALTY = -0.5


@dataclass
class ScoreBreakdown:
    root_cause: float = 0.0
    failure_path: float = 0.0
    verdict: float = 0.0
    calibration: float = 0.0
    hallucination: float = 0.0
    malformed: bool = False
    matched_symbol: str = ""
    notes: list[str] = field(default_factory=list)


def _normalize_symbol(s: str) -> tuple[str, str]:
    """Return (file_part, symbol_part) lowercased, splitting on ``::``."""
    if not s:
        return "", ""
    s = s.strip().lower().replace("\\", "/")
    if "::" in s:
        f, sym = s.split("::", 1)
        return f.strip(), sym.strip()
    return "", s


def _root_cause_score(pred: str, ground_symbols: list[str]) -> tuple[float, str]:
    if not pred or not ground_symbols:
        return 0.0, ""
    pf, ps = _normalize_symbol(pred)
    best = 0.0
    matched = ""
    for gt in ground_symbols:
        gf, gs = _normalize_symbol(gt)
        score = 0.0
        if pf and ps and pf == gf and ps == gs:
            score = 1.0
        elif ps and ps == gs:
            score = 0.7
        elif pf and pf == gf:
            score = 0.3
        if score > best:
            best = score
            matched = gt
    return best, matched


def _failure_path_score(predicted: list[str], ground: list[str]) -> float:
    if not predicted or not ground:
        return 0.0
    p_set = {x.strip().lower() for x in predicted if x.strip()}
    g_set = {x.strip().lower() for x in ground if x.strip()}
    if not p_set or not g_set:
        return 0.0
    inter = len(p_set & g_set)
    union = len(p_set | g_set)
    jaccard = inter / union if union else 0.0
    # Order bonus: same first/last element gets a small boost.
    order_bonus = 0.0
    if predicted and ground and predicted[0].strip().lower() == ground[0].strip().lower():
        order_bonus += 0.1
    if predicted and ground and predicted[-1].strip().lower() == ground[-1].strip().lower():
        order_bonus += 0.2
    return min(1.0, jaccard + order_bonus)


def _verdict_score(predicted: str, expected: str) -> float:
    return 1.0 if predicted.upper() == expected.upper() else 0.0


def _calibration_score(confidence: float, root_correct: float) -> tuple[float, float]:
    """Return (calibration_score in [0,1], calibration_error in [0,1])."""
    correct_binary = 1.0 if root_correct >= 0.7 else 0.0
    err = (confidence - correct_binary) ** 2
    err = max(0.0, min(1.0, err))
    return 1.0 - err, err


def _hallucination_penalty(
    prediction: ReviewerPrediction,
    exec_result: ExecutionResult,
    ground_symbols: list[str],
    pr_diff: str,
) -> tuple[float, list[str]]:
    """Return (penalty in [0,1], offending tokens)."""
    if not prediction.root_cause:
        return 0.0, []
    pf, ps = _normalize_symbol(prediction.root_cause)
    repo_tokens = " ".join(t.nodeid.lower() for t in exec_result.failed_tests)
    repo_tokens += " " + (pr_diff or "").lower()
    repo_tokens += " " + " ".join(s.lower() for s in ground_symbols)
    offenders: list[str] = []
    if ps and ps not in repo_tokens:
        offenders.append(ps)
    if pf and pf not in repo_tokens:
        offenders.append(pf)
    return (1.0 if offenders else 0.0), offenders


class CausalScorer:
    """Public scorer used by :class:`env_v2.exec_arena_env.CodeReviewArenaEnv`."""

    def score(
        self,
        prediction: PredictionLike,
        ground_truth: dict[str, Any],
        exec_result: ExecutionResult,
        mutations: list[Any],
        pr_diff: str = "",
    ) -> tuple[float, dict[str, Any]]:
        breakdown = ScoreBreakdown()
        info: dict[str, Any] = {}

        if isinstance(prediction, MalformedPrediction):
            breakdown.malformed = True
            breakdown.notes.append(f"malformed: {prediction.reason}")
            info["breakdown"] = breakdown.__dict__
            info["reward_components"] = {
                "root_cause": 0.0,
                "failure_path": 0.0,
                "verdict": 0.0,
                "calibration": 0.0,
                "hallucination": 0.0,
                "malformed_penalty": MALFORMED_PENALTY,
            }
            info["pred_verdict"] = prediction.verdict
            info["pred_root_cause"] = prediction.root_cause
            info["gt_root_cause"] = ground_truth.get("root_cause", "")
            info["gt_verdict"] = ground_truth.get("verdict", "")
            info["episode_outcome"] = "malformed"
            return _clamp(MALFORMED_PENALTY), info

        ground_symbols = list(ground_truth.get("root_cause_symbols") or [])
        if ground_truth.get("root_cause"):
            ground_symbols = [ground_truth["root_cause"], *ground_symbols]
        ground_symbols = list(dict.fromkeys(ground_symbols))

        root_score, matched = _root_cause_score(prediction.root_cause, ground_symbols)
        breakdown.root_cause = root_score
        breakdown.matched_symbol = matched

        path_score = _failure_path_score(
            prediction.failure_path, list(ground_truth.get("failure_path") or [])
        )
        breakdown.failure_path = path_score

        verdict_match = _verdict_score(prediction.verdict, str(ground_truth.get("verdict", "")))
        breakdown.verdict = verdict_match

        cal_score, cal_err = _calibration_score(prediction.confidence, root_score)
        breakdown.calibration = cal_score

        halluc_pen, offenders = _hallucination_penalty(
            prediction, exec_result, ground_symbols, pr_diff
        )
        breakdown.hallucination = halluc_pen

        total = (
            W_ROOT * root_score
            + W_PATH * path_score
            + W_VERDICT * verdict_match
            - W_HALLUC * halluc_pen
            - W_CAL * cal_err
        )
        total = _clamp(total)

        info["breakdown"] = breakdown.__dict__
        info["reward_components"] = {
            "root_cause": round(W_ROOT * root_score, 4),
            "failure_path": round(W_PATH * path_score, 4),
            "verdict": round(W_VERDICT * verdict_match, 4),
            "hallucination": round(-W_HALLUC * halluc_pen, 4),
            "calibration": round(-W_CAL * cal_err, 4),
        }
        info["calibration_error"] = round(cal_err, 4)
        info["hallucinated_tokens"] = offenders
        info["pred_verdict"] = prediction.verdict
        info["pred_root_cause"] = prediction.root_cause
        info["gt_root_cause"] = ground_truth.get("root_cause", "")
        info["gt_verdict"] = ground_truth.get("verdict", "")
        info["matched_root_cause"] = matched
        outcome = "perfect"
        if root_score < 0.7:
            outcome = "wrong_root_cause"
        elif halluc_pen > 0:
            outcome = "hallucinated"
        elif path_score < 0.5:
            outcome = "weak_path"
        info["episode_outcome"] = outcome
        return total, info


def _clamp(x: float) -> float:
    return max(MIN_REWARD, min(MAX_REWARD, float(x)))
