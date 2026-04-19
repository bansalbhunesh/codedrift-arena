"""
RewardScorer — deterministic, interpretable reward function.

Mention detection uses the **ISSUES** section only. If that section cannot be
parsed, mention text is treated as empty so models cannot earn credit from
echoing the codebase into REASON alone.

Identifiers (function names) use word-boundary matching to reduce substring
false positives (e.g. ``getUserDatab`` should not match ``getUserData``).

**Diff grounding** (``info["diff_grounding"]``) is diagnostic only: it checks
whether each ground-truth stale artifact appears as text in the PR diff. It
does **not** change the reward—judges use it to spot ISSUES token-stuffing
without diff support.
"""

from __future__ import annotations

import re

from agents.drift_agent import DriftAction


def verdict_emoji(info: dict, reward: float) -> str:
    """Compact visual hint for demos (reward is unused but kept for API symmetry)."""
    _ = reward
    out = info.get("episode_outcome") or ""
    if out in ("correct_approve", "perfect"):
        return "🟢"
    if out in ("false_rejection", "missed_all"):
        return "🔴"
    if out == "partial":
        return "🟡"
    return "⚪"


def judge_summary_line(info: dict, reward: float) -> str:
    """One-line human translation for judges (does not affect reward)."""
    _ = reward
    n = int(info.get("n_stale_refs", 0) or 0)
    out = info.get("episode_outcome") or ""
    if n == 0:
        if out == "correct_approve":
            return "Model correctly approved: no injected drift in this episode."
        if out == "false_rejection":
            return "Model blocked a clean PR — no stale refs were injected."
        return "Clean-PR episode — see verdict and ISSUES format."
    if out == "perfect":
        return "Model caught every injected stale reference and requested changes."
    if out == "partial":
        return "Model caught some drift but missed at least one stale item."
    if out == "missed_all":
        return "Model missed drift (unsafe APPROVE or ISSUES did not evidence all stale items)."
    return "See metric_strip and JSON for details."


def judge_why_matters_line(info: dict, actions: list[DriftAction]) -> str:
    """Connect technical outcome to production impact (diagnostic; does not affect reward)."""
    n = len(actions)
    out = info.get("episode_outcome") or ""
    types = sorted({a.drift_type for a in actions}) if actions else []
    type_hint = ", ".join(types) if types else "none"
    if n == 0:
        if out == "correct_approve":
            return "Prevents false alarms: merge when the PR already matches the live codebase."
        if out == "false_rejection":
            return "Would waste engineering time blocking a change that already matches production."
        return "Clean-PR episode — impact is merge velocity vs needless churn."
    if out == "perfect":
        return (
            "Would prevent a production bug caused by outdated code in this diff "
            f"({type_hint})."
        )
    if out == "partial":
        return "Would still let part of the schema drift ship — latent outages, rollbacks, or hotfixes."
    if out == "missed_all":
        return "Would ship broken code: runtime errors, bad behavior, or failed imports/deploys."
    return "See drift keys and outcome for operational risk."


def judge_confidence_line(info: dict) -> str:
    """Heuristic confidence from diff grounding + recall (diagnostic only)."""
    n = int(info.get("n_stale_refs", 0) or 0)
    out = info.get("episode_outcome") or ""
    recall = float(info.get("recall", 0.0))
    frac = float(info.get("diff_grounding_fraction", 0.0))
    if n == 0:
        return "confidence: HIGH — no injected drift; rubric is approve vs. format only."
    if out == "perfect" and frac >= 1.0 - 1e-9 and recall >= 1.0 - 1e-9:
        return "confidence: HIGH — all stale references grounded in PR diff and fully caught in ISSUES."
    if frac >= 1.0 - 1e-9:
        return "confidence: MEDIUM — every stale ref appears in the diff; check ISSUES recall / verdict."
    if recall >= 1.0 - 1e-9:
        return "confidence: MEDIUM — full ISSUES recall; not every ref is verbatim in diff (see diff_grounding)."
    if recall > 0.0:
        return "confidence: MEDIUM — partial catch; ship decision still risky."
    return "confidence: LOW — missed drift or malformed review; do not treat as production-ready."


def _stale_token_in_pr_diff(action: DriftAction, pr_diff: str) -> bool:
    """Heuristic: stale artifact text appears in the diff (diagnostic, not used for reward)."""
    d = (pr_diff or "").lower()
    if not d.strip():
        return False
    if action.drift_type == "removal":
        mod = (action.metadata.get("module") or "").lower()
        st = action.stale_ref.lower()
        return bool(
            st in d
            or (mod and mod in d)
            or (mod and mod.replace(".", "/") in d)
        )
    if action.drift_type == "contract":
        fn = (action.metadata.get("function") or "").lower()
        bare = action.stale_ref.split("(")[0].lower()
        return fn in d or bare in d or action.stale_ref.lower() in d
    bare = action.stale_ref.split("(")[0].lower()
    return bare in d or action.stale_ref.lower() in d


def _normalize_issues_text(raw: str) -> str:
    """Light cleanup so markdown noise does not break matching."""
    s = raw.replace("`", " ").replace("**", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def _param_token_in_issues(issues_norm: str, param: str) -> bool:
    """
    True if ISSUES text plausibly references ``param`` as a parameter name.

    Longer identifiers use word boundaries to avoid substring games; very
    short names (e.g. ``to``) stay substring-based to avoid brittle prose match.
    """
    pl = param.lower()
    if not pl:
        return False
    if len(pl) >= 3 and re.fullmatch(r"[a-z_][a-z0-9_]*", pl):
        return re.search(rf"\b{re.escape(pl)}\b", issues_norm) is not None
    return pl in issues_norm


def _identifier_mentioned(issues_norm: str, ident: str) -> bool:
    """
    True if ``ident`` appears in ISSUES text. Paths / file refs use substring
    match; simple identifiers use word-boundary matching to avoid substring
    false positives (e.g. ``getuserdatab`` vs ``getuserdata``).
    ``issues_norm`` is already lowercased.
    """
    if not ident:
        return False
    ident_l = ident.lower()
    if "/" in ident_l or ident_l.endswith(".py"):
        return ident_l in issues_norm
    # Dotted imports / modules: substring is more robust than ``\\b`` on dots.
    if "." in ident_l:
        return ident_l in issues_norm
    if re.fullmatch(r"[a-z_][a-z0-9_]*", ident_l):
        return re.search(rf"\b{re.escape(ident_l)}\b", issues_norm) is not None
    return ident_l in issues_norm


class RewardScorer:
    R_CAUGHT_STALE = 1.0
    R_CORRECT_APPROVE = 0.5
    R_MISSED_STALE = -1.0
    R_FALSE_REJECTION = -0.3
    R_PARTIAL_CREDIT = 0.4

    @staticmethod
    def _parse_issues_section(response: str) -> str | None:
        """Return ISSUES body, or None if no ISSUES: block found."""
        m = re.search(
            r"ISSUES:\s*(.+?)(?:^\s*REASON\s*:|\Z)",
            response,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if not m:
            return None
        return _normalize_issues_text(m.group(1))

    def score(
        self,
        agent_response: str,
        actions: list[DriftAction],
        pr_diff: str,
    ) -> tuple[float, dict]:
        """
        Returns (total_reward, info_dict).
        info_dict contains per-signal breakdown for logging.
        """
        if agent_response is None:
            agent_response = ""
        verdict = self._extract_verdict(agent_response)

        total = 0.0
        info: dict = {
            "verdict": verdict,
            "n_stale_refs": len(actions),
            "caught": [],
            "missed": [],
            "partial": [],
            "breakdown": {},
            "episode_outcome": None,
        }

        if not actions:
            parsed = self._parse_issues_section(agent_response)
            info["malformed_issues"] = parsed is None
            info["diff_grounding"] = []
            info["diff_grounded_count"] = 0
            if verdict == "APPROVE":
                total += self.R_CORRECT_APPROVE
                info["breakdown"]["correct_approve"] = self.R_CORRECT_APPROVE
                info["episode_outcome"] = "correct_approve"
            else:
                total += self.R_FALSE_REJECTION
                info["breakdown"]["false_rejection"] = self.R_FALSE_REJECTION
                info["episode_outcome"] = "false_rejection"
            mal = "yes" if info["malformed_issues"] else "no"
            info["metric_strip"] = (
                f"reward={total:+.2f} | verdict={verdict} | n_stale=0 | malformed_issues={mal}"
            )
            info["judge_emoji"] = verdict_emoji(info, total)
            info["judge_summary"] = judge_summary_line(info, total)
            info["judge_why_matters"] = judge_why_matters_line(info, actions)
            info["confidence_strip"] = judge_confidence_line(info)
            return total, info

        info["expected_stale_keys"] = [self._stale_key(a) for a in actions]

        parsed = self._parse_issues_section(agent_response)
        issues_norm = parsed if parsed is not None else ""
        info["malformed_issues"] = parsed is None
        if info["malformed_issues"] and verdict == "APPROVE":
            info["warning"] = "malformed_issues_with_approve_on_drifted_pr"

        for action in actions:
            key = self._stale_key(action)
            mentioned = self._mentioned(action, issues_norm)

            if mentioned and verdict == "REQUEST_CHANGES":
                total += self.R_CAUGHT_STALE
                info["caught"].append(key)
                info["breakdown"][f"caught_{key}"] = self.R_CAUGHT_STALE

            elif mentioned and verdict == "APPROVE":
                total += self.R_PARTIAL_CREDIT
                info["partial"].append(key)
                info["breakdown"][f"partial_{key}"] = self.R_PARTIAL_CREDIT

            else:
                total += self.R_MISSED_STALE
                info["missed"].append(key)
                info["breakdown"][f"missed_{key}"] = self.R_MISSED_STALE

        n = len(actions)
        nc = len(info["caught"])
        info["recall"] = nc / max(1, n)
        info["precision_hint"] = nc / max(1, nc + len(info["partial"]))
        info["failure_rate"] = 1.0 - (nc / max(1, n))
        info["ambiguous_approve"] = bool(info["partial"])

        if nc == n:
            info["episode_outcome"] = "perfect"
        elif info["caught"]:
            info["episode_outcome"] = "partial"
        else:
            info["episode_outcome"] = "missed_all"

        grounds = [
            {
                "key": self._stale_key(a),
                "stale_token_in_pr_diff": _stale_token_in_pr_diff(a, pr_diff),
            }
            for a in actions
        ]
        info["diff_grounding"] = grounds
        ng = sum(1 for g in grounds if g["stale_token_in_pr_diff"])
        info["diff_grounded_count"] = ng
        info["diff_grounding_fraction"] = ng / max(1, n)

        mal = "yes" if info["malformed_issues"] else "no"
        info["metric_strip"] = (
            f"reward={total:+.2f} | recall={info['recall']:.0%} | verdict={verdict} | "
            f"malformed_issues={mal} | grounded_in_diff={ng}/{n}"
        )
        info["judge_emoji"] = verdict_emoji(info, total)
        info["judge_summary"] = judge_summary_line(info, total)
        info["judge_why_matters"] = judge_why_matters_line(info, actions)
        info["confidence_strip"] = judge_confidence_line(info)

        return total, info

    def _mentioned(self, action: DriftAction, issues_norm: str) -> bool:
        """
        Whether ISSUES text evidences the *stale* artifact.
        ``issues_norm`` is empty when ISSUES: could not be parsed — no mentions.
        """
        stale_bare = action.stale_ref.split("(")[0].lower()

        if action.drift_type == "removal":
            module = (action.metadata.get("module") or "").lower()
            return stale_bare in issues_norm or (module and module in issues_norm)

        if action.drift_type == "rename":
            return _identifier_mentioned(issues_norm, stale_bare)

        if action.drift_type == "contract":
            old_params = action.metadata.get("old_params") or []
            param_sig = ", ".join(old_params).lower().replace(" ", "")
            compact_issues = issues_norm.replace(" ", "")
            if not _identifier_mentioned(issues_norm, stale_bare):
                return False
            if not old_params:
                return False
            params_all_named = all(_param_token_in_issues(issues_norm, p) for p in old_params)
            return (
                param_sig in compact_issues
                or ", ".join(old_params).lower() in issues_norm
                or params_all_named
            )

        return _identifier_mentioned(issues_norm, stale_bare)

    def _extract_verdict(self, response: str) -> str:
        """Only trust an explicit VERDICT line; default conservative for unknown."""
        match = re.search(
            r"VERDICT\s*:\s*(APPROVE|REQUEST_CHANGES)",
            response,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()
        return "REQUEST_CHANGES"

    def _stale_key(self, action: DriftAction) -> str:
        bare = action.stale_ref.split("(")[0]
        return f"{action.drift_type}:{bare}"
