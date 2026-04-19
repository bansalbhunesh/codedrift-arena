"""
RewardScorer — deterministic, interpretable reward function.

Mention detection uses the **ISSUES** section only. If that section cannot be
parsed, mention text is treated as empty so models cannot earn credit from
echoing the codebase into REASON alone.
"""

from __future__ import annotations

import re

from agents.drift_agent import DriftAction


class RewardScorer:
    R_CAUGHT_STALE = 1.0
    R_CORRECT_APPROVE = 0.5
    R_MISSED_STALE = -1.0
    R_FALSE_REJECTION = -0.3
    R_PARTIAL_CREDIT = 0.4

    @staticmethod
    def _parse_issues_section(response: str) -> str | None:
        """Return lowercased ISSUES body, or None if no ISSUES: block found."""
        m = re.search(
            r"ISSUES:\s*(.+?)(?:^\s*REASON\s*:|\Z)",
            response,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if not m:
            return None
        return m.group(1).strip().lower()

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
        _ = pr_diff
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
            if verdict == "APPROVE":
                total += self.R_CORRECT_APPROVE
                info["breakdown"]["correct_approve"] = self.R_CORRECT_APPROVE
                info["episode_outcome"] = "correct_approve"
            else:
                total += self.R_FALSE_REJECTION
                info["breakdown"]["false_rejection"] = self.R_FALSE_REJECTION
                info["episode_outcome"] = "false_rejection"
            return total, info

        parsed = self._parse_issues_section(agent_response)
        issues_lower = parsed if parsed is not None else ""
        info["malformed_issues"] = parsed is None

        for action in actions:
            key = self._stale_key(action)
            mentioned = self._mentioned(action, issues_lower)

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

        if nc == n:
            info["episode_outcome"] = "perfect"
        elif info["caught"]:
            info["episode_outcome"] = "partial"
        else:
            info["episode_outcome"] = "missed_all"

        return total, info

    def _mentioned(self, action: DriftAction, issues_lower: str) -> bool:
        """
        Whether ISSUES text evidences the *stale* artifact.
        ``issues_lower`` is empty when ISSUES: could not be parsed — no mentions.
        """
        stale_bare = action.stale_ref.split("(")[0].lower()

        if action.drift_type == "removal":
            module = action.metadata.get("module", "").lower()
            return stale_bare in issues_lower or module in issues_lower

        if action.drift_type == "rename":
            return stale_bare in issues_lower

        if action.drift_type == "contract":
            old_params = action.metadata.get("old_params") or []
            param_sig = ", ".join(old_params).lower().replace(" ", "")
            compact_issues = issues_lower.replace(" ", "")
            if stale_bare not in issues_lower:
                return False
            if not old_params:
                return False
            return param_sig in compact_issues or ", ".join(old_params).lower() in issues_lower

        return stale_bare in issues_lower

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
