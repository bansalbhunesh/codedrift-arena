"""
RewardScorer — deterministic, interpretable reward function.

Design principles:
  1. Fully deterministic — same input always gives same reward.
  2. Partial credit — catches some but not all stale refs still rewarded.
  3. Penalty asymmetry — missing a stale ref (-1.0) is worse than
     a false positive (-0.3).
  4. Structured info dict — training script can log per-signal metrics.

Mention detection uses the **ISSUES** section when present so models cannot
earn credit by echoing the codebase block into REASON alone.
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
    def _issues_text_lower(response: str) -> str:
        """Body of ISSUES: … up to REASON: (fallback: whole response)."""
        m = re.search(
            r"ISSUES:\s*(.+?)(?:^\s*REASON\s*:|\Z)",
            response,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        if m:
            return m.group(1).strip().lower()
        return response.lower()

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
        issues_lower = self._issues_text_lower(agent_response)

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
        Whether ISSUES (or full response if unparseable) evidences the *stale* artifact.
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
        match = re.search(
            r"VERDICT\s*:\s*(APPROVE|REQUEST_CHANGES)",
            response,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()
        lower = response.lower()
        if "request_changes" in lower or "request changes" in lower:
            return "REQUEST_CHANGES"
        if "approve" in lower:
            return "APPROVE"
        return "REQUEST_CHANGES"

    def _stale_key(self, action: DriftAction) -> str:
        bare = action.stale_ref.split("(")[0]
        return f"{action.drift_type}:{bare}"
