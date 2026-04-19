"""
RewardScorer — deterministic, interpretable reward function.

Design principles:
  1. Fully deterministic — same input always gives same reward.
  2. Partial credit — catches some but not all stale refs still rewarded.
  3. Penalty asymmetry — missing a stale ref (-1.0) is worse than
     a false positive (-0.3).
  4. Structured info dict — training script can log per-signal metrics.
"""

import re

from agents.drift_agent import DriftAction


class RewardScorer:
    R_CAUGHT_STALE = 1.0
    R_CORRECT_APPROVE = 0.5
    R_MISSED_STALE = -1.0
    R_FALSE_REJECTION = -0.3
    R_PARTIAL_CREDIT = 0.4

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
        response_lower = agent_response.lower()

        total = 0.0
        info = {
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
            mentioned = self._mentioned(action, response_lower)

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

        if len(info["caught"]) == len(actions):
            info["episode_outcome"] = "perfect"
        elif info["caught"]:
            info["episode_outcome"] = "partial"
        else:
            info["episode_outcome"] = "missed_all"

        return total, info

    def _mentioned(self, action: DriftAction, response_lower: str) -> bool:
        """Check if agent mentioned the stale reference in any form."""
        stale_bare = action.stale_ref.split("(")[0].lower()
        current_bare = action.current_ref.split("(")[0].lower()

        if action.drift_type == "removal":
            module = action.metadata.get("module", "").lower()
            return (
                stale_bare in response_lower
                or module.replace(".", "/") in response_lower
                or module.replace("/", ".") in response_lower
            )

        return stale_bare in response_lower or current_bare in response_lower

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
