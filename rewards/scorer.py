"""
RewardScorer — deterministic, interpretable reward function.

Mention detection uses the **ISSUES** section only. If that section cannot be
parsed, mention text is treated as empty so models cannot earn credit from
echoing the codebase into REASON alone.

Identifiers (function names) use word-boundary matching to reduce substring
false positives (e.g. ``getUserDatab`` should not match ``getUserData``).

**Diff grounding** (``info["diff_grounding"]``) checks whether each ground-truth
stale artifact appears as text in the PR diff. Grounding now softly affects
reward for caught drift: ungrounded catches receive reduced credit, while full
grounding keeps full catch reward.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from agents.drift_agent import API_CONTRACT_CHANGES, FUNCTION_RENAMES, REMOVABLE_FILES, DriftAction

if TYPE_CHECKING:
    from env.failure_cascade import FailureCascade


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
        return "confidence: HIGH (no injected drift)"
    if out == "perfect" and frac >= 1.0 - 1e-9 and recall >= 1.0 - 1e-9:
        return "confidence: HIGH (fully grounded + full recall)"
    if frac >= 1.0 - 1e-9:
        return "confidence: MEDIUM (all stale tokens in diff; check ISSUES)"
    if recall >= 1.0 - 1e-9:
        return "confidence: MEDIUM (full recall; diff match partial)"
    if recall > 0.0:
        return "confidence: MEDIUM (partial recall)"
    return "confidence: LOW (missed drift or bad format)"


def judge_keyword_line(info: dict) -> str:
    """Visceral SUCCESS / FAILURE headline for demos (diagnostic only)."""
    n = int(info.get("n_stale_refs", 0) or 0)
    out = info.get("episode_outcome") or ""
    if n == 0:
        if out == "correct_approve":
            return "🟢 SUCCESS: cleared clean PR"
        if out == "false_rejection":
            return "🔴 FAILURE: churn on clean PR"
        return "⚪ REVIEW: clean-PR episode"
    if out == "perfect":
        if n > 1:
            return "🟢 SUCCESS: multiple drifts blocked"
        return "🟢 SUCCESS: blocked outdated code"
    if out == "partial":
        return "🟡 PARTIAL: some drift still ships"
    if out == "missed_all":
        return "🔴 FAILURE: missed schema drift"
    return "⚪ OUTCOME: see JSON"


def _stale_token_in_pr_diff(action: DriftAction, pr_diff: str) -> bool:
    """Heuristic: stale artifact text appears in the diff."""
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


def _catalog_stale_identifiers() -> set[str]:
    """
    Canonical stale identifiers from the drift catalog.

    Used to penalize ISSUES keyword-dumping of stale symbols unrelated to the
    active episode.
    """
    out: set[str] = set()
    for old_name, _new_name in FUNCTION_RENAMES:
        out.add(old_name.lower())
    for item in API_CONTRACT_CHANGES:
        fn = str(item.get("function", "")).strip().lower()
        if fn:
            out.add(fn)
    for path in REMOVABLE_FILES:
        p = path.lower()
        out.add(p)
        out.add(p.replace("/", ".").replace(".py", ""))
    return out


CATALOG_STALE_IDENTIFIERS = _catalog_stale_identifiers()


def verbosity_penalty(issues_norm: str, pr_diff: str) -> tuple[float, dict]:
    """
    Penalize ISSUES that are huge relative to the PR diff (catalog / keyword spam).

    Uses normalized ISSUES text (same channel as mention credit). Only meaningful
    when a non-empty diff exists so we compare apples to apples.
    """
    if not (issues_norm or "").strip():
        return 0.0, {}
    d = (pr_diff or "").strip()
    if not d:
        return 0.0, {}
    issues_tokens = len(issues_norm.split())
    diff_tokens = max(len(d.split()), 1)
    ratio = issues_tokens / diff_tokens
    meta = {"issues_word_count": issues_tokens, "diff_word_count": diff_tokens, "verbosity_ratio": round(ratio, 3)}
    # Threshold: ISSUES word count / diff word count must exceed 4.0 before penalizing.
    # This avoids penalizing concise multi-drift ISSUES on small diffs.
    _VERBOSITY_RATIO_START = 4.0
    if ratio <= _VERBOSITY_RATIO_START:
        return 0.0, meta
    pen = -min(0.3, (ratio - _VERBOSITY_RATIO_START) * 0.05)
    meta["verbosity_penalty"] = pen
    return pen, meta


# Patterns that are harder to spot than a plain rename — earn extra credit when caught.
HARD_PATTERNS: frozenset[str] = frozenset({"condition_flip", "off_by_one"})

# Expected runtime exception / error-type keywords for each bug pattern.
# A reviewer that names the error type demonstrates genuine causal understanding.
_EXPECTED_ERRORS: dict[str, list[str]] = {
    "rename":         ["attributeerror", "nameerror"],
    "partial_rename": ["attributeerror", "nameerror"],
    "removal":        ["modulenotfounderror", "importerror"],
    "contract":       ["typeerror", "missing required"],
    "null_missing":   ["attributeerror", "nonetype", "none"],
    "type_mismatch":  ["typeerror", "valueerror"],
    "condition_flip": ["valueerror", "assertionerror", "wrong", "incorrect", "inverted"],
    "off_by_one":     ["indexerror", "keyerror", "wrong page", "0-based", "1-based", "index"],
}


class RewardScorer:
    R_CAUGHT_STALE = 1.0
    R_CORRECT_APPROVE = 0.5
    R_MISSED_STALE = -1.0
    R_FALSE_REJECTION = -0.3
    R_PARTIAL_CREDIT = 0.4
    R_SPURIOUS_STALE_MENTION = -0.25
    R_UNGROUNDED_CATCH_SCALE = 0.7
    # Causal reasoning bonuses — additive on top of base catch reward
    R_ROOT_CAUSE_CORRECT = 0.5
    R_FAILURE_PATH_CREDIT = 0.25
    R_CONFIDENCE_CALIBRATED = 0.1
    # Rich signal — qualitative depth bonuses
    R_ERROR_TYPE_NAMED = 0.15   # reviewer identifies the runtime exception type
    R_HARD_PATTERN_BONUS = 0.25  # extra for condition_flip / off_by_one (subtle bugs)
    R_COMPLETENESS_BONUS = 0.20  # all stale refs caught in a multi-ref (n>=2) episode
    R_FORMAT_COMPLETE = 0.05    # all 6 required fields present and non-empty

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

    @staticmethod
    def _parse_root_cause(response: str) -> str | None:
        m = re.search(
            r"ROOT_CAUSE\s*:\s*(.+?)(?:\n|FAILURE_PATH|CONFIDENCE|ISSUES|REASON|\Z)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return None
        return _normalize_issues_text(m.group(1))

    @staticmethod
    def _parse_failure_path(response: str) -> str | None:
        m = re.search(
            r"FAILURE_PATH\s*:\s*(.+?)(?:\n|CONFIDENCE|ISSUES|REASON|\Z)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return None
        return _normalize_issues_text(m.group(1))

    @staticmethod
    def _parse_confidence(response: str) -> float | None:
        m = re.search(r"CONFIDENCE\s*:\s*([0-9]*\.?[0-9]+)", response, re.IGNORECASE)
        if not m:
            return None
        try:
            return max(0.0, min(1.0, float(m.group(1))))
        except ValueError:
            return None

    def score(
        self,
        agent_response: str,
        actions: list[DriftAction],
        pr_diff: str,
        failure_cascade: "FailureCascade | None" = None,
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
            info["judge_keyword_line"] = judge_keyword_line(info)
            return total, info

        info["expected_stale_keys"] = [self._stale_key(a) for a in actions]

        parsed = self._parse_issues_section(agent_response)
        issues_norm = parsed if parsed is not None else ""
        info["malformed_issues"] = parsed is None
        if info["malformed_issues"] and verdict == "APPROVE":
            info["warning"] = "malformed_issues_with_approve_on_drifted_pr"

        diff_available = bool((pr_diff or "").strip())
        grounds = []
        for action in actions:
            key = self._stale_key(action)
            mentioned = self._mentioned(action, issues_norm)
            grounded = _stale_token_in_pr_diff(action, pr_diff)
            grounds.append({"key": key, "stale_token_in_pr_diff": grounded})

            if mentioned and verdict == "REQUEST_CHANGES":
                reward_delta = self.R_CAUGHT_STALE
                if diff_available and not grounded:
                    reward_delta *= self.R_UNGROUNDED_CATCH_SCALE
                total += reward_delta
                info["caught"].append(key)
                if grounded:
                    info["breakdown"][f"caught_{key}"] = reward_delta
                else:
                    info["breakdown"][f"caught_ungrounded_{key}"] = reward_delta

            elif mentioned and verdict == "APPROVE":
                total += self.R_PARTIAL_CREDIT
                info["partial"].append(key)
                info["breakdown"][f"partial_{key}"] = self.R_PARTIAL_CREDIT

            else:
                total += self.R_MISSED_STALE
                info["missed"].append(key)
                info["breakdown"][f"missed_{key}"] = self.R_MISSED_STALE

        expected_mentions = self._expected_episode_mentions(actions)
        mentioned_catalog = {
            ident
            for ident in CATALOG_STALE_IDENTIFIERS
            if _identifier_mentioned(issues_norm, ident)
        }
        spurious = sorted(m for m in mentioned_catalog if m not in expected_mentions)
        if spurious:
            spam_pen = self.R_SPURIOUS_STALE_MENTION * len(spurious)
            total += spam_pen
            info["spurious_mentions"] = spurious
            info["breakdown"]["spurious_stale_mentions"] = spam_pen
        else:
            info["spurious_mentions"] = []

        vpen, vmeta = verbosity_penalty(issues_norm, pr_diff)
        if vpen:
            total += vpen
            info["verbosity"] = vmeta
            info["breakdown"]["verbosity_penalty"] = vpen
        elif vmeta:
            info["verbosity"] = vmeta

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

        info["diff_grounding"] = grounds
        ng = sum(1 for g in grounds if g["stale_token_in_pr_diff"])
        info["diff_grounded_count"] = ng
        info["diff_grounding_fraction"] = ng / max(1, n)

        # ── Causal reasoning bonus (additive, does not affect base recall) ─────
        causal_bonus = 0.0
        causal_info: dict = {}

        root_cause_text = self._parse_root_cause(agent_response)
        failure_path_text = self._parse_failure_path(agent_response)
        confidence_val = self._parse_confidence(agent_response)

        causal_info["has_root_cause"] = root_cause_text is not None
        causal_info["has_failure_path"] = failure_path_text is not None
        causal_info["confidence_given"] = confidence_val

        stale_ids = {a.stale_ref.split("(")[0].lower() for a in actions}
        # Also include module names for removal actions
        for a in actions:
            if a.drift_type == "removal":
                mod = (a.metadata.get("module") or "").lower()
                if mod:
                    stale_ids.add(mod)

        if root_cause_text:
            root_cause_correct = any(
                _identifier_mentioned(root_cause_text, sid) for sid in stale_ids
            )
            causal_info["root_cause_correct"] = root_cause_correct
            if root_cause_correct:
                causal_bonus += self.R_ROOT_CAUSE_CORRECT
        else:
            causal_info["root_cause_correct"] = False

        if failure_path_text:
            path_has_stale = any(_identifier_mentioned(failure_path_text, sid) for sid in stale_ids)
            path_has_arrow = "→" in failure_path_text or "->" in failure_path_text
            # Also accept path tokens from failure cascade if available
            if failure_cascade is not None:
                cascade_tokens = [t.lower() for t in failure_cascade.all_path_tokens()]
                path_has_chain = sum(
                    1 for tok in cascade_tokens if tok in failure_path_text
                ) >= 2
            else:
                path_has_chain = path_has_stale and path_has_arrow
            causal_info["failure_path_credit"] = path_has_chain
            if path_has_chain:
                causal_bonus += self.R_FAILURE_PATH_CREDIT
        else:
            causal_info["failure_path_credit"] = False

        if confidence_val is not None:
            reviewer_correct = info.get("episode_outcome") == "perfect"
            calibrated = (reviewer_correct and confidence_val >= 0.5) or (
                not reviewer_correct and confidence_val <= 0.6
            )
            causal_info["confidence_calibrated"] = calibrated
            if calibrated:
                causal_bonus += self.R_CONFIDENCE_CALIBRATED
        else:
            causal_info["confidence_calibrated"] = False

        if causal_bonus > 0:
            total += causal_bonus
            info["breakdown"]["causal_bonus"] = causal_bonus
        info["causal_reasoning"] = causal_info
        # ── End causal bonus ──────────────────────────────────────────────────

        # ── Rich signal bonuses (qualitative depth) ──────────────────────────
        rich_bonus, rich_info = self._rich_signal_bonus(
            agent_response=agent_response,
            actions=actions,
            caught_keys=info["caught"],
            episode_outcome=info["episode_outcome"] or "",
        )
        if rich_bonus > 0:
            total += rich_bonus
            info["breakdown"]["rich_bonus"] = rich_bonus
        info["rich_signal"] = rich_info
        # ─────────────────────────────────────────────────────────────────────

        info["reward"] = round(total, 4)

        mal = "yes" if info["malformed_issues"] else "no"
        verb = f" | verbosity_pen={info['breakdown'].get('verbosity_penalty', 0):+.2f}" if vpen else ""
        causal_str = f" | causal={causal_bonus:+.2f}" if causal_bonus else ""
        rich_str = f" | rich={rich_bonus:+.2f}" if rich_bonus else ""
        info["metric_strip"] = (
            f"reward={total:+.2f} | recall={info['recall']:.0%} | verdict={verdict} | "
            f"malformed_issues={mal} | grounded_in_diff={ng}/{n} | spurious={len(info['spurious_mentions'])}"
            f"{verb}{causal_str}{rich_str}"
        )
        info["judge_emoji"] = verdict_emoji(info, total)
        info["judge_summary"] = judge_summary_line(info, total)
        info["judge_why_matters"] = judge_why_matters_line(info, actions)
        info["confidence_strip"] = judge_confidence_line(info)
        info["judge_keyword_line"] = judge_keyword_line(info)

        return total, info

    # ── Rich signal helpers ───────────────────────────────────────────────────

    def _rich_signal_bonus(
        self,
        agent_response: str,
        actions: list[DriftAction],
        caught_keys: list[str],
        episode_outcome: str,
    ) -> tuple[float, dict]:
        """
        Four additive bonuses that reward qualitative depth of the review:

        1. Error-type identification  — names the expected runtime exception
        2. Hard-pattern bonus         — condition_flip / off_by_one are subtle
        3. Completeness bonus         — all refs caught in a multi-ref episode
        4. Format completeness        — all 6 required fields present

        None of these bonuses modify the base catch/miss signal. They exist to
        widen the reward gap between shallow ("stale ref exists") and deep
        ("this will raise AttributeError because the function was renamed")
        responses, giving GRPO a richer gradient to train against.
        """
        bonus = 0.0
        info: dict = {}

        full_norm = _normalize_issues_text(agent_response.replace("\n", " "))

        # 1. Error type identification — at least one expected error type appears
        #    in the full response for every *caught* action.
        type_hits = 0
        for action in actions:
            key = self._stale_key(action)
            if key not in caught_keys:
                continue
            pat = action.bug_pattern or action.drift_type
            expected = _EXPECTED_ERRORS.get(pat, [])
            if expected and any(e in full_norm for e in expected):
                type_hits += 1
        n_caught = len(caught_keys)
        if n_caught > 0 and type_hits == n_caught:
            bonus += self.R_ERROR_TYPE_NAMED
            info["error_type_named"] = True
        else:
            info["error_type_named"] = False
            info["error_type_hits"] = f"{type_hits}/{n_caught}"

        # 2. Hard pattern bonus — extra for catching condition_flip / off_by_one
        hard_caught = sum(
            1 for a in actions
            if (a.bug_pattern or a.drift_type) in HARD_PATTERNS
            and self._stale_key(a) in caught_keys
        )
        if hard_caught > 0:
            hard_b = self.R_HARD_PATTERN_BONUS * hard_caught
            bonus += hard_b
            info["hard_pattern_bonus"] = round(hard_b, 3)
        else:
            info["hard_pattern_bonus"] = 0.0

        # 3. Completeness — all refs caught in a multi-ref episode
        if len(actions) >= 2 and n_caught == len(actions):
            bonus += self.R_COMPLETENESS_BONUS
            info["completeness_bonus"] = True
        else:
            info["completeness_bonus"] = False

        # 4. Format completeness — all 6 required fields present and non-empty
        _REQUIRED_FIELDS = ["VERDICT", "ROOT_CAUSE", "FAILURE_PATH", "CONFIDENCE", "ISSUES", "REASON"]
        fields_found = [
            f for f in _REQUIRED_FIELDS
            if re.search(rf"^{f}\s*:\s*\S", agent_response, re.IGNORECASE | re.MULTILINE)
        ]
        if len(fields_found) == len(_REQUIRED_FIELDS):
            bonus += self.R_FORMAT_COMPLETE
            info["format_complete"] = True
        else:
            info["format_complete"] = False
            info["fields_found"] = len(fields_found)

        return round(bonus, 4), info

    def _mentioned(self, action: DriftAction, issues_norm: str) -> bool:
        """
        Whether ISSUES text evidences the *stale* artifact.
        ``issues_norm`` is empty when ISSUES: could not be parsed — no mentions.
        """
        pat = action.bug_pattern or action.drift_type
        stale_bare = action.stale_ref.split("(")[0].lower()

        # ── New realistic patterns ────────────────────────────────────────────
        if pat == "partial_rename":
            # Must mention the stale function name (same as rename)
            return _identifier_mentioned(issues_norm, stale_bare)

        if pat == "null_missing":
            fn = (action.metadata.get("function") or stale_bare).lower()
            attr = (action.metadata.get("nullable_attribute") or "").lower()
            fn_mentioned = _identifier_mentioned(issues_norm, fn)
            null_mentioned = any(tok in issues_norm for tok in ("none", "null", "nonetype", "optional"))
            return fn_mentioned and (null_mentioned or (attr and attr in issues_norm))

        if pat == "type_mismatch":
            fn = (action.metadata.get("function") or stale_bare).lower()
            param = (action.metadata.get("param") or "").lower()
            old_type = (action.metadata.get("old_type") or "").lower()
            new_type = (action.metadata.get("new_type") or "").lower()
            fn_mentioned = _identifier_mentioned(issues_norm, fn)
            type_context = (
                (param and _param_token_in_issues(issues_norm, param))
                or (old_type and old_type in issues_norm)
                or (new_type and new_type in issues_norm)
                or "type" in issues_norm
            )
            return fn_mentioned and type_context

        if pat == "condition_flip":
            fn = (action.metadata.get("function") or stale_bare).lower()
            param = (action.metadata.get("param") or "").lower()
            fn_mentioned = _identifier_mentioned(issues_norm, fn)
            flag_context = (
                (param and _param_token_in_issues(issues_norm, param))
                or any(tok in issues_norm for tok in ("invert", "flip", "semantic", "wrong value", "boolean", "flag"))
            )
            return fn_mentioned and flag_context

        if pat == "off_by_one":
            fn = (action.metadata.get("function") or stale_bare).lower()
            param = (action.metadata.get("param") or "").lower()
            fn_mentioned = _identifier_mentioned(issues_norm, fn)
            offset_context = (
                (param and _param_token_in_issues(issues_norm, param))
                or any(tok in issues_norm for tok in ("0-based", "1-based", "index", "offset", "off by one", "page"))
            )
            return fn_mentioned and offset_context

        # ── Legacy patterns ───────────────────────────────────────────────────
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
        label = action.bug_pattern or action.drift_type
        return f"{label}:{bare}"

    def _expected_episode_mentions(self, actions: list[DriftAction]) -> set[str]:
        """
        Stale identifiers that are valid to mention for this episode.

        Includes file-module form for removal drifts.
        """
        out: set[str] = set()
        for action in actions:
            bare = action.stale_ref.split("(")[0].lower()
            if bare:
                out.add(bare)
            cur = action.current_ref.split("(")[0].lower()
            if cur and cur != "[deleted]":
                out.add(cur)
            if action.drift_type == "removal":
                mod = str(action.metadata.get("module", "")).strip().lower()
                if mod:
                    out.add(mod)
        return out
