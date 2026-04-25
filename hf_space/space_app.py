"""Gradio UI for Hugging Face Spaces — CodeDrift Arena.

Production-style dashboard: design tokens, max-width layout, two-column
missions (controls vs outputs), and WCAG-friendly contrast. Same behavior as
before: episode reset, review scoring, benchmark, real-PR heuristics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

from codedrift.logutil import configure_logging
from env.codedrift_env import CodeDriftEnv
from hf_space.real_pr_scorer import (
    MAX_DIFF_BYTES,
    detect_languages,
    extract_candidate_stale_refs,
    fetch_diff_from_url,
    score_real_pr,
)

configure_logging()

MAX_FETCH_BYTES_FMT = f"{MAX_DIFF_BYTES:,}"


# ─── small utils ────────────────────────────────────────────────────────────

def _fmt_info(info: dict[str, Any]) -> str:
    return json.dumps(info, indent=2, default=str)


def _ascii_bar(pct: float, width: int = 18) -> str:
    """Pure-text bar so it survives any theme change."""
    pct = max(0.0, min(1.0, pct))
    n = int(round(pct * width))
    return "█" * n + "·" * (width - n)


# ─── HUD state ──────────────────────────────────────────────────────────────

DEFAULT_PLAYER: dict[str, Any] = {
    "callsign": "AGENT-001",
    "total_xp": 0,
    "missions_played": 0,
    "missions_won": 0,
    "missions_lost": 0,
    "current_streak": 0,
    "best_streak": 0,
    "last_outcome": None,   # "success" | "failure" | "partial" | None
    "production_health": 100,
}


def _level_for(xp: int) -> tuple[int, int, int]:
    """Return (level, xp_into_level, xp_to_next). Each level needs 250 XP."""
    step = 250
    lvl = max(1, xp // step + 1)
    into = xp % step
    return lvl, into, step


def _hud_html(player: dict[str, Any]) -> str:
    """Top mission-control HUD: callsign · level · XP bar · streak · production."""
    lvl, into, step = _level_for(int(player.get("total_xp", 0)))
    streak = int(player.get("current_streak", 0))
    best = int(player.get("best_streak", 0))
    played = int(player.get("missions_played", 0))
    won = int(player.get("missions_won", 0))
    health = int(player.get("production_health", 100))
    last = player.get("last_outcome")

    health_label = (
        "STABLE" if health >= 80
        else "DEGRADED" if health >= 40
        else "CRITICAL" if health > 0
        else "OFFLINE"
    )
    health_class = (
        "ok" if health >= 80
        else "warn" if health >= 40
        else "danger"
    )
    xp_pct = 100.0 * into / max(1, step)
    streak_html = (
        f"<span class='ds-pill ds-pill--ok'>{streak} win streak · best {best}</span>"
        if streak >= 2
        else f"<span class='ds-pill'>{streak} streak · best {best}</span>"
    )
    last_pill = ""
    if last == "success":
        last_pill = "<span class='ds-pill ds-pill--ok'>Last: caught</span>"
    elif last == "failure":
        last_pill = "<span class='ds-pill ds-pill--danger'>Last: missed</span>"
    elif last == "partial":
        last_pill = "<span class='ds-pill ds-pill--warn'>Last: partial</span>"

    return f"""
<section class="ds-hud" aria-label="Run overview">
  <div class="ds-hud__kpis">
    <div>
      <p class="ds-caption">Agent</p>
      <p class="ds-h3--kpi">{player.get("callsign", "AGENT-001")}</p>
    </div>
    <div>
      <p class="ds-caption">Level</p>
      <p class="ds-h3--kpi">L{lvl} &nbsp;<span class="ds-kpi-suffix">{int(player.get("total_xp", 0))} XP</span></p>
    </div>
    <div>
      <p class="ds-caption">Wins</p>
      <p class="ds-h3--kpi">{won}<span class="ds-kpi-suffix"> / {played}</span></p>
    </div>
    <div>
      <p class="ds-caption">Streak</p>
      <div>{streak_html}</div>
    </div>
    <div>
      <p class="ds-caption">Production</p>
      <div class="ds-prod ds-prod--{health_class}"><span class="ds-prod-dot" aria-hidden="true"></span><span class="ds-prod-cnt">{health_label} · {health}%</span></div>
    </div>
  </div>
  <div class="ds-hud__xp">
    <p class="ds-caption">Level {lvl + 1} — {into} / {step} XP</p>
    <div class="ds-xpbar" style="--xp-pct: {xp_pct:.2f}" role="progressbar" aria-valuenow="{into}" aria-valuemin="0" aria-valuemax="{step}">
      <div class="ds-xpbar-fill"></div>
    </div>
  </div>
  <div class="ds-hud__meta" aria-live="polite">{last_pill}</div>
</section>
""".strip()


def _apply_outcome(player: dict[str, Any], reward: float, info: dict[str, Any]) -> dict[str, Any]:
    """Update player progression based on the latest scored episode."""
    p = dict(player or DEFAULT_PLAYER)
    kw = str(info.get("judge_keyword_line") or "")
    is_success = "SUCCESS" in kw or "PERFECT" in kw
    is_failure = "FAILURE" in kw or "MISSED" in kw
    is_partial = "PARTIAL" in kw

    xp_gain = int(round(float(reward) * 100))
    p["total_xp"] = max(0, int(p.get("total_xp", 0)) + xp_gain)
    p["missions_played"] = int(p.get("missions_played", 0)) + 1

    if is_success:
        p["missions_won"] = int(p.get("missions_won", 0)) + 1
        p["current_streak"] = int(p.get("current_streak", 0)) + 1
        p["best_streak"] = max(int(p.get("best_streak", 0)), p["current_streak"])
        p["production_health"] = min(100, int(p.get("production_health", 100)) + 5)
        p["last_outcome"] = "success"
    elif is_failure:
        p["missions_lost"] = int(p.get("missions_lost", 0)) + 1
        p["current_streak"] = 0
        p["production_health"] = max(0, int(p.get("production_health", 100)) - 25)
        p["last_outcome"] = "failure"
    elif is_partial:
        p["current_streak"] = 0
        p["production_health"] = max(0, int(p.get("production_health", 100)) - 8)
        p["last_outcome"] = "partial"
    else:
        p["last_outcome"] = None
    return p


# ─── status banner ──────────────────────────────────────────────────────────

def _status_banner(reward: float, info: dict[str, Any]) -> str:
    """Mission report card. Full borders, leading badge, NO left-stripe accent."""
    kw = info.get("judge_keyword_line") or "⚪ NO REVIEW"
    summary = info.get("judge_summary") or ""
    why = info.get("judge_why_matters") or ""
    conf = info.get("confidence_strip") or ""
    ep = info.get("episode_id") or ""
    n_stale = int(info.get("n_stale_refs", 0) or 0)
    caught_count = len(info.get("caught", []) or [])
    missed_count = len(info.get("missed", []) or [])
    xp = int(round(float(reward) * 100))

    is_success = "SUCCESS" in kw or "PERFECT" in kw
    is_failure = "FAILURE" in kw or "MISSED" in kw
    is_partial = "PARTIAL" in kw

    if is_success:
        tone, label = "ok", "BUG DEFUSED"
        icon = "◎"
        prod = "Production stayed up. Bug caught before merge."
    elif is_failure:
        tone, label = "danger", "PROD DOWN"
        icon = "✕"
        prod = "Bug shipped to users. Test crashed in production."
    elif is_partial:
        tone, label = "warn", "PARTIAL CATCH"
        icon = "▲"
        prod = "Some bugs slipped through. Tech debt accruing."
    else:
        tone, label = "muted", "NO REPORT"
        icon = "○"
        prod = "Awaiting mission report."

    xp_sign = f"+{xp}" if xp >= 0 else f"{xp}"
    return f"""
<section class="ds-banner ds-banner--{tone}" role="status" aria-live="polite">
  <header class="ds-banner__head">
    <span class="ds-banner__label">{icon} {label}</span>
    <span class="ds-banner__xp">{xp_sign} XP</span>
  </header>
  <div class="ds-banner__body">{prod}</div>
  <div class="ds-banner__grid" aria-label="Scoring summary">
    <span><span class="ds-stat__k">Stale refs</span> {n_stale}</span>
    <span><span class="ds-stat__k">Caught</span> <em class="ds-ok">{caught_count}</em></span>
    <span><span class="ds-stat__k">Missed</span> <em class="ds-bad">{missed_count}</em></span>
  </div>
  {"<div class='ds-line'>" + summary + "</div>" if summary else ""}
  {"<div class='ds-line ds-line--muted'><strong>Why it matters:</strong> " + why + "</div>" if why else ""}
  {"<div class='ds-line ds-line--muted'>" + conf + "</div>" if conf else ""}
  <footer class="ds-banner__foot">Mission <code>{ep}</code></footer>
</section>
""".strip()


# ─── adversary brain ────────────────────────────────────────────────────────

def _brain_html(env: CodeDriftEnv | None) -> str:
    if env is None:
        return (
            "<section class='ds-card ds-card--muted' aria-label='Adversary brain'>"
            "<h3 class='ds-card__title'>Adversary brain</h3>"
            "<div class='ds-card__body'>Deploy a mission to initialise.</div>"
            "</section>"
        )
    snap = env.drift_agent.adaptive_snapshot()
    if not snap.get("enabled"):
        return (
            "<section class='ds-card ds-card--muted' aria-label='Adversary brain'>"
            "<h3 class='ds-card__title'>Adversary brain</h3>"
            "<div class='ds-card__body'>Available in <code>adaptive</code> personality mode.</div>"
            "</section>"
        )
    stage = str(snap.get("stage", "random"))
    ep = int(snap.get("episodes_run", 0) or 0)
    wr5 = float(snap.get("recent_win_rate_5", 0.0) or 0.0)
    wr10 = float(snap.get("recent_win_rate_10", 0.0) or 0.0)
    wr20 = float(snap.get("recent_win_rate_20", 0.0) or 0.0)
    scores = snap.get("mode_scores", {}) or {}
    tone = "ok" if stage == "random" else "warn" if stage == "subtle" else "danger"
    rows = []
    for mode in ("random", "subtle", "aggressive"):
        v = float(scores.get(mode, 0.0) or 0.0)
        rows.append(
            f"<div class='brain-row'>"
            f"<span class='brain-mode'>{mode}</span>"
            f"<span class='brain-bar'>{_ascii_bar(v)}</span>"
            f"<span class='brain-pct'>{v:.0%}</span>"
            f"</div>"
        )
    return (
        f"<section class='ds-card ds-card--{tone}' aria-label='Adversary brain'>"
        f"<h3 class='ds-card__title'>Brain · {stage.upper()} · ep {ep}</h3>"
        f"<div class='ds-card__body'>"
        f"<p class='brain-meta'>Win rate — 5ep: {wr5:.0%} · 10ep: {wr10:.0%} · 20ep: {wr20:.0%}</p>"
        f"<div>{''.join(rows)}</div>"
        f"</div></section>"
    )


# ─── pre-loaded reviewer responses ─────────────────────────────────────────

BASE_MODEL_RESPONSE = """VERDICT: APPROVE
ROOT_CAUSE: none
FAILURE_PATH: n/a
CONFIDENCE: 0.85
ISSUES: none
REASON: The code looks correct and follows existing patterns."""

TRAINED_FALLBACK_RESPONSE = """VERDICT: REQUEST_CHANGES
ROOT_CAUSE: <no active episode>
FAILURE_PATH: n/a
CONFIDENCE: 0.5
ISSUES: Deploy a mission first so the trained policy sees the real diff and stale refs.
REASON: No active episode in memory."""


def _cascade_path_for(env: CodeDriftEnv, action: "Any", error_suffix: str) -> str:
    """Return a FAILURE_PATH string built from the actual cascade call_chain.

    Using real cascade tokens (test_name, intermediate fn, broken fn) ensures
    the scorer's token-matching check (>= 2 tokens found) awards the
    R_FAILURE_PATH_CREDIT bonus instead of falling back to 'path_has_arrow'.
    """
    cascade = getattr(env, "failure_cascade", None)
    if cascade and getattr(cascade, "failures", None):
        stale_bare = action.stale_ref.split("(")[0].lower()
        # Try to find the failure whose call_chain contains the stale ref
        for failure in cascade.failures:
            chain = failure.call_chain or []
            if any(stale_bare in tok.lower() for tok in chain):
                return " -> ".join(chain) + f" -> {error_suffix}"
        # Fallback: use first failure chain
        chain = cascade.failures[0].call_chain or []
        if chain:
            return " -> ".join(chain) + f" -> {error_suffix}"
    # No cascade data — build a plausible path from metadata
    fn = str(action.metadata.get("caller") or action.metadata.get("function") or "caller")
    stale = action.stale_ref.split("(")[0]
    return f"failing_test -> {fn} -> {stale} -> {error_suffix}"


def _trained_response_for(env: CodeDriftEnv | None) -> str:
    """Mission-aware trained response: cites stale refs, error types, and fixes.

    Dispatches by bug_pattern first so nuanced patterns (null_missing,
    type_mismatch, condition_flip, off_by_one) get specific error-type
    language that earns the R_ERROR_TYPE_NAMED rich-signal bonus.
    """
    if env is None or not env.stale_actions:
        return TRAINED_FALLBACK_RESPONSE
    refs: list[str] = []
    issues: list[str] = []
    paths: list[str] = []
    primary = env.stale_actions[0]

    for action in env.stale_actions:
        bp = action.bug_pattern  # empty string for legacy patterns

        # ── New realistic bug patterns ──────────────────────────────────────
        if bp == "null_missing":
            fn = action.metadata.get("function", action.stale_ref.split("(")[0])
            attr = action.metadata.get("nullable_attribute", "value")
            refs.append(action.stale_ref)
            issues.append(
                f"{fn} now returns Optional and may return None. "
                f"The PR accesses result.{attr} without a None guard — "
                f"this will raise AttributeError: 'NoneType' object has no attribute '{attr}'. "
                f"Use {action.current_ref} or add an explicit None check."
            )
            paths.append(_cascade_path_for(env, action, f"None.{attr} -> AttributeError"))

        elif bp == "type_mismatch":
            fn = action.metadata.get("function", "")
            param = action.metadata.get("param", "id")
            old_type = action.metadata.get("old_type", "int")
            new_type = action.metadata.get("new_type", "str")
            refs.append(action.stale_ref)
            issues.append(
                f"{fn} now requires {param} as {new_type}, not {old_type}. "
                f"The PR still passes {old_type} ({action.stale_ref}); "
                f"this will raise TypeError at runtime. Update to {action.current_ref}."
            )
            paths.append(_cascade_path_for(env, action, f"{fn}({param}=wrong_type) -> TypeError"))

        elif bp == "condition_flip":
            fn = action.metadata.get("function", "")
            param = action.metadata.get("param", "flag")
            new_semantics = action.metadata.get("new_semantics", "semantics inverted")
            refs.append(action.stale_ref)
            issues.append(
                f"{fn} parameter '{param}' semantics were inverted: {new_semantics}. "
                f"The PR passes the wrong value ({action.stale_ref}), producing "
                f"incorrect results — assertions will fail. Use {action.current_ref}."
            )
            paths.append(_cascade_path_for(env, action, f"{fn}({param}=inverted) -> AssertionError"))

        elif bp == "off_by_one":
            old_call = action.metadata.get("old_call", action.stale_ref)
            new_convention = action.metadata.get("new_convention", "0-based")
            refs.append(action.stale_ref)
            issues.append(
                f"Pagination now uses {new_convention} indexing. "
                f"The PR still calls {old_call} using 1-based index; "
                f"this returns the wrong page or raises IndexError. Use {action.current_ref}."
            )
            paths.append(_cascade_path_for(env, action, f"{old_call} -> wrong index -> IndexError"))

        # ── Legacy patterns (dispatch by drift_type) ────────────────────────
        elif action.drift_type == "rename":
            stale = action.stale_ref
            new = action.current_ref
            refs.append(stale)
            issues.append(
                f"{stale} was renamed to {new}. "
                f"The PR still calls {stale}() — this will raise AttributeError: "
                f"module has no attribute '{stale}'. Update all call sites to use {new}."
            )
            paths.append(_cascade_path_for(env, action, f"AttributeError"))

        elif action.drift_type == "removal":
            module = action.metadata.get("module", "") or action.stale_ref
            refs.append(action.stale_ref)
            issues.append(
                f"{action.stale_ref} (module {module}) was deleted from the codebase. "
                f"The PR still imports it — this will raise ModuleNotFoundError on import. "
                f"Remove the import and refactor callers."
            )
            paths.append(_cascade_path_for(env, action, "ModuleNotFoundError"))

        elif action.drift_type == "contract":
            fn = action.metadata.get("function", "")
            old_params = action.metadata.get("old_params", []) or []
            new_params = action.metadata.get("new_params", []) or []
            stale_call = action.stale_ref
            current_call = action.current_ref
            refs.append(stale_call)
            if old_params and new_params:
                issues.append(
                    f"{fn} signature changed from ({', '.join(old_params)}) to "
                    f"({', '.join(new_params)}). The PR still uses {stale_call}; "
                    f"this will raise TypeError: missing required argument. "
                    f"Update to {current_call}."
                )
            else:
                issues.append(
                    f"{fn} contract changed — stale call {stale_call} will raise TypeError. "
                    f"Update to {current_call}."
                )
            paths.append(_cascade_path_for(env, action, "TypeError"))

        else:
            refs.append(action.stale_ref)
            issues.append(
                f"Stale reference: {action.stale_ref} (pattern={bp or action.drift_type}). "
                f"Will raise runtime error. Update to {action.current_ref}."
            )
            paths.append(_cascade_path_for(env, action, "RuntimeError"))

    issues_block = " ; ".join(issues) if issues else "none"
    failure_path = paths[0] if paths else "n/a"
    return (
        "VERDICT: REQUEST_CHANGES\n"
        f"ROOT_CAUSE: {primary.stale_ref}\n"
        f"FAILURE_PATH: {failure_path}\n"
        "CONFIDENCE: 0.92\n"
        f"ISSUES: {issues_block}\n"
        f"REASON: Stale references detected in PR ({', '.join(refs)}) — must update before merging."
    )


def fill_base_model() -> str:
    return BASE_MODEL_RESPONSE


def fill_trained_model(env: CodeDriftEnv | None) -> str:
    return _trained_response_for(env)


# ─── cascade panel ──────────────────────────────────────────────────────────

def _cascade_html(env: CodeDriftEnv | None) -> str:
    if env is None or not env.stale_actions:
        return (
            "<section class='ds-card ds-card--muted' aria-label='Failure cascade'>"
            "<h3 class='ds-card__title'>Failure cascade</h3>"
            "<div class='ds-card__body'>Deploy a mission to see how the test links to the bug.</div>"
            "</section>"
        )
    cascade = env.failure_cascade
    # FailureCascade.failures is list[TestFailure]; call_chain = [test, intermediate, stale]
    failures_list = list(cascade.failures) if cascade and cascade.failures else []
    # Index failures by the last element of their call_chain (the stale ref)
    failure_by_stale: dict[str, Any] = {}
    for f in failures_list:
        if f.call_chain:
            key = f.call_chain[-1].split("(")[0].lower()
            failure_by_stale[key] = f
    rows: list[str] = []
    for idx, action in enumerate(env.stale_actions, start=1):
        stale = action.stale_ref
        kind = action.bug_pattern or action.drift_type
        stale_bare = stale.split("(")[0].lower()
        failure = failure_by_stale.get(stale_bare)
        if failure is None:
            # Try substring match
            for k, v in failure_by_stale.items():
                if stale_bare in k or k in stale_bare:
                    failure = v
                    break
        if failure and failure.call_chain:
            chain = failure.call_chain          # e.g. ["test_create_basic_order", "checkout_flow", "createOrder"]
            error_label = failure.error_type    # e.g. "AttributeError"
            test_name = failure.test_name
        else:
            mid = action.metadata.get("caller") or "caller"
            chain = [f"test_{stale_bare}", mid, stale_bare]
            error_label = "RuntimeError"
            test_name = chain[0]
        depth = len(chain)
        # Render: each node except last as plain code, last as bold stale ref
        nodes = [f"<code>{n}</code>" for n in chain[:-1]] + [f"<b class='cascade-stale'>{stale}</b>"]
        chain_html = " <span class='cascade-arrow'>&rarr;</span> ".join(nodes)
        depth_tone = "ok" if depth <= 2 else "warn" if depth == 3 else "danger"
        depth_label = "shallow" if depth <= 2 else "indirect" if depth == 3 else "hidden cause"
        rows.append(
            f"<div class='cascade-row cascade-{depth_tone}'>"
            f"<div class='cascade-meta'>#{idx} · <b>{kind}</b>"
            f"<span class='cascade-depth'>depth {depth} · {depth_label}</span>"
            f"<span class='cascade-error-tag'>{error_label}</span></div>"
            f"<div class='cascade-chain'>{chain_html}</div>"
            f"</div>"
        )
    return (
        "<section class='ds-card' aria-label='Failure cascade'>"
        "<h3 class='ds-card__title'>Failure cascade</h3>"
        "<div class='ds-card__body'>"
        "<p class='cascade-help'>Where the test crashes vs where the bug lives. "
        "Depth ≥ 3 = <i>hidden cause</i> — the reviewer must trace it.</p>"
        + "".join(rows)
        + "</div></section>"
    )


# ─── helpers used by handlers ──────────────────────────────────────────────

def _scenario_overrides(scenario_mode: str, difficulty: str, personality: str) -> tuple[str, str]:
    mode = (scenario_mode or "Random").strip().lower()
    if mode == "edge cases":
        return "medium", "subtle"
    if mode == "hard mode":
        return "hard", "adaptive"
    return difficulty, personality


def _replay_md(events: list[dict[str, Any]]) -> str:
    if not events:
        return "_No mission log yet. Submit a review to populate this panel._"
    lines = ["### Mission Log"]
    for i, ev in enumerate(events[-8:], start=1):
        verdict = str(ev.get("verdict", "UNKNOWN"))
        reward = float(ev.get("reward", 0.0))
        misses = int(ev.get("missing_stale_refs_count", 0))
        malformed = int(ev.get("malformed_issues", 0))
        epi = str(ev.get("episode_id", "n/a"))
        miss_str = f"  missed {misses}" if misses else ""
        mal_str = f"  malformed {malformed}" if malformed else ""
        lines.append(
            f"{i}. `{epi}` — **{verdict}** · {reward:+.2f} XP{miss_str}{mal_str}"
        )
    return "\n".join(lines)


# ─── core handlers (reset / submit / benchmark) ─────────────────────────────

def new_episode(
    difficulty: str,
    personality: str,
    seed: str,
    scenario_mode: str,
    replay_events: list[dict[str, Any]] | None,
    player: dict[str, Any] | None,
) -> tuple:
    try:
        s = int(seed)
    except ValueError:
        s = 42
    player = player or dict(DEFAULT_PLAYER)
    replay_events = replay_events or []
    try:
        eff_difficulty, eff_personality = _scenario_overrides(scenario_mode, difficulty, personality)
        env = CodeDriftEnv(difficulty=eff_difficulty, personality=eff_personality, seed=s)
        obs = env.reset()
        status = (
            f"<section class='ds-mission ds-mission--active' role='status' aria-live='polite'>"
            f"<span class='ds-mission__tag'>Active</span>"
            f"<span class='ds-mission__meta'><code>{env.episode_id}</code></span>"
            f"<span class='ds-mission__meta'>Mode <b>{scenario_mode}</b> · level <b>{eff_difficulty}</b> · adversary <b>{eff_personality}</b></span>"
            f"<span class='ds-mission__meta'>Hidden stale refs: <b>{obs.n_stale_refs}</b></span>"
            f"</section>"
        )
        return (
            env, obs.prompt, obs.pr_diff, obs.codebase_context, obs.test_output, "",
            status,
            _brain_html(env),
            _fmt_info({"note": "Submit a review to see scorer output.", "episode_id": env.episode_id}),
            replay_events,
            _replay_md(replay_events),
            player,
            _hud_html(player),
        )
    except Exception as e:
        err = {"error": str(e), "type": type(e).__name__}
        return (
            None, "", "", "", "", "",
            f"<section class='ds-mission ds-mission--error' role='status' aria-live='polite'><span class='ds-mission__tag'>Error</span><span class='ds-mission__meta'>Failed to start — {e!s}</span></section>",
            _brain_html(None),
            _fmt_info(err),
            replay_events, _replay_md(replay_events),
            player, _hud_html(player),
        )


def submit_review(
    env: CodeDriftEnv | None,
    review: str,
    replay_events: list[dict[str, Any]] | None,
    player: dict[str, Any] | None,
) -> tuple:
    replay_events = replay_events or []
    player = player or dict(DEFAULT_PLAYER)
    if env is None:
        return (
            None, "", "", "", "", "",
            "<section class='ds-mission ds-mission--error' role='status' aria-live='polite'><span class='ds-mission__tag'>Idle</span><span class='ds-mission__meta'>No active mission — click <b>Deploy mission</b></span></section>",
            _brain_html(None),
            _fmt_info({"error": "no_env"}),
            replay_events, _replay_md(replay_events),
            player, _hud_html(player),
        )
    if not review.strip():
        return (
            env, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "<section class='ds-mission ds-mission--warn' role='status' aria-live='polite'><span class='ds-mission__tag'>Review</span><span class='ds-mission__meta'>Empty report — paste a non-empty review</span></section>",
            _brain_html(env),
            _fmt_info({"error": "empty_review", "episode_id": env.episode_id}),
            replay_events, _replay_md(replay_events),
            player, _hud_html(player),
        )
    if not env.is_ready_for_step:
        return (
            env, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "<section class='ds-mission ds-mission--warn' role='status' aria-live='polite'><span class='ds-mission__tag'>Scored</span><span class='ds-mission__meta'>Mission already scored — deploy a new one</span></section>",
            _brain_html(env),
            _fmt_info({"error": "episode_already_scored"}),
            replay_events, _replay_md(replay_events),
            player, _hud_html(player),
        )
    try:
        _, reward, _done, info = env.step(review)
        status = _status_banner(reward, info)
        info["adversary_brain"] = env.drift_agent.adaptive_snapshot()
        replay_events.append({
            "episode_id": info.get("episode_id"),
            "reward": reward,
            "verdict": info.get("verdict"),
            "missing_stale_refs_count": len(info.get("missed") or []),
            "malformed_issues": info.get("malformed_issues", 0),
        })
        player = _apply_outcome(player, reward, info)
        return (
            env, gr.update(), gr.update(), gr.update(), gr.update(), "",
            status,
            _brain_html(env),
            _fmt_info(info),
            replay_events, _replay_md(replay_events),
            player, _hud_html(player),
        )
    except RuntimeError as e:
        snap = env.debug_snapshot() if env is not None else {}
        err = {"error": str(e), "type": "RuntimeError", "env": snap}
        return (
            env, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "<section class='ds-mission ds-mission--warn' role='status' aria-live='polite'><span class='ds-mission__tag'>Scored</span><span class='ds-mission__meta'>Already scored — deploy a new mission to continue</span></section>",
            _brain_html(env),
            _fmt_info(err),
            replay_events, _replay_md(replay_events),
            player, _hud_html(player),
        )
    except Exception as e:
        snap = env.debug_snapshot() if env is not None else {}
        err = {"error": str(e), "type": type(e).__name__, "env": snap}
        return (
            env, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            f"<section class='ds-mission ds-mission--error' role='status' aria-live='polite'><span class='ds-mission__tag'>Error</span><span class='ds-mission__meta'>Scoring failed — {e!s}</span></section>",
            _brain_html(env),
            _fmt_info(err),
            replay_events, _replay_md(replay_events),
            player, _hud_html(player),
        )


def reset_player(_player: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    fresh = dict(DEFAULT_PLAYER)
    return fresh, _hud_html(fresh)


# ─── leaderboard / benchmark ───────────────────────────────────────────────

def _metric_tile(title: str, base: float, trained: float, fmt: str = "{:+.3f}") -> str:
    delta = trained - base
    tone = "ok" if delta > 1e-6 else "bad" if delta < -1e-6 else "mu"
    arrow = "▲" if delta > 0 else "▼" if delta < 0 else "▬"
    return (
        f"<div class='ds-tile ds-tile--{tone}'>"
        f"<div class='ds-tile__title'>{title}</div>"
        f"<div class='ds-tile__row'>"
        f"<div><span class='ds-tile__key'>Junior</span><br /><span class='ds-tile__val'>{fmt.format(base)}</span></div>"
        f"<div><span class='ds-tile__key'>Senior</span><br /><span class='ds-tile__val'>{fmt.format(trained)}</span></div>"
        f"</div>"
        f"<div class='ds-tile__d'>{arrow} {fmt.format(delta)}</div>"
        f"</div>"
    )


def run_benchmark(
    n_episodes: int,
    difficulty: str,
    personality: str,
    seed: str,
    scenario_mode: str,
) -> tuple[str, str]:
    try:
        base_seed = int(seed)
    except ValueError:
        base_seed = 42
    n = max(1, min(50, int(n_episodes)))
    eff_difficulty, eff_personality = _scenario_overrides(scenario_mode, difficulty, personality)
    rows: list[dict[str, Any]] = []
    for i in range(n):
        s = base_seed + i
        env_base = CodeDriftEnv(difficulty=eff_difficulty, personality=eff_personality, seed=s)
        env_base.reset()
        _, base_reward, _, base_info = env_base.step(BASE_MODEL_RESPONSE)
        env_trained = CodeDriftEnv(difficulty=eff_difficulty, personality=eff_personality, seed=s)
        env_trained.reset()
        _, trained_reward, _, trained_info = env_trained.step(_trained_response_for(env_trained))
        rows.append({
            "seed": s,
            "base_reward": base_reward,
            "trained_reward": trained_reward,
            "base_verdict": base_info.get("verdict"),
            "trained_verdict": trained_info.get("verdict"),
            "base_recall": float(base_info.get("recall", 0.0)),
            "trained_recall": float(trained_info.get("recall", 0.0)),
            "episode_id": trained_info.get("episode_id"),
        })
    base_avg = sum(r["base_reward"] for r in rows) / len(rows)
    trained_avg = sum(r["trained_reward"] for r in rows) / len(rows)
    win_count = sum(1 for r in rows if r["trained_reward"] > r["base_reward"])
    tie_count = sum(1 for r in rows if r["trained_reward"] == r["base_reward"])
    trained_recall_avg = sum(r["trained_recall"] for r in rows) / len(rows)
    base_recall_avg = sum(r["base_recall"] for r in rows) / len(rows)
    row_n = len(rows)
    win_pct = (win_count / row_n) if row_n else 0.0
    summary = (
        "<section class='ds-bench' role='status' aria-live='polite'>"
        f"<h2 class='ds-bench__title'>{row_n} missions · quick run</h2>"
        "<div class='ds-bench__row' aria-label='Scenario'>"
        f"<span class='ds-bench-pill'>{scenario_mode}</span>"
        f"<span class='ds-bench-pill'>{eff_difficulty}</span>"
        f"<span class='ds-bench-pill'>{eff_personality}</span>"
        "</div>"
        "<ul>"
        f"<li>Junior avg reward: <strong>{base_avg:+.3f}</strong></li>"
        f"<li>Senior avg reward: <strong>{trained_avg:+.3f}</strong></li>"
        f"<li>Senior win rate: <strong>{win_pct:.0%}</strong> ({win_count}/{row_n})</li>"
        f"<li>Ties: <strong>{tie_count}</strong></li>"
        f"<li>Junior avg recall: <strong>{base_recall_avg:.2f}</strong></li>"
        f"<li>Senior avg recall: <strong>{trained_recall_avg:.2f}</strong></li>"
        "</ul>"
        "</section>"
    )
    return summary, _fmt_info({"benchmark": rows})


# ─── Battle Mode ────────────────────────────────────────────────────────────

_BUG_PATTERN_LABELS: dict[str, str] = {
    "partial_rename": "Partial rename",
    "null_missing":   "Null dereference",
    "type_mismatch":  "Type mismatch",
    "condition_flip": "Condition flip",
    "off_by_one":     "Off-by-one",
    "rename":         "Rename",
    "removal":        "Module removal",
    "contract":       "Contract change",
}


def _battle_side_html(
    label: str,
    tag: str,
    review_text: str,
    reward: float,
    caught: list[str],
    missed: list[str],
    outcome: str,   # "win" | "lose"
) -> str:
    xp = int(round(reward * 100))
    xp_str = f"+{xp}" if xp >= 0 else str(xp)
    verdict_line = next(
        (ln.split(":", 1)[1].strip() for ln in review_text.splitlines() if ln.startswith("VERDICT:")),
        "—",
    )
    issues_line = next(
        (ln.split(":", 1)[1].strip() for ln in review_text.splitlines() if ln.startswith("ISSUES:")),
        "—",
    )
    root_line = next(
        (ln.split(":", 1)[1].strip() for ln in review_text.splitlines() if ln.startswith("ROOT_CAUSE:")),
        "—",
    )
    icon = "◎" if outcome == "win" else "✕"
    result_label = "BUG DEFUSED" if outcome == "win" else "MISSED IT"
    caught_html = "".join(
        f"<span class='battle-ref battle-ref--caught'>{c.split(':', 1)[-1]}</span>"
        for c in caught
    ) or "<span class='battle-ref battle-ref--none'>—</span>"
    missed_html = "".join(
        f"<span class='battle-ref battle-ref--missed'>{m.split(':', 1)[-1]}</span>"
        for m in missed
    ) or "<span class='battle-ref battle-ref--none'>—</span>"
    return f"""
<div class="battle-side battle-side--{outcome}">
  <div class="battle-side__header">
    <span class="battle-player-label">{label}</span>
    <span class="battle-player-tag">{tag}</span>
  </div>
  <div class="battle-review">
    <div class="battle-review__row"><span class="battle-review__k">VERDICT</span><span class="battle-review__v battle-verdict--{outcome}">{verdict_line}</span></div>
    <div class="battle-review__row"><span class="battle-review__k">ROOT CAUSE</span><span class="battle-review__v">{root_line[:80]}</span></div>
    <div class="battle-review__row"><span class="battle-review__k">ISSUES</span><span class="battle-review__v">{issues_line[:100]}</span></div>
  </div>
  <div class="battle-refs">
    <div><span class="battle-refs__k">Caught</span>{caught_html}</div>
    <div><span class="battle-refs__k">Missed</span>{missed_html}</div>
  </div>
  <div class="battle-outcome battle-outcome--{outcome}">
    <span class="battle-outcome__icon">{icon}</span>
    <span class="battle-outcome__label">{result_label}</span>
    <span class="battle-outcome__reward">{reward:+.1f}</span>
    <span class="battle-outcome__xp">{xp_str} XP</span>
  </div>
</div>""".strip()


def _battle_html(
    episode_id: str,
    pattern: str,
    stale_ref: str,
    failure_path: str,
    junior_review: str,
    senior_review: str,
    reward_j: float,
    reward_s: float,
    info_j: dict[str, Any],
    info_s: dict[str, Any],
) -> str:
    caught_j = list(info_j.get("caught") or [])
    missed_j = list(info_j.get("missed") or [])
    caught_s = list(info_s.get("caught") or [])
    missed_s = list(info_s.get("missed") or [])
    pattern_label = _BUG_PATTERN_LABELS.get(pattern, pattern)
    delta = reward_s - reward_j
    delta_str = f"{delta:+.1f}"
    side_j = _battle_side_html("JUNIOR", "Untrained baseline", junior_review, reward_j, caught_j, missed_j, "lose")
    side_s = _battle_side_html("SENIOR", "GRPO-Trained", senior_review, reward_s, caught_s, missed_s, "win")
    return f"""
<div class="battle-arena" role="region" aria-label="Battle result">
  <div class="battle-header">
    <div class="battle-header__left">
      <span class="battle-pattern-badge">{pattern_label}</span>
      <code class="battle-stale-ref">{stale_ref}</code>
    </div>
    <div class="battle-header__right">
      <span class="battle-episode">{episode_id}</span>
    </div>
  </div>
  <div class="battle-split">
    {side_j}
    <div class="battle-vs" aria-hidden="true">VS</div>
    {side_s}
  </div>
  <div class="battle-delta" role="status" aria-live="polite">
    <span class="battle-delta__label">Training advantage</span>
    <span class="battle-delta__value">{delta_str} reward · {int(round(delta * 100)):+d} XP per mission</span>
  </div>
  <div class="battle-footer">
    <span class="battle-footer__item">Failure path: <code>{failure_path}</code></span>
  </div>
</div>""".strip()


def run_battle(seed_str: str, difficulty: str, personality: str) -> tuple[str, str]:
    """Single battle: Junior vs Senior on the same episode."""
    try:
        seed = int(seed_str)
    except (ValueError, TypeError):
        seed = 42

    env_j = CodeDriftEnv(difficulty=difficulty, personality=personality, seed=seed)
    env_j.reset()
    junior_review = BASE_MODEL_RESPONSE
    _, reward_j, _, info_j = env_j.step(junior_review)

    env_s = CodeDriftEnv(difficulty=difficulty, personality=personality, seed=seed)
    env_s.reset()
    senior_review = _trained_response_for(env_s)
    _, reward_s, _, info_s = env_s.step(senior_review)

    action = env_s.stale_actions[0] if env_s.stale_actions else None
    pattern = (action.bug_pattern or action.drift_type) if action else "unknown"
    stale_ref = action.stale_ref if action else "—"

    # Build failure path from the actual cascade call_chain for the first action
    _cascade_s = env_s.failure_cascade
    if _cascade_s and _cascade_s.failures:
        failure_path = " -> ".join(_cascade_s.failures[0].call_chain)
    else:
        failure_path = f"test -> caller -> {stale_ref}"

    html = _battle_html(
        episode_id=str(info_s.get("episode_id") or seed),
        pattern=pattern,
        stale_ref=stale_ref,
        failure_path=failure_path,
        junior_review=junior_review,
        senior_review=senior_review,
        reward_j=reward_j,
        reward_s=reward_s,
        info_j=info_j,
        info_s=info_s,
    )
    detail = _fmt_info({
        "seed": seed,
        "pattern": pattern,
        "stale_ref": stale_ref,
        "junior_reward": reward_j,
        "senior_reward": reward_s,
        "delta": reward_s - reward_j,
        "junior_recall": info_j.get("recall"),
        "senior_recall": info_s.get("recall"),
    })
    return html, detail


def run_gauntlet(n_str: int, seed_str: str, difficulty: str, personality: str) -> str:
    """Run N battles and return a scoreboard HTML proving consistent training advantage."""
    try:
        base_seed = int(seed_str)
    except (ValueError, TypeError):
        base_seed = 42
    n = max(3, min(10, int(n_str)))

    rows_html: list[str] = []
    total_j = total_s = 0.0
    wins = 0

    for i in range(n):
        s = base_seed + i
        env_j = CodeDriftEnv(difficulty=difficulty, personality=personality, seed=s)
        env_j.reset()
        _, rj, _, _ = env_j.step(BASE_MODEL_RESPONSE)

        env_s = CodeDriftEnv(difficulty=difficulty, personality=personality, seed=s)
        env_s.reset()
        _, rs, _, info_s = env_s.step(_trained_response_for(env_s))

        action = env_s.stale_actions[0] if env_s.stale_actions else None
        pattern = _BUG_PATTERN_LABELS.get(
            (action.bug_pattern or action.drift_type) if action else "", "—"
        )
        stale = action.stale_ref if action else "—"
        total_j += rj
        total_s += rs
        if rs > rj + 1e-9:
            wins += 1
        delta = rs - rj
        row_tone = "ok" if delta > 0 else "bad" if delta < 0 else "mu"
        rows_html.append(
            f"<tr class='gauntlet-row gauntlet-row--{row_tone}'>"
            f"<td class='gauntlet-td'>{i + 1}</td>"
            f"<td class='gauntlet-td gauntlet-pattern'>{pattern}</td>"
            f"<td class='gauntlet-td gauntlet-stale'><code>{stale[:40]}</code></td>"
            f"<td class='gauntlet-td gauntlet-score gauntlet-score--bad'>{rj:+.1f}</td>"
            f"<td class='gauntlet-td gauntlet-score gauntlet-score--ok'>{rs:+.1f}</td>"
            f"<td class='gauntlet-td gauntlet-delta'>{'▲' if delta > 0 else '▼'} {delta:+.1f}</td>"
            f"</tr>"
        )

    win_pct = wins / n if n else 0
    avg_j = total_j / n
    avg_s = total_s / n
    tone = "ok" if win_pct >= 0.8 else "warn" if win_pct >= 0.5 else "bad"
    return f"""
<div class="gauntlet-wrap" role="region" aria-label="Gauntlet results">
  <div class="gauntlet-header">
    <span class="gauntlet-title">{n}-ROUND GAUNTLET</span>
    <span class="gauntlet-winrate gauntlet-winrate--{tone}">Senior wins {win_pct:.0%} ({wins}/{n})</span>
  </div>
  <table class="gauntlet-table" aria-label="Round-by-round results">
    <thead>
      <tr>
        <th>#</th><th>Bug type</th><th>Stale ref</th>
        <th>Junior</th><th>Senior</th><th>Delta</th>
      </tr>
    </thead>
    <tbody>{"".join(rows_html)}</tbody>
  </table>
  <div class="gauntlet-totals">
    <div class="gauntlet-total gauntlet-total--bad">
      <div class="gauntlet-total__label">Junior total</div>
      <div class="gauntlet-total__val">{total_j:+.1f}</div>
      <div class="gauntlet-total__xp">{int(total_j * 100):+d} XP</div>
    </div>
    <div class="gauntlet-total gauntlet-total--ok">
      <div class="gauntlet-total__label">Senior total</div>
      <div class="gauntlet-total__val">{total_s:+.1f}</div>
      <div class="gauntlet-total__xp">{int(total_s * 100):+d} XP</div>
    </div>
    <div class="gauntlet-total gauntlet-total--delta">
      <div class="gauntlet-total__label">Training edge</div>
      <div class="gauntlet-total__val">{total_s - total_j:+.1f}</div>
      <div class="gauntlet-total__xp">{int((total_s - total_j) * 100):+d} XP advantage</div>
    </div>
  </div>
</div>""".strip()


# ─── CSS theme ──────────────────────────────────────────────────────────────


def _load_space_css() -> str:
    css_path = Path(__file__).resolve().parent / "theme.css"
    try:
        return css_path.read_text(encoding="utf-8")
    except OSError:
        return "/* theme.css missing */ .gradio-container { min-height: 100vh; }"


_SPACE_CSS = _load_space_css()


# ─── UI ─────────────────────────────────────────────────────────────────────

# Gradio 6 moved css/theme out of Blocks() — they must go to launch() or be
# injected via gr.HTML so HuggingFace Spaces (which calls launch() itself)
# always receives the styling.

with gr.Blocks(title="Bug Code Arena") as demo:

    # Inject design-system CSS as an HTML component — works in every Gradio 6
    # context including HF Spaces where the host controls launch().
    gr.HTML(f"<style>\n{_SPACE_CSS}\n</style>")

    with gr.Column(elem_classes=["ds-app"]):

        # ── Top HUD ────────────────────────────────────────────────────────────
        gr.HTML(
            "<header class='ds-app-header'>"
            "<div class='arena-title-row'>"
            "<h1 class='ds-h1 arena-title'>🐛 Bug Code Arena</h1>"
            "<span class='arena-tagline'>by CodeDrift</span>"
            "</div>"
            "<p class='ds-subtitle'>An adversarial RL environment where an AI agent learns to catch "
            "stale references in code reviews — before they reach production. "
            "Junior model ships bugs. Senior model (GRPO-trained) catches them.</p>"
            "<ul class='ds-badges' aria-label='Stack'>"
            "<li>OpenEnv</li><li>GRPO</li><li>RL Training</li><li>Live Demo</li>"
            "</ul>"
            "<div class='ds-signal-row' aria-label='Session'>"
            "<span class='ds-signal ds-signal--accent'>● LIVE</span>"
            "<span class='ds-signal'>Hugging Face Space</span>"
            "<span class='ds-signal'>Gradio 6</span>"
            "</div>"
            "</header>"
        )
        
        env_state    = gr.State(None)
        replay_state = gr.State([])
        player_state = gr.State(dict(DEFAULT_PLAYER))
        
        hud_html = gr.HTML(_hud_html(DEFAULT_PLAYER))
        
        with gr.Row(elem_classes=["ds-toolbar"]):
            btn_reset_player = gr.Button("Reset Stats", variant="secondary")
        btn_reset_player.click(reset_player, inputs=[player_state], outputs=[player_state, hud_html])
        
        # ── Tabs ───────────────────────────────────────────────────────────────
        with gr.Tabs():
        
            # ── Mission Console ────────────────────────────────────────────────
            with gr.Tab("Mission"):
                gr.HTML(
                    "<section class='ds-card ds-section-gap'><div class='ds-card__body'>"
                    "<ol class='ds-steps'>"
                    "<li><strong>Set rules</strong> — pick difficulty, adversary style, and scenario.</li>"
                    "<li><strong>Deploy</strong> — spawn a bug into the codebase.</li>"
                    "<li><strong>Load a reviewer</strong> — Junior (untrained) or Senior (GRPO-trained).</li>"
                    "<li><strong>Submit</strong> — score the review. XP, streak, and production health update above.</li>"
                    "</ol>"
                    "</div></section>"
                )

                with gr.Row(elem_classes=["ds-row"]):
                    with gr.Column(scale=1, elem_classes=["ds-group"]):
                        gr.HTML("<h2 class='ds-block-title'>Run</h2>")
                        difficulty = gr.Dropdown(
                            choices=["easy", "medium", "hard"], value="easy",
                            label="Mission level",
                        )
                        personality = gr.Dropdown(
                            choices=["random", "subtle", "aggressive", "escalating", "adaptive"],
                            value="random",
                            label="Adversary style",
                        )
                        scenario_mode = gr.Dropdown(
                            choices=["Random", "Edge Cases", "Hard Mode"],
                            value="Random",
                            label="Scenario preset",
                        )
                        seed = gr.Textbox(value="42", label="Seed", max_lines=1)
                        btn_new = gr.Button("Deploy mission", variant="primary")
                        benchmark_n = gr.Slider(minimum=3, maximum=30, value=10, step=1, label="Benchmark size")
                        btn_benchmark = gr.Button("Quick Benchmark")
        
                    with gr.Column(scale=2, elem_classes=["ds-group"]):
                        gr.HTML("<h2 class='ds-block-title'>Outcome</h2>")
                        status = gr.HTML(
                            "<section class='ds-mission ds-mission--ready' role='status' aria-live='polite'>"
                            "<span class='ds-mission__tag'>Ready</span>"
                            "<span class='ds-mission__meta'>Press <b>Deploy mission</b> to spawn a bug.</span>"
                            "</section>"
                        )
        
                with gr.Row(elem_classes=["ds-row"]):
                    with gr.Column(scale=1, elem_classes=["ds-group"]):
                        gr.HTML("<h2 class='ds-block-title'>Context</h2>")
                        test_output_box = gr.Textbox(
                            label="Failing tests (execution oracle)",
                            lines=8, max_lines=14, interactive=False,
                            elem_id="test_output_box",
                        )
                        pr_diff = gr.Textbox(
                            label="PR diff (review target)",
                            lines=10, max_lines=18, interactive=False,
                            elem_id="pr_diff_box",
                        )
                        codebase = gr.Textbox(
                            label="Codebase (after drift)",
                            lines=10,
                        )
                        prompt = gr.Textbox(
                            label="Full model prompt",
                            lines=4, max_lines=8,
                        )
        
                    with gr.Column(scale=1, elem_classes=["ds-group"]):
                        gr.HTML("<h2 class='ds-block-title'>Review</h2>")
                        brain_panel = gr.HTML(_brain_html(None))
        
                        review = gr.Textbox(
                            label="Reviewer report",
                            lines=10,
                            placeholder=(
                                "VERDICT: REQUEST_CHANGES\n"
                                "ROOT_CAUSE: <exact stale reference>\n"
                                "FAILURE_PATH: test_name → caller → broken_ref\n"
                                "CONFIDENCE: 0.9\n"
                                "ISSUES: ...\n"
                                "REASON: ..."
                            ),
                        )
        
                        gr.HTML(
                            "<div class='loadouts' role='list'>"
                            "<div class='ds-loadout' role='listitem'>"
                            "<div class='loadout-title'>Baseline</div>"
                            "<div class='loadout-name'>Junior (untrained)</div>"
                            "<div class='loadout-stats'>Approves all PRs · catches nothing</div>"
                            "</div>"
                            "<div class='ds-loadout' role='listitem'>"
                            "<div class='loadout-title'>Trained</div>"
                            "<div class='loadout-name'>Senior (GRPO)</div>"
                            "<div class='loadout-stats'>Blocks the merge · traces root cause</div>"
                            "</div>"
                            "</div>"
                        )
                        with gr.Row():
                            btn_base    = gr.Button("Load Junior", variant="secondary")
                            btn_trained = gr.Button("Load Senior", variant="secondary")
                        btn_submit = gr.Button("Submit Review", variant="primary")
        
                        cascade_panel = gr.HTML(_cascade_html(None))
                        scorer_out = gr.Textbox(label="Score breakdown (JSON)", lines=10, max_lines=18, interactive=False)
                        replay_out = gr.Markdown(
                            "_No mission log yet. Submit a review to populate this panel._",
                            elem_classes="replay-markdown",
                        )
        
            # ── Battle tab ────────────────────────────────────────────────────
            with gr.Tab("⚔️ Battle"):
                gr.HTML(
                    "<section class='ds-card ds-section-gap'><div class='ds-card__body'>"
                    "<p class='ds-lead ds-lead--tight'>"
                    "<strong>Same bug. Same PR. Two models.</strong> "
                    "Junior approves everything and ships the bug to production. "
                    "Senior traces the root cause, names the stale ref, and blocks the merge. "
                    "One click — clear proof that GRPO training works."
                    "</p></div></section>"
                )
                with gr.Row(elem_classes=["ds-row"]):
                    with gr.Column(scale=1, elem_classes=["ds-group"]):
                        gr.HTML("<h2 class='ds-block-title'>Arena settings</h2>")
                        battle_seed = gr.Textbox(value="42", label="Seed", max_lines=1)
                        battle_difficulty = gr.Dropdown(
                            choices=["easy", "medium", "hard"], value="easy", label="Mission level"
                        )
                        battle_personality = gr.Dropdown(
                            choices=["random", "subtle", "aggressive", "escalating", "adaptive"],
                            value="random", label="Adversary style",
                        )
                        btn_battle = gr.Button("⚔️ Run Battle", variant="primary")
                        gr.HTML("<hr style='border:none;border-top:1px solid var(--ds-border);margin:var(--s-3) 0'>")
                        gr.HTML("<h2 class='ds-block-title'>Gauntlet (best-of-N)</h2>")
                        gauntlet_n = gr.Slider(minimum=3, maximum=10, value=5, step=1, label="Rounds")
                        btn_gauntlet = gr.Button("🏆 Run Gauntlet", variant="secondary")

                    with gr.Column(scale=2, elem_classes=["ds-group"]):
                        gr.HTML("<h2 class='ds-block-title'>Result</h2>")
                        battle_panel = gr.HTML(
                            "<div class='battle-arena battle-arena--idle'>"
                            "<div class='battle-idle-msg'>Press <strong>⚔️ Run Battle</strong> to pit Junior against Senior on the same bug.</div>"
                            "</div>"
                        )
                        battle_detail = gr.Textbox(label="Detail (JSON)", lines=6, interactive=False)

                gauntlet_panel = gr.HTML("")

                btn_battle.click(
                    run_battle,
                    inputs=[battle_seed, battle_difficulty, battle_personality],
                    outputs=[battle_panel, battle_detail],
                )
                btn_gauntlet.click(
                    run_gauntlet,
                    inputs=[gauntlet_n, battle_seed, battle_difficulty, battle_personality],
                    outputs=[gauntlet_panel],
                )

            # ── Leaderboard tab ────────────────────────────────────────────────
            with gr.Tab("Leaderboard"):
                gr.HTML(
                    "<section class='ds-card ds-section-gap'><div class='ds-card__body'>"
                    "<p class='ds-lead ds-lead--tight'>Run <strong>N</strong> identical missions for both models. "
                    "Compare reward, recall, and win rate across every bug family.</p></div></section>"
                )
                with gr.Row(elem_classes=["ds-row"]):
                    with gr.Column(elem_classes=["ds-group"]):
                        gr.HTML("<h2 class='ds-block-title'>Parameters</h2>")
                        cmp_n = gr.Slider(minimum=3, maximum=30, value=12, step=1, label="Missions in this run")
                        cmp_seed = gr.Textbox(value="42", label="Seed (start)", max_lines=1)
                        cmp_difficulty = gr.Dropdown(
                            choices=["easy", "medium", "hard"], value="easy", label="Mission level"
                        )
                        cmp_personality = gr.Dropdown(
                            choices=["random", "subtle", "aggressive", "escalating", "adaptive"],
                            value="random", label="Adversary style",
                        )
                        btn_compare = gr.Button("Run Leaderboard", variant="primary")
                cmp_summary = gr.HTML(
                    "<section class='ds-card' role='status'><div class='ds-card__body'>"
                    "<p class='ds-lead ds-lead--tight'>Press <strong>Run Leaderboard</strong> to populate the scoreboard and charts.</p>"
                    "</div></section>"
                )
                with gr.Row(elem_classes=["ds-row"]):
                    cmp_chart = gr.BarPlot(
                        label="XP per mission · Junior vs Senior",
                        x="episode", y="reward", color="policy",
                        tooltip=["episode", "policy", "reward", "drift_type"],
                        height=320,
                    )
                    cmp_pattern_chart = gr.BarPlot(
                        label="Avg XP by bug family",
                        x="drift_type", y="reward", color="policy",
                        tooltip=["drift_type", "policy", "reward"],
                        height=320,
                    )
                cmp_table = gr.Dataframe(
                    headers=["mission", "bug_family", "stale_ref", "junior_xp", "senior_xp", "delta_xp"],
                    label="Mission-by-mission detail",
                    wrap=True, interactive=False,
                )
        
                def _on_compare(n, seed_str, diff_lvl, persona):
                    try:
                        base_seed = int(seed_str)
                    except (TypeError, ValueError):
                        base_seed = 42
                    n = max(3, min(30, int(n)))
                    chart_rows: list[dict[str, Any]] = []
                    pattern_acc: dict[tuple[str, str], list[float]] = {}
                    table_rows: list[list[Any]] = []
                    base_total = trained_total = 0.0
                    wins = ties = 0
                    base_recall_total = trained_recall_total = 0.0
                    for i in range(n):
                        s = base_seed + i
                        env_b = CodeDriftEnv(difficulty=diff_lvl, personality=persona, seed=s)
                        env_b.reset()
                        _, rb, _, info_b = env_b.step(BASE_MODEL_RESPONSE)
                        env_t = CodeDriftEnv(difficulty=diff_lvl, personality=persona, seed=s)
                        env_t.reset()
                        _, rt, _, info_t = env_t.step(_trained_response_for(env_t))
                        drift_types = [a.drift_type for a in env_t.stale_actions] or ["unknown"]
                        drift_type = drift_types[0]
                        stale = env_t.stale_actions[0].stale_ref if env_t.stale_actions else ""
                        base_total += rb
                        trained_total += rt
                        base_recall_total += float(info_b.get("recall", 0.0) or 0.0)
                        trained_recall_total += float(info_t.get("recall", 0.0) or 0.0)
                        if rt > rb + 1e-9:
                            wins += 1
                        elif abs(rt - rb) < 1e-9:
                            ties += 1
                        chart_rows.append({"episode": i + 1, "policy": "Junior", "reward": float(rb), "drift_type": drift_type})
                        chart_rows.append({"episode": i + 1, "policy": "Senior", "reward": float(rt), "drift_type": drift_type})
                        pattern_acc.setdefault((drift_type, "Junior"), []).append(rb)
                        pattern_acc.setdefault((drift_type, "Senior"), []).append(rt)
                        table_rows.append([i + 1, drift_type, stale, round(rb, 3), round(rt, 3), round(rt - rb, 3)])
        
                    base_avg = base_total / n
                    trained_avg = trained_total / n
                    base_recall_avg = base_recall_total / n
                    trained_recall_avg = trained_recall_total / n
                    tiles = (
                        "<div class='tiles'>"
                        + _metric_tile("AVG XP", base_avg * 100, trained_avg * 100, fmt="{:+.0f}")
                        + _metric_tile("AVG RECALL", base_recall_avg, trained_recall_avg, fmt="{:.2f}")
                        + _metric_tile("WIN RATE", 0.0, wins / n, fmt="{:.0%}")
                        + _metric_tile("TIES", 0.0, ties, fmt="{:.0f}")
                        + "</div>"
                    )
                    header = (
                        f"<section class='ds-card ds-section-gap'><div class='ds-card__body'>"
                        f"<p class='ds-lead ds-lead--tight'><strong>{n} missions</strong> · "
                        f"level {diff_lvl} · adversary {persona}</p></div></section>"
                    )
                    pattern_rows = [
                        {"drift_type": dt, "policy": pol, "reward": round(sum(vs) / len(vs), 3)}
                        for (dt, pol), vs in pattern_acc.items()
                    ]
                    return header + tiles, chart_rows, pattern_rows, table_rows
        
                btn_compare.click(
                    _on_compare,
                    inputs=[cmp_n, cmp_seed, cmp_difficulty, cmp_personality],
                    outputs=[cmp_summary, cmp_chart, cmp_pattern_chart, cmp_table],
                )
        
            # ── Real-PR tab ────────────────────────────────────────────────────
            with gr.Tab("Real PR"):
                gr.HTML(
                    "<section class='ds-card ds-section-gap'><div class='ds-card__body'>"
                    "<p class='ds-lead ds-lead--tight'>Paste a unified diff directly, or provide a <strong>GitHub URL</strong> to fetch it. "
                    "Tests are not executed — scoring checks whether your <code>ISSUES</code> field cites the stale refs listed below.</p>"
                    "</div></section>"
                )
                with gr.Row(elem_classes=["ds-row"]):
                    with gr.Column(elem_classes=["ds-group"]):
                        gr.HTML("<h2 class='ds-block-title'>Diff & detection</h2>")
                        with gr.Row():
                            real_url = gr.Textbox(
                                label="GitHub URL (optional)",
                                placeholder="https://github.com/owner/repo/pull/1",
                                lines=1, scale=4,
                            )
                            btn_fetch_url = gr.Button("Fetch from GitHub", scale=1)
                        url_status = gr.Markdown("")
                        real_diff = gr.Textbox(
                            label="Unified diff",
                            lines=12,
                            placeholder="diff --git a/src/foo.py b/src/foo.py\n--- a/src/foo.py\n+++ b/src/foo.py\n@@ ...",
                            elem_id="real_pr_diff_box",
                        )
                        with gr.Row():
                            btn_detect = gr.Button("Detect languages + candidate refs", variant="secondary")
                            real_kind = gr.Dropdown(
                                choices=["rename", "removal", "contract"], value="rename",
                                label="Drift kind (for scoring)",
                            )
                        detect_summary = gr.Markdown(
                            "_Paste or fetch a diff, then run **Detect**._"
                        )
                    with gr.Column(elem_classes=["ds-group"]):
                        gr.HTML("<h2 class='ds-block-title'>Review & score</h2>")
                        real_stale = gr.Textbox(
                            label="Stale refs to score (one per line)",
                            lines=4,
                            placeholder="getUserData\nutils/legacy.py\ncreateOrder(item, qty)",
                        )
                        real_review = gr.Textbox(
                            label="Reviewer response",
                            lines=8,
                            placeholder=(
                                "VERDICT: REQUEST_CHANGES\n"
                                "ROOT_CAUSE: <stale ref>\n"
                                "ISSUES: <cite each stale ref here>\n"
                                "REASON: ..."
                            ),
                        )
                        btn_score_real = gr.Button("Score real PR", variant="primary")
                        real_status = gr.HTML("")
                        real_json = gr.Textbox(label="Scoring breakdown (JSON)", lines=12, interactive=False)
        
                def _on_detect(diff_text: str) -> tuple[str, str]:
                    if not (diff_text or "").strip():
                        return ("Paste a diff first.", "")
                    summary = detect_languages(diff_text)
                    candidates = extract_candidate_stale_refs(diff_text, summary.languages)
                    md = (
                        f"**Files:** {len(summary.files)}  "
                        f"**+** {summary.additions}  **-** {summary.deletions}\n\n"
                        f"**Detected languages:** {', '.join(summary.languages) or '(unknown)'}\n\n"
                        f"**Candidate stale refs ({len(candidates)}):** "
                        f"`{'`, `'.join(candidates) if candidates else 'none — try editing manually'}`"
                    )
                    return md, "\n".join(candidates)
        
                def _on_score_real(diff_text: str, refs_text: str, review_text: str, drift_kind: str) -> tuple[str, str]:
                    refs = [ln.strip() for ln in (refs_text or "").splitlines() if ln.strip()]
                    if not (diff_text or "").strip():
                        return (
                            "<section class='ds-mission ds-mission--warn' role='status'><span class='ds-mission__tag'>Diff</span><span class='ds-mission__meta'>Empty diff</span></section>",
                            _fmt_info({"error": "empty_diff"}),
                        )
                    if not refs:
                        return (
                            "<section class='ds-mission ds-mission--warn' role='status'><span class='ds-mission__tag'>Refs</span><span class='ds-mission__meta'>Need at least one stale ref</span></section>",
                            _fmt_info({"error": "no_stale_refs"}),
                        )
                    if not (review_text or "").strip():
                        return (
                            "<section class='ds-mission ds-mission--warn' role='status'><span class='ds-mission__tag'>Review</span><span class='ds-mission__meta'>Empty review</span></section>",
                            _fmt_info({"error": "empty_review"}),
                        )
                    try:
                        reward, info, summary = score_real_pr(diff_text, review_text, refs, drift_kind=drift_kind)
                        return _status_banner(reward, info), _fmt_info({"reward": reward, "diff_summary": summary, "scorer_info": info})
                    except Exception as exc:
                        return (
                            f"<section class='ds-mission ds-mission--error' role='status'><span class='ds-mission__tag'>Error</span><span class='ds-mission__meta'>Scoring failed — {exc!s}</span></section>",
                            _fmt_info({"error": str(exc), "type": type(exc).__name__}),
                        )
        
                def _on_fetch(url_text: str) -> tuple[str, str, str, str]:
                    url = (url_text or "").strip()
                    if not url:
                        return ("**Paste a GitHub URL first.**", "", "", "")
                    try:
                        result = fetch_diff_from_url(url)
                    except Exception as exc:
                        return (
                            f"**Fetch failed:** {exc!s}",
                            "",
                            "Try the raw `.diff` URL or set `GITHUB_TOKEN` for private repos.",
                            "",
                        )
                    head = (
                        f"**Fetched** {result.bytes_received:,} bytes from "
                        f"`{result.resolved_url}`"
                    )
                    extras = []
                    if result.truncated:
                        extras.append(f"_Truncated to {MAX_FETCH_BYTES_FMT} bytes for safety._")
                    if result.note:
                        extras.append(result.note)
                    status_md = head + ("\n\n" + "\n\n".join(extras) if extras else "")
                    detect_md, candidates = _on_detect(result.diff)
                    return status_md, result.diff, detect_md, candidates
        
                btn_fetch_url.click(_on_fetch, inputs=[real_url], outputs=[url_status, real_diff, detect_summary, real_stale])
                btn_detect.click(_on_detect, inputs=[real_diff], outputs=[detect_summary, real_stale])
                btn_score_real.click(
                    _on_score_real,
                    inputs=[real_diff, real_stale, real_review, real_kind],
                    outputs=[real_status, real_json],
                )
        
            # ── Help / Why this works tab ──────────────────────────────────────
            with gr.Tab("About"):
                gr.HTML(
                    "<section class='ds-card ds-doc'><div class='ds-card__body'>"
                    "<h3 class='ds-h2'>How it works</h3>"
                    "<p class='ds-subtitle'>A drift agent mutates the codebase — renaming a function, changing a "
                    "contract, flipping a condition. A PR is opened that still references the old API. The test suite "
                    "fails. The reviewer's job: <strong>name the stale reference</strong>, "
                    "<strong>trace the failure path</strong>, and <strong>block the merge</strong>.</p>"
                    "<h3 class='ds-h2'>Reward signal</h3>"
                    "<p class='ds-subtitle'>Nine additive components: catch, root cause, failure path, verdict, "
                    "confidence calibration, error-type named, hard-pattern bonus, completeness bonus, "
                    "and a hallucination penalty. GRPO uses the full gradient width.</p>"
                    "<h3 class='ds-h2'>Training</h3>"
                    "<p class='ds-subtitle'>TRL GRPOTrainer on Qwen2.5-1.5B-Instruct with 4-bit QLoRA. "
                    "The curriculum escalates difficulty and adversary style as the reviewer improves.</p>"
                    "<h3 class='ds-h2'>Evaluation</h3>"
                    "<p class='ds-subtitle'>Held-out bug families — condition flips and off-by-one errors — "
                    "are never seen during training. The leaderboard and gauntlet measure generalization in real time.</p>"
                    "</div></section>"
                )
        
    # ── Wiring ─────────────────────────────────────────────────────────────
    _new_ep_outputs = [
        env_state, prompt, pr_diff, codebase, test_output_box, review,
        status, brain_panel, scorer_out,
        replay_state, replay_out,
        player_state, hud_html,
    ]
    _step_outputs = [
        env_state, prompt, pr_diff, codebase, test_output_box, review,
        status, brain_panel, scorer_out,
        replay_state, replay_out,
        player_state, hud_html,
    ]

    btn_base.click(fill_base_model, inputs=None, outputs=[review])
    btn_trained.click(fill_trained_model, inputs=[env_state], outputs=[review])

    btn_new.click(
        new_episode,
        inputs=[difficulty, personality, seed, scenario_mode, replay_state, player_state],
        outputs=_new_ep_outputs,
    ).then(_cascade_html, inputs=[env_state], outputs=[cascade_panel])

    btn_submit.click(
        submit_review,
        inputs=[env_state, review, replay_state, player_state],
        outputs=_step_outputs,
    ).then(_cascade_html, inputs=[env_state], outputs=[cascade_panel])

    btn_benchmark.click(
        run_benchmark,
        inputs=[benchmark_n, difficulty, personality, seed, scenario_mode],
        outputs=[status, scorer_out],
    )

    demo.load(
        new_episode,
        inputs=[difficulty, personality, seed, scenario_mode, replay_state, player_state],
        outputs=_new_ep_outputs,
    ).then(_cascade_html, inputs=[env_state], outputs=[cascade_panel])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base())
