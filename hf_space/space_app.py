"""Gradio UI for Hugging Face Spaces — Bug Bounty Arena.

A mission-control / arcade-HUD reskin of the CodeDrift environment. Judges can:
    - deploy a mission (env reset),
    - draft a review (or load Junior / Senior loadouts),
    - submit and see XP / streak / production-health update,
    - play the Junior-vs-Senior leaderboard, and
    - hunt bugs in real GitHub PRs (multi-language, URL fetch).

The styling is intentionally distinctive: deep navy-charcoal with amber / mint /
crimson accents, two display faces (Major Mono Display + Chakra Petch), and a
corner-bracket framing language inspired by 80s mission consoles + arcade HUDs.
No left-stripe callouts, no gradient text, no generic AI cyberpunk neon.
"""

from __future__ import annotations

import json
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
    pct = into / step
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
    streak_pill = (
        f"<span class='hud-pill hud-pill-mint'>★ {streak} streak · best {best}</span>"
        if streak >= 2
        else f"<span class='hud-pill hud-pill-muted'>streak {streak} · best {best}</span>"
    )
    last_pill = ""
    if last == "success":
        last_pill = "<span class='hud-pill hud-pill-mint'>last: BUG DEFUSED</span>"
    elif last == "failure":
        last_pill = "<span class='hud-pill hud-pill-danger'>last: PROD DOWN</span>"
    elif last == "partial":
        last_pill = "<span class='hud-pill hud-pill-amber'>last: PARTIAL</span>"

    return f"""
<section class="hud">
  <div class="hud-row">
    <div class="hud-cell">
      <div class="hud-label">CALLSIGN</div>
      <div class="hud-value">{player.get("callsign", "AGENT-001")}</div>
    </div>
    <div class="hud-cell hud-grow">
      <div class="hud-label">LVL {lvl} · {int(player.get("total_xp", 0))} XP · {into}/{step} to next</div>
      <div class="xpbar"><div class="xpbar-fill" style="width:{pct * 100:.1f}%"></div></div>
    </div>
    <div class="hud-cell">
      <div class="hud-label">MISSIONS</div>
      <div class="hud-value">{won}<span class="hud-sub">/{played}</span></div>
    </div>
    <div class="hud-cell">
      <div class="hud-label">STREAK</div>
      <div class="hud-value">{streak_pill}</div>
    </div>
    <div class="hud-cell hud-cell-right">
      <div class="hud-label">PRODUCTION</div>
      <div class="prod prod-{health_class}">
        <span class="prod-dot"></span>
        <span class="prod-text">{health_label}</span>
        <span class="prod-pct">{health}%</span>
      </div>
    </div>
  </div>
  <div class="hud-meta">{last_pill}</div>
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
<section class="banner banner-{tone}">
  <header class="banner-head">
    <span class="banner-badge">{icon} {label}</span>
    <span class="banner-xp">{xp_sign} XP</span>
  </header>
  <div class="banner-prod">{prod}</div>
  <div class="banner-stats">
    <span><span class="stat-key">bugs</span> {n_stale}</span>
    <span><span class="stat-key">caught</span> <em class="ok">{caught_count}</em></span>
    <span><span class="stat-key">missed</span> <em class="danger">{missed_count}</em></span>
  </div>
  {"<div class='banner-line'>" + summary + "</div>" if summary else ""}
  {"<div class='banner-line muted'><b>Why it matters:</b> " + why + "</div>" if why else ""}
  {"<div class='banner-line muted'>" + conf + "</div>" if conf else ""}
  <footer class="banner-foot">mission-id <code>{ep}</code></footer>
</section>
""".strip()


# ─── adversary brain ────────────────────────────────────────────────────────

def _brain_html(env: CodeDriftEnv | None) -> str:
    if env is None:
        return (
            "<div class='hud-card hud-card-muted'>"
            "<div class='hud-card-title'>ADVERSARY BRAIN</div>"
            "<div class='hud-card-body'>Deploy a mission to initialise.</div>"
            "</div>"
        )
    snap = env.drift_agent.adaptive_snapshot()
    if not snap.get("enabled"):
        return (
            "<div class='hud-card hud-card-muted'>"
            "<div class='hud-card-title'>ADVERSARY BRAIN</div>"
            "<div class='hud-card-body'>Available in <code>adaptive</code> personality mode.</div>"
            "</div>"
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
        f"<div class='hud-card hud-card-{tone}'>"
        f"<div class='hud-card-title'>ADVERSARY BRAIN · STAGE {stage.upper()} · EP {ep}</div>"
        f"<div class='hud-card-body'>"
        f"<div class='brain-meta'>reviewer win-rate · 5={wr5:.0%} · 10={wr10:.0%} · 20={wr20:.0%}</div>"
        f"<div class='brain-grid'>{''.join(rows)}</div>"
        f"</div></div>"
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
ISSUES: Click 'Start Mission' first so the trained policy can analyze the actual diff.
REASON: No episode loaded."""


def _trained_response_for(env: CodeDriftEnv | None) -> str:
    """Mission-aware trained response: cites the actual stale refs in this episode."""
    if env is None or not env.stale_actions:
        return TRAINED_FALLBACK_RESPONSE
    refs: list[str] = []
    issues: list[str] = []
    paths: list[str] = []
    primary = env.stale_actions[0]
    for action in env.stale_actions:
        if action.drift_type == "rename":
            stale = action.stale_ref
            new = action.current_ref
            refs.append(stale)
            issues.append(
                f"{stale} is no longer defined — it was renamed to {new}. "
                f"The PR still calls {stale}() and will raise AttributeError."
            )
            paths.append(f"failing_test → caller → {stale}")
        elif action.drift_type == "removal":
            module = action.metadata.get("module", "") or action.stale_ref
            refs.append(action.stale_ref)
            issues.append(
                f"{action.stale_ref} (module {module}) was deleted. "
                f"The PR still imports it; this will raise ModuleNotFoundError on import."
            )
            paths.append(f"failing_test → import {module} → missing module {action.stale_ref}")
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
                    f"({', '.join(new_params)}). The PR still uses {stale_call}."
                )
            else:
                issues.append(
                    f"{fn} contract changed: stale call {stale_call} no longer valid; "
                    f"current is {current_call}."
                )
            paths.append(f"failing_test → caller → {fn}")
        else:
            refs.append(action.stale_ref)
            issues.append(f"Stale reference: {action.stale_ref} (type={action.drift_type}).")
            paths.append(f"failing_test → caller → {action.stale_ref}")

    issues_block = " ; ".join(issues) if issues else "none"
    failure_path = paths[0] if paths else "n/a"
    return (
        "VERDICT: REQUEST_CHANGES\n"
        f"ROOT_CAUSE: {primary.stale_ref}\n"
        f"FAILURE_PATH: {failure_path}\n"
        "CONFIDENCE: 0.92\n"
        f"ISSUES: {issues_block}\n"
        f"REASON: Stale references in PR ({', '.join(refs)}) — must update before merging."
    )


def fill_base_model() -> str:
    return BASE_MODEL_RESPONSE


def fill_trained_model(env: CodeDriftEnv | None) -> str:
    return _trained_response_for(env)


# ─── cascade panel ──────────────────────────────────────────────────────────

def _cascade_html(env: CodeDriftEnv | None) -> str:
    if env is None or not env.stale_actions:
        return (
            "<div class='hud-card hud-card-muted'>"
            "<div class='hud-card-title'>FAILURE CASCADE</div>"
            "<div class='hud-card-body'>Deploy a mission to see how the test crash traces back to the bug.</div>"
            "</div>"
        )
    cascade = env.failure_cascade
    cascade_calls = list(getattr(cascade, "calls", []) or []) if cascade else []
    chains_by_action: dict[str, list[str]] = {}
    for call in cascade_calls:
        key = getattr(call, "stale_ref", "")
        chains_by_action.setdefault(str(key), []).append(
            f"{getattr(call, 'caller', '?')}() at {getattr(call, 'file', '?')}:{getattr(call, 'line', '?')}"
        )
    rows: list[str] = []
    for idx, action in enumerate(env.stale_actions, start=1):
        stale = action.stale_ref
        kind = action.drift_type
        chain_frames = chains_by_action.get(stale, [])
        if not chain_frames and action.metadata:
            surface = action.metadata.get("surface_function") or "failing_test"
            mid = action.metadata.get("intermediate") or action.metadata.get("caller") or "caller"
            chain_frames = [f"{surface}()", f"{mid}()"]
        if not chain_frames:
            chain_frames = ["failing_test()"]
        depth = len(chain_frames) + 1
        nodes = chain_frames + [f"<b>{stale}</b>"]
        chain_html = " <span class='cascade-arrow'>→</span> ".join(nodes)
        depth_tone = "ok" if depth <= 2 else "warn" if depth == 3 else "danger"
        depth_label = "shallow" if depth <= 2 else "indirect" if depth == 3 else "hidden cause"
        rows.append(
            f"<div class='cascade-row cascade-{depth_tone}'>"
            f"<div class='cascade-meta'>#{idx} · {kind}<span class='cascade-depth'>depth {depth} · {depth_label}</span></div>"
            f"<div class='cascade-chain'>{chain_html}</div>"
            f"</div>"
        )
    return (
        "<div class='hud-card'>"
        "<div class='hud-card-title'>FAILURE CASCADE</div>"
        "<div class='hud-card-body'>"
        "<div class='cascade-help'>Where the test crashes vs where the bug actually lives. "
        "Depth ≥ 3 = <i>hidden cause</i> — the reviewer must trace it.</div>"
        + "".join(rows)
        + "</div></div>"
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
        return "_No mission log yet. Submit a mission report to populate this panel._"
    lines = ["### MISSION LOG"]
    for i, ev in enumerate(events[-8:], start=1):
        verdict = str(ev.get("verdict", "UNKNOWN"))
        reward = float(ev.get("reward", 0.0))
        misses = int(ev.get("missing_stale_refs_count", 0))
        malformed = int(ev.get("malformed_issues", 0))
        epi = str(ev.get("episode_id", "n/a"))
        lines.append(
            f"{i}. `{epi}` · verdict={verdict} · reward={reward:+.2f} · missing={misses} · malformed={malformed}"
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
            f"<div class='mission-strip mission-active'>"
            f"<span class='mission-tag'>MISSION ACTIVE</span>"
            f"<span class='mission-id'><code>{env.episode_id}</code></span>"
            f"<span class='mission-meta'>mode <b>{scenario_mode}</b> · level <b>{eff_difficulty}</b> · adversary <b>{eff_personality}</b></span>"
            f"<span class='mission-meta'>ground-truth bugs: <b>{obs.n_stale_refs}</b> (hidden)</span>"
            f"</div>"
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
            f"<div class='mission-strip mission-error'>FAILED TO START · {e!s}</div>",
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
            "<div class='mission-strip mission-error'>NO ACTIVE MISSION · click <b>Deploy Mission</b></div>",
            _brain_html(None),
            _fmt_info({"error": "no_env"}),
            replay_events, _replay_md(replay_events),
            player, _hud_html(player),
        )
    if not review.strip():
        return (
            env, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "<div class='mission-strip mission-warn'>EMPTY REPORT · paste a non-empty review</div>",
            _brain_html(env),
            _fmt_info({"error": "empty_review", "episode_id": env.episode_id}),
            replay_events, _replay_md(replay_events),
            player, _hud_html(player),
        )
    if not env.is_ready_for_step:
        return (
            env, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "<div class='mission-strip mission-warn'>MISSION ALREADY SCORED · deploy a new mission</div>",
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
            "missing_stale_refs_count": info.get("missing_stale_refs_count", 0),
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
            "<div class='mission-strip mission-warn'>ALREADY SCORED · deploy a new mission to continue</div>",
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
            f"<div class='mission-strip mission-error'>SCORING FAILED · {e!s}</div>",
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
    tone = "ok" if delta > 1e-6 else "danger" if delta < -1e-6 else "muted"
    arrow = "▲" if delta > 0 else "▼" if delta < 0 else "▬"
    return (
        f"<div class='tile tile-{tone}'>"
        f"<div class='tile-title'>{title}</div>"
        f"<div class='tile-row'>"
        f"<div><span class='tile-key'>Junior</span><span class='tile-val'>{fmt.format(base)}</span></div>"
        f"<div><span class='tile-key'>Senior</span><span class='tile-val'>{fmt.format(trained)}</span></div>"
        f"</div>"
        f"<div class='tile-delta'>{arrow} {fmt.format(delta)}</div>"
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
    summary = (
        f"### LEADERBOARD · {len(rows)} missions\n"
        f"mode **{scenario_mode}** · level **{eff_difficulty}** · adversary **{eff_personality}**\n\n"
        f"- Junior avg reward: **{base_avg:+.3f}**\n"
        f"- Senior avg reward: **{trained_avg:+.3f}**\n"
        f"- Senior win rate: **{(win_count / len(rows)):.0%}** ({win_count}/{len(rows)})\n"
        f"- Ties: **{tie_count}**\n"
        f"- Junior avg recall: **{base_recall_avg:.2f}**\n"
        f"- Senior avg recall: **{trained_recall_avg:.2f}**"
    )
    return summary, _fmt_info({"benchmark": rows})


# ─── CSS theme ──────────────────────────────────────────────────────────────

_SPACE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Major+Mono+Display&family=Chakra+Petch:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
  --bg:        oklch(15% 0.025 250);
  --bg-2:      oklch(18% 0.028 250);
  --surface:   oklch(21% 0.030 250);
  --surface-2: oklch(25% 0.035 250);
  --border:    oklch(35% 0.040 250);
  --border-2:  oklch(45% 0.050 250);
  --text:      oklch(94% 0.010 250);
  --muted:     oklch(70% 0.015 250);
  --dim:       oklch(55% 0.015 250);

  --amber:   oklch(80% 0.16 75);
  --amber-2: oklch(70% 0.18 70);
  --mint:    oklch(80% 0.16 165);
  --crimson: oklch(66% 0.22 22);
  --violet:  oklch(72% 0.16 290);

  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;

  --display: 'Major Mono Display', ui-monospace, monospace;
  --body:    'Chakra Petch', ui-sans-serif, system-ui, sans-serif;
  --code:    'JetBrains Mono', ui-monospace, Menlo, Consolas, monospace;
}

/* Gradio shell */
.gradio-container, .gradio-container * { font-family: var(--body); }
.gradio-container {
  background: radial-gradient(1200px 700px at 70% -10%, oklch(22% 0.04 250) 0%, var(--bg) 65%) !important;
  color: var(--text) !important;
  min-height: 100vh;
}
.gradio-container code, .gradio-container pre { font-family: var(--code); }

/* Inputs / textareas: terminal-mono */
.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"] {
  font-family: var(--code) !important;
  font-size: 12.5px !important;
  background: var(--bg-2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
.gradio-container textarea:focus,
.gradio-container input:focus {
  border-color: var(--amber) !important;
  outline: none !important;
}
#pr_diff_box textarea, #test_output_box textarea, #real_pr_diff_box textarea {
  font-family: var(--code) !important;
  font-size: 12px !important;
}

/* Labels */
.gradio-container label, .gradio-container .label-wrap span {
  font-family: var(--body) !important;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  font-size: 11px !important;
  color: var(--muted) !important;
}

/* Buttons — arcade chunky */
.gradio-container button {
  font-family: var(--body) !important;
  font-weight: 600 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  border-radius: 8px !important;
  border: 1px solid var(--border-2) !important;
  background: var(--surface) !important;
  color: var(--text) !important;
  transition: transform 120ms ease-out, background 120ms ease-out, border 120ms ease-out;
}
.gradio-container button:hover {
  transform: translateY(-1px);
  background: var(--surface-2) !important;
  border-color: var(--amber) !important;
}
.gradio-container button.primary, .gradio-container .primary > button {
  background: var(--amber) !important;
  color: oklch(20% 0.02 250) !important;
  border-color: var(--amber-2) !important;
}
.gradio-container button.primary:hover, .gradio-container .primary > button:hover {
  background: var(--amber-2) !important;
}

/* Tabs */
.gradio-container .tab-nav button {
  font-family: var(--display) !important;
  letter-spacing: 0.10em !important;
  font-size: 13px !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  color: var(--muted) !important;
}
.gradio-container .tab-nav button.selected {
  color: var(--amber) !important;
  border-bottom-color: var(--amber) !important;
}

/* Generic surface used for every panel */
.gradio-container .block, .gradio-container .form, .gradio-container .gradio-html {
  background: transparent !important;
}

/* ── HUD ───────────────────────────────────────────── */
.hud {
  border: 1px solid var(--border);
  background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface) 100%);
  border-radius: 14px;
  padding: var(--space-4) var(--space-5);
  position: relative;
  overflow: hidden;
}
.hud::before, .hud::after {
  content: '';
  position: absolute;
  width: 14px; height: 14px;
  border: 2px solid var(--amber);
  opacity: 0.6;
}
.hud::before { top: 6px; left: 6px; border-right: none; border-bottom: none; }
.hud::after  { bottom: 6px; right: 6px; border-left: none; border-top: none; }

.hud-row {
  display: grid;
  grid-template-columns: auto 1.4fr auto auto auto;
  gap: var(--space-5);
  align-items: center;
}
.hud-cell-right { justify-self: end; text-align: right; }
.hud-grow { min-width: 220px; }
.hud-label {
  font-family: var(--display);
  font-size: 10.5px;
  letter-spacing: 0.18em;
  color: var(--muted);
  text-transform: uppercase;
  margin-bottom: 4px;
}
.hud-value {
  font-family: var(--display);
  font-size: 22px;
  color: var(--text);
}
.hud-value .hud-sub { font-size: 13px; color: var(--muted); margin-left: 4px; }
.hud-meta { margin-top: var(--space-3); display: flex; gap: var(--space-2); }

.hud-pill {
  font-family: var(--body);
  font-size: 11px;
  letter-spacing: 0.06em;
  padding: 3px 10px;
  border-radius: 999px;
  border: 1px solid var(--border-2);
  background: var(--bg-2);
  text-transform: uppercase;
}
.hud-pill-mint   { color: var(--mint);    border-color: oklch(50% 0.10 165); }
.hud-pill-amber  { color: var(--amber);   border-color: oklch(50% 0.10  75); }
.hud-pill-danger { color: var(--crimson); border-color: oklch(50% 0.12  22); }
.hud-pill-muted  { color: var(--muted); }

/* XP bar */
.xpbar {
  width: 100%;
  height: 12px;
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
  margin-top: 4px;
  position: relative;
}
.xpbar-fill {
  height: 100%;
  background: repeating-linear-gradient(
    90deg,
    var(--amber) 0 8px,
    var(--amber-2) 8px 16px
  );
  transition: width 600ms cubic-bezier(0.22, 1, 0.36, 1);
}

/* Production status */
.prod {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-family: var(--display);
  font-size: 16px;
  padding: 4px 10px;
  border: 1px solid var(--border-2);
  border-radius: 8px;
  background: var(--bg-2);
}
.prod-dot {
  width: 10px; height: 10px; border-radius: 50%;
  box-shadow: 0 0 0 2px var(--bg-2);
}
.prod-pct { font-size: 12px; color: var(--muted); }
.prod-ok      { color: var(--mint);    border-color: oklch(50% 0.10 165); }
.prod-ok      .prod-dot { background: var(--mint); animation: pulse 2.4s infinite; }
.prod-warn    { color: var(--amber);   border-color: oklch(50% 0.10  75); }
.prod-warn    .prod-dot { background: var(--amber); animation: pulse 1.6s infinite; }
.prod-danger  { color: var(--crimson); border-color: oklch(50% 0.12  22); }
.prod-danger  .prod-dot { background: var(--crimson); animation: pulse 0.8s infinite; }

@keyframes pulse {
  0%,100% { opacity: 1; transform: scale(1); }
  50%     { opacity: 0.55; transform: scale(0.85); }
}

/* ── Mission strip (shown on episode load) ───────────── */
.mission-strip {
  display: flex;
  flex-wrap: wrap;
  gap: var(--space-3);
  align-items: center;
  padding: var(--space-3) var(--space-4);
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--surface);
  font-size: 13px;
}
.mission-tag {
  font-family: var(--display);
  letter-spacing: 0.18em;
  font-size: 11px;
  padding: 4px 10px;
  border-radius: 6px;
  background: var(--amber);
  color: oklch(20% 0.02 250);
}
.mission-error .mission-tag, .mission-error { color: var(--crimson); border-color: oklch(45% 0.12 22); }
.mission-warn  .mission-tag { background: var(--amber); }
.mission-error .mission-tag { background: var(--crimson); color: var(--text); }
.mission-id code { color: var(--amber); font-family: var(--code); font-size: 12px; }
.mission-meta { color: var(--muted); }

/* ── Status banner (mission report) ────────────────── */
.banner {
  border: 1px solid var(--border-2);
  border-radius: 14px;
  padding: var(--space-4) var(--space-5);
  background: var(--surface);
  position: relative;
}
.banner-ok      { border-color: oklch(50% 0.10 165); box-shadow: 0 0 0 1px oklch(50% 0.10 165) inset; }
.banner-danger  { border-color: oklch(50% 0.12  22); box-shadow: 0 0 0 1px oklch(50% 0.12  22) inset; }
.banner-warn    { border-color: oklch(50% 0.10  75); box-shadow: 0 0 0 1px oklch(50% 0.10  75) inset; }
.banner-muted   { border-color: var(--border); }
.banner-head {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: var(--space-2);
}
.banner-badge {
  font-family: var(--display);
  letter-spacing: 0.18em;
  font-size: 14px;
}
.banner-ok    .banner-badge { color: var(--mint); }
.banner-warn  .banner-badge { color: var(--amber); }
.banner-danger .banner-badge { color: var(--crimson); }
.banner-muted .banner-badge { color: var(--muted); }
.banner-xp {
  font-family: var(--display);
  font-size: 24px;
  color: var(--amber);
}
.banner-prod { color: var(--text); margin-bottom: var(--space-3); }
.banner-stats {
  display: flex; gap: var(--space-5);
  font-family: var(--code); font-size: 13px;
  padding: var(--space-2) 0; margin-bottom: var(--space-2);
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
}
.banner-stats em { font-style: normal; }
.banner-stats em.ok      { color: var(--mint); }
.banner-stats em.danger  { color: var(--crimson); }
.banner-stats .stat-key  { color: var(--muted); margin-right: 4px; text-transform: uppercase; font-size: 11px; }
.banner-line { font-size: 13.5px; margin-top: 6px; }
.banner-line.muted { color: var(--muted); }
.banner-foot { margin-top: var(--space-3); font-size: 11px; color: var(--dim); }
.banner-foot code { color: var(--amber); }

/* ── Generic side cards (brain, cascade) ──────────── */
.hud-card {
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--surface);
  margin-bottom: var(--space-3);
  overflow: hidden;
}
.hud-card-title {
  font-family: var(--display);
  letter-spacing: 0.16em;
  font-size: 11px;
  color: var(--muted);
  padding: var(--space-2) var(--space-4);
  border-bottom: 1px solid var(--border);
  background: var(--bg-2);
}
.hud-card-body { padding: var(--space-3) var(--space-4); font-size: 13px; }
.hud-card-ok      .hud-card-title { color: var(--mint); }
.hud-card-warn    .hud-card-title { color: var(--amber); }
.hud-card-danger  .hud-card-title { color: var(--crimson); }
.hud-card-muted   .hud-card-title { color: var(--dim); }

.brain-meta { color: var(--muted); font-size: 12px; margin-bottom: var(--space-2); }
.brain-grid { display: grid; gap: 4px; }
.brain-row { display: grid; grid-template-columns: 90px 1fr 50px; align-items: center; gap: var(--space-3); font-family: var(--code); font-size: 12.5px; }
.brain-mode { color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; font-size: 11px; }
.brain-bar  { color: var(--amber); }
.brain-pct  { color: var(--text); text-align: right; }

/* Cascade rows */
.cascade-help { color: var(--muted); font-size: 12px; margin-bottom: var(--space-3); }
.cascade-row {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: var(--space-2) var(--space-3);
  margin-top: var(--space-2);
  background: var(--bg-2);
}
.cascade-ok     { border-color: oklch(40% 0.08 165); }
.cascade-warn   { border-color: oklch(40% 0.08  75); }
.cascade-danger { border-color: oklch(40% 0.10  22); }
.cascade-meta {
  font-family: var(--display); font-size: 10.5px; letter-spacing: 0.16em;
  color: var(--muted); display: flex; justify-content: space-between;
  text-transform: uppercase;
}
.cascade-ok     .cascade-depth { color: var(--mint); }
.cascade-warn   .cascade-depth { color: var(--amber); }
.cascade-danger .cascade-depth { color: var(--crimson); }
.cascade-chain { font-family: var(--code); font-size: 12.5px; margin-top: 4px; }
.cascade-arrow { color: var(--dim); margin: 0 4px; }

/* ── Loadout buttons block ───────────────────────────── */
.loadouts {
  display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-3);
  margin: var(--space-2) 0;
}
.loadout-card {
  border: 1px solid var(--border);
  border-radius: 10px; padding: var(--space-3);
  background: var(--bg-2);
}
.loadout-title { font-family: var(--display); font-size: 12px; letter-spacing: 0.14em; color: var(--muted); }
.loadout-name  { font-size: 14px; font-weight: 600; margin-top: 2px; }
.loadout-stats { font-family: var(--code); font-size: 11px; color: var(--muted); margin-top: 4px; }

/* ── Leaderboard tiles ──────────────────────────────── */
.tiles { display: flex; flex-wrap: wrap; gap: var(--space-3); margin: var(--space-3) 0; }
.tile {
  flex: 1 1 200px;
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--surface);
  padding: var(--space-3) var(--space-4);
}
.tile-title {
  font-family: var(--display); font-size: 10.5px; letter-spacing: 0.16em;
  color: var(--muted); text-transform: uppercase;
}
.tile-row { display: flex; gap: var(--space-4); margin-top: var(--space-2); }
.tile-key { display: block; font-size: 10.5px; color: var(--dim); text-transform: uppercase; letter-spacing: 0.10em; }
.tile-val { font-family: var(--display); font-size: 18px; color: var(--text); }
.tile-delta { font-family: var(--code); font-size: 12.5px; margin-top: 6px; }
.tile-ok     { box-shadow: 0 0 0 1px oklch(45% 0.10 165) inset; }
.tile-ok     .tile-delta { color: var(--mint); }
.tile-danger { box-shadow: 0 0 0 1px oklch(45% 0.12 22) inset; }
.tile-danger .tile-delta { color: var(--crimson); }
.tile-muted  .tile-delta { color: var(--muted); }

/* Help cards */
.help-card {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: var(--space-3) var(--space-4);
  background: var(--bg-2);
  font-size: 13px;
  color: var(--muted);
}
.help-card b, .help-card code { color: var(--text); }

/* H1 / page title */
.page-title {
  font-family: var(--display);
  letter-spacing: 0.20em;
  font-size: clamp(28px, 4vw, 44px);
  color: var(--text);
  margin: 0;
}
.page-sub {
  color: var(--muted); font-size: 14px; margin-top: 4px;
  max-width: 70ch;
}

/* DataFrames */
.gradio-container .table-wrap, .gradio-container .gr-dataframe {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
"""


# ─── output tuples (kept identical to v1 for back-compat) ───────────────────

_NEW_EP_OUTPUTS_COUNT = 13   # env, prompt, pr_diff, codebase, test_output, review, status, brain, scorer, replay_state, replay_out, player_state, hud_html
_STEP_OUTPUTS_COUNT   = 13


# ─── UI ─────────────────────────────────────────────────────────────────────

_BLOCKS_KWARGS: dict[str, Any] = {"title": "Bug Bounty Arena · CodeDrift"}
try:
    # Gradio < 6.0 accepts theme/css on Blocks; Gradio >= 6.0 wants them on launch().
    _gr_major = int(str(getattr(gr, "__version__", "5.0.0")).split(".", 1)[0])
except Exception:
    _gr_major = 5
if _gr_major < 6:
    _BLOCKS_KWARGS["theme"] = gr.themes.Base()
    _BLOCKS_KWARGS["css"] = _SPACE_CSS

with gr.Blocks(**_BLOCKS_KWARGS) as demo:

    # ── Top HUD ────────────────────────────────────────────────────────────
    gr.HTML(
        "<h1 class='page-title'>BUG · BOUNTY · ARENA</h1>"
        "<p class='page-sub'>An adversary mutates the codebase. A PR ships referencing the <em>old world</em>. "
        "Tests crash. Your reviewer agent has one shot to <b>find the bug, name it, and trace the failure path</b> "
        "— or production goes down for real.</p>"
    )

    env_state    = gr.State(None)
    replay_state = gr.State([])
    player_state = gr.State(dict(DEFAULT_PLAYER))

    hud_html = gr.HTML(_hud_html(DEFAULT_PLAYER))

    with gr.Row():
        btn_reset_player = gr.Button("↻ Reset HUD (XP, streak, production)", variant="secondary")
    btn_reset_player.click(reset_player, inputs=[player_state], outputs=[player_state, hud_html])

    # ── Tabs ───────────────────────────────────────────────────────────────
    with gr.Tabs():

        # ── Mission Console ────────────────────────────────────────────────
        with gr.Tab("◤ MISSION CONSOLE"):
            gr.HTML(
                "<div class='help-card'>"
                "<b>1.</b> Tune the mission rules · <b>2.</b> Deploy a mission · <b>3.</b> Load a loadout "
                "(<b>Junior Dev</b> for the failure baseline, <b>Senior Reviewer</b> for the trained policy) · "
                "<b>4.</b> Submit the report. Each scored mission updates your <b>XP</b>, <b>streak</b>, "
                "and <b>production health</b> at the top."
                "</div>"
            )

            with gr.Row():
                with gr.Column(scale=1):
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
                    btn_new = gr.Button("▶ DEPLOY MISSION", variant="primary")
                    benchmark_n = gr.Slider(minimum=3, maximum=30, value=10, step=1, label="Quick leaderboard size")
                    btn_benchmark = gr.Button("▦ QUICK LEADERBOARD")

                with gr.Column(scale=2):
                    status = gr.HTML(
                        "<div class='mission-strip'>"
                        "<span class='mission-tag'>READY</span>"
                        "<span class='mission-meta'>Press <b>Deploy Mission</b> to spawn a bug.</span>"
                        "</div>"
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    test_output_box = gr.Textbox(
                        label="🔴 Failing tests (execution oracle)",
                        lines=8, max_lines=14, interactive=False,
                        elem_id="test_output_box",
                    )
                    pr_diff = gr.Textbox(
                        label="📄 PR diff — what the reviewer must catch",
                        lines=10, max_lines=18, interactive=False,
                        elem_id="pr_diff_box",
                    )
                    codebase = gr.Textbox(
                        label="📦 Current codebase (after drift)",
                        lines=10,
                    )
                    prompt = gr.Textbox(
                        label="🧾 Full model prompt",
                        lines=4, max_lines=8,
                    )

                with gr.Column(scale=1):
                    brain_panel = gr.HTML(_brain_html(None))

                    review = gr.Textbox(
                        label="📝 Reviewer report",
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
                        "<div class='loadouts'>"
                        "<div class='loadout-card'>"
                        "<div class='loadout-title'>LOADOUT · BASELINE</div>"
                        "<div class='loadout-name'>Junior Dev (untrained)</div>"
                        "<div class='loadout-stats'>verdict: APPROVE · catches: 0 · ships bugs</div>"
                        "</div>"
                        "<div class='loadout-card'>"
                        "<div class='loadout-title'>LOADOUT · TRAINED</div>"
                        "<div class='loadout-name'>Senior Reviewer (GRPO)</div>"
                        "<div class='loadout-stats'>verdict: REQUEST_CHANGES · cites stale refs · traces path</div>"
                        "</div>"
                        "</div>"
                    )
                    with gr.Row():
                        btn_base    = gr.Button("LOAD JUNIOR")
                        btn_trained = gr.Button("LOAD SENIOR")
                    btn_submit = gr.Button("⌖ SUBMIT MISSION REPORT", variant="primary")

                    cascade_panel = gr.HTML(_cascade_html(None))
                    scorer_out = gr.Textbox(label="XP breakdown (JSON)", lines=10, max_lines=18, interactive=False)
                    replay_out = gr.Markdown("_No mission log yet. Submit a mission report to populate this panel._")

        # ── Leaderboard tab ────────────────────────────────────────────────
        with gr.Tab("◣ LEADERBOARD"):
            gr.HTML(
                "<div class='help-card'>"
                "Send the <b>Junior</b> and the <b>Senior</b> on the same N missions. "
                "Compare XP, win rate, and which bug families the Senior dominates."
                "</div>"
            )
            with gr.Row():
                cmp_n = gr.Slider(minimum=3, maximum=30, value=12, step=1, label="Missions in this run")
                cmp_seed = gr.Textbox(value="42", label="Seed (start)", max_lines=1)
                cmp_difficulty = gr.Dropdown(
                    choices=["easy", "medium", "hard"], value="easy", label="Mission level"
                )
                cmp_personality = gr.Dropdown(
                    choices=["random", "subtle", "aggressive", "escalating", "adaptive"],
                    value="random", label="Adversary style",
                )
            btn_compare = gr.Button("▶ RUN LEADERBOARD", variant="primary")
            cmp_summary = gr.HTML("<div class='help-card'>Press <b>Run Leaderboard</b> to populate the scoreboard.</div>")
            with gr.Row():
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
                    f"<div class='help-card' style='margin-bottom:8px'>"
                    f"<b>{n} missions</b> · level={diff_lvl} · adversary={persona}"
                    f"</div>"
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
        with gr.Tab("◢ REAL-PR HUNTER"):
            gr.HTML(
                "<div class='help-card'>"
                "<b>Two ways to load a PR:</b><br>"
                "&nbsp;• <b>Paste a unified diff</b> directly, or<br>"
                "&nbsp;• <b>Paste a GitHub URL</b> (PR / commit / compare / raw <code>.diff</code>) "
                "and click <b>Fetch from GitHub</b> — the diff and detected stale refs auto-populate.<br><br>"
                "<b>Honest note:</b> this does <i>not</i> run the PR's real test suite. The reward is based on "
                "whether your review's <code>ISSUES</code> block cites the stale refs you (or the detector) confirmed."
                "</div>"
            )
            with gr.Row():
                real_url = gr.Textbox(
                    label="GitHub URL (PR / commit / compare / raw .diff)",
                    placeholder="https://github.com/bansalbhunesh/codedrift-arena/pull/1",
                    lines=1, scale=4,
                )
                btn_fetch_url = gr.Button("⬇ FETCH FROM GITHUB", scale=1)
            url_status = gr.Markdown("")
            real_diff = gr.Textbox(
                label="Unified diff (paste here OR fetched from URL above)",
                lines=12,
                placeholder="diff --git a/src/foo.py b/src/foo.py\n--- a/src/foo.py\n+++ b/src/foo.py\n@@ ...",
                elem_id="real_pr_diff_box",
            )
            with gr.Row():
                btn_detect = gr.Button("◉ DETECT LANGUAGES + STALE REFS")
                real_kind = gr.Dropdown(
                    choices=["rename", "removal", "contract"], value="rename",
                    label="Drift kind (how to score)",
                )
            detect_summary = gr.Markdown("_Click **Detect** after pasting a diff or fetching one from a URL._")
            real_stale = gr.Textbox(
                label="Stale refs to score against (one per line — edit before scoring)",
                lines=4,
                placeholder="getUserData\nutils/legacy.py\ncreateOrder(item, qty)",
            )
            real_review = gr.Textbox(
                label="Reviewer response (VERDICT / ROOT_CAUSE / ISSUES / REASON)",
                lines=8,
                placeholder=(
                    "VERDICT: REQUEST_CHANGES\n"
                    "ROOT_CAUSE: <stale ref>\n"
                    "ISSUES: <cite each stale ref here>\n"
                    "REASON: ..."
                ),
            )
            btn_score_real = gr.Button("⚖ SCORE REAL PR", variant="primary")
            real_status = gr.HTML("")
            real_json = gr.Textbox(label="Real-PR scoring breakdown (JSON)", lines=12, interactive=False)

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
                    return ("<div class='mission-strip mission-warn'>EMPTY DIFF</div>", _fmt_info({"error": "empty_diff"}))
                if not refs:
                    return (
                        "<div class='mission-strip mission-warn'>NEED ≥1 STALE REF</div>",
                        _fmt_info({"error": "no_stale_refs"}),
                    )
                if not (review_text or "").strip():
                    return (
                        "<div class='mission-strip mission-warn'>EMPTY REVIEW</div>",
                        _fmt_info({"error": "empty_review"}),
                    )
                try:
                    reward, info, summary = score_real_pr(diff_text, review_text, refs, drift_kind=drift_kind)
                    return _status_banner(reward, info), _fmt_info({"reward": reward, "diff_summary": summary, "scorer_info": info})
                except Exception as exc:
                    return (
                        f"<div class='mission-strip mission-error'>SCORING FAILED · {exc!s}</div>",
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
        with gr.Tab("◆ WHY THIS WORKS"):
            gr.HTML(
                "<div class='help-card'>"
                "<b>The setup</b> — A generator agent mutates the codebase (rename, removal, contract change, "
                "or a multi-frame cascade). A PR is shipped that still references the <i>old world</i>. The test "
                "suite crashes — that's the execution oracle. The reviewer's job is to <b>name the stale ref</b>, "
                "<b>cite the failure path</b>, and <b>request changes</b>.<br><br>"
                "<b>The reward</b> — Multi-component causal score: root cause + failure path + verdict + confidence "
                "calibration − hallucination penalty. Easy to optimise honestly, hard to game.<br><br>"
                "<b>The training</b> — TRL GRPO with a small Qwen2.5-1.5B-Instruct + LoRA. The generator runs an "
                "adaptive curriculum (escalates to harder bug families when the reviewer wins too often).<br><br>"
                "<b>The proof</b> — Held-out bug patterns the model never saw in training. The Leaderboard tab "
                "shows reward delta and per-bug-family wins; the V2 evaluator script measures generalisation."
                "</div>"
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
    if _gr_major >= 6:
        demo.launch(theme=gr.themes.Base(), css=_SPACE_CSS)
    else:
        demo.launch()
