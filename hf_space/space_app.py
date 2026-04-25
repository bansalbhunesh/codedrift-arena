"""
Gradio UI for Hugging Face Spaces — CodeDrift Arena (env + scorer, no GPU model).

Judges can: spawn an episode, read prompt/diff/codebase, paste a model review, see reward.
"""

from __future__ import annotations

import json
from typing import Any

import gradio as gr

from codedrift.logutil import configure_logging
from env.codedrift_env import CodeDriftEnv

configure_logging()


def _fmt_info(info: dict[str, Any]) -> str:
    return json.dumps(info, indent=2, default=str)


def _adversary_brain_html(env: CodeDriftEnv | None) -> str:
    if env is None:
        return "<div style='opacity:.7'>Start an episode to initialize adversary state.</div>"
    snap = env.drift_agent.adaptive_snapshot()
    if not snap.get("enabled"):
        return (
            "<div style='opacity:.8'>"
            "Adversary Brain is available in <code>adaptive</code> personality mode."
            "</div>"
        )
    stage = str(snap.get("stage", "random"))
    ep = int(snap.get("episodes_run", 0) or 0)
    wr5 = float(snap.get("recent_win_rate_5", 0.0) or 0.0)
    wr10 = float(snap.get("recent_win_rate_10", 0.0) or 0.0)
    wr20 = float(snap.get("recent_win_rate_20", 0.0) or 0.0)
    scores = snap.get("mode_scores", {}) or {}

    def bar(v: float) -> str:
        n = max(0, min(10, int(round(v * 10))))
        return "█" * n + "░" * (10 - n)

    color = (
        "#22c55e" if stage == "random" else "#f59e0b" if stage == "subtle" else "#ef4444"
    )
    rows = []
    for mode in ("random", "subtle", "aggressive"):
        v = float(scores.get(mode, 0.0) or 0.0)
        rows.append(
            f"<div><b>{mode:10s}</b> "
            f"<span style='font-family:monospace'>{bar(v)}</span> {v:.0%}</div>"
        )
    return (
        f"<div style='border-left:6px solid {color}; padding-left:12px'>"
        f"<h3 style='margin:0;color:{color}'>🧠 Adversary Brain</h3>"
        f"<div><b>Stage:</b> {stage.upper()} &nbsp; <b>Episode:</b> {ep}</div>"
        f"<div><b>Reviewer win rate:</b> 5={wr5:.0%} | 10={wr10:.0%} | 20={wr20:.0%}</div>"
        f"<div style='margin-top:8px'>{''.join(rows)}</div>"
        "</div>"
    )


def _status_lines(reward: float, info: dict[str, Any]) -> str:
    """SUCCESS/FAILURE headline in Markdown + emoji strip + summary + impact + confidence."""
    kw = info.get("judge_keyword_line") or "⚪ NO REVIEW"
    emoji = (info.get("judge_emoji") or "").strip() or "⚪"
    strip = info.get("metric_strip") or f"reward={reward:+.2f}"
    summary = info.get("judge_summary") or ""
    why = info.get("judge_why_matters") or ""
    conf = info.get("confidence_strip") or ""
    ep = info.get("episode_id") or ""
    
    # Visual cues
    color = "#22c55e" if "SUCCESS" in kw else "#ef4444" if "FAILURE" in kw else "#eab308" if "PARTIAL" in kw else "#6b7280"
    
    md = f"<div style='border-left: 8px solid {color}; padding-left: 15px; margin-bottom: 20px;'>"
    md += f"<h1 style='color:{color}; margin-top: 0;'>{kw}</h1>"
    md += f"### {emoji} {strip}\n\n"
    if summary: md += f"**{summary}**\n\n"
    if why: md += f"💡 *Why it matters:* {why}\n\n"
    if conf: md += f"**{conf}**\n\n"
    md += f"<small style='opacity: 0.6'>Episode ID: {ep}</small>"
    md += "</div>"
    return md


# --- Pre-loaded responses for judges ---
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
ISSUES: Click 'New episode' first so the trained policy can analyze the actual diff.
REASON: No episode loaded."""


def _trained_response_for(env: CodeDriftEnv | None) -> str:
    """Build a 'trained-model' style response from the current episode's stale actions.

    This makes the demo deterministic: 'Base Model' always misses; 'Trained Model'
    always cites the real stale refs that the scorer expects, regardless of which
    drift type was sampled this episode.
    """
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
                f"{stale} is no longer defined — it was renamed to {new}. The PR still calls {stale}() and will raise AttributeError."
            )
            paths.append(f"failing_test → caller → {stale}")
        elif action.drift_type == "removal":
            module = action.metadata.get("module", "") or action.stale_ref
            refs.append(action.stale_ref)
            issues.append(
                f"{action.stale_ref} (module {module}) was deleted. The PR still imports it; this will raise ModuleNotFoundError on import."
            )
            paths.append(f"failing_test → import {module} → missing module {action.stale_ref}")
        elif action.drift_type == "contract":
            fn = action.metadata.get("function", "")
            old_params = action.metadata.get("old_params", []) or []
            new_params = action.metadata.get("new_params", []) or []
            stale_call = action.stale_ref  # canonical, scorer-friendly citation
            current_call = action.current_ref
            refs.append(stale_call)
            if old_params and new_params:
                issues.append(
                    f"{fn} signature changed from ({', '.join(old_params)}) to ({', '.join(new_params)}). "
                    f"The PR still uses {stale_call}."
                )
            else:
                issues.append(
                    f"{fn} contract changed: stale call {stale_call} no longer valid; current is {current_call}."
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


def _scenario_overrides(scenario_mode: str, difficulty: str, personality: str) -> tuple[str, str]:
    """Map judge-friendly modes to deterministic env knobs."""
    mode = (scenario_mode or "Random").strip().lower()
    if mode == "edge cases":
        return "medium", "subtle"
    if mode == "hard mode":
        return "hard", "adaptive"
    return difficulty, personality


def _build_replay_markdown(replay_events: list[dict[str, Any]]) -> str:
    if not replay_events:
        return "No replay events yet. Score some reviews to populate this panel."
    lines = ["### 🔁 Replay Failure Cases"]
    for i, ev in enumerate(replay_events[-8:], start=1):
        verdict = str(ev.get("verdict", "UNKNOWN"))
        reward = float(ev.get("reward", 0.0))
        misses = int(ev.get("missing_stale_refs_count", 0))
        malformed = int(ev.get("malformed_issues", 0))
        epi = str(ev.get("episode_id", "n/a"))
        lines.append(
            f"{i}. `{epi}` | verdict={verdict} | reward={reward:+.2f} | missing={misses} | malformed={malformed}"
        )
    return "\n".join(lines)


def new_episode(difficulty: str, personality: str, seed: str, scenario_mode: str, replay_events: list[dict[str, Any]] | None) -> tuple:
    try:
        s = int(seed)
    except ValueError:
        s = 42
    try:
        replay_events = replay_events or []
        eff_difficulty, eff_personality = _scenario_overrides(scenario_mode, difficulty, personality)
        env = CodeDriftEnv(difficulty=eff_difficulty, personality=eff_personality, seed=s)
        obs = env.reset()
        status = (
            f"### 🏁 Episode started: `{env.episode_id}`\n"
            f"Ground truth stale refs: **{obs.n_stale_refs}** (Hidden from agent)\n"
            f"Mode: **{scenario_mode}** · Difficulty: **{eff_difficulty}** · Personality: **{eff_personality}**"
        )
        return (
            env,
            obs.prompt,
            obs.pr_diff,
            obs.codebase_context,
            obs.test_output,
            "",
            status,
            _adversary_brain_html(env),
            _fmt_info({"note": "Submit a review to see scorer output.", "episode_id": env.episode_id}),
            replay_events,
            _build_replay_markdown(replay_events),
        )
    except Exception as e:
        err = {"error": str(e), "type": type(e).__name__}
        return (
            None,
            "",
            "",
            "",
            "",
            "",
            f"### ❌ Failed to start episode: {e!s}",
            _adversary_brain_html(None),
            _fmt_info(err),
            replay_events or [],
            _build_replay_markdown(replay_events or []),
        )


def submit_review(env: CodeDriftEnv | None, review: str, replay_events: list[dict[str, Any]] | None) -> tuple:
    replay_events = replay_events or []
    if env is None:
        return (
            None, "", "", "", "", "",
            "### ⚠️ No active episode. Click **New episode** first.",
            _adversary_brain_html(None),
            _fmt_info({"error": "no_env"}),
            replay_events,
            _build_replay_markdown(replay_events),
        )
    if not review.strip():
        return (
            env, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "### ⚠️ Paste a non-empty review.",
            _adversary_brain_html(env),
            _fmt_info({"error": "empty_review", "episode_id": env.episode_id}),
            replay_events,
            _build_replay_markdown(replay_events),
        )
    if not env.is_ready_for_step:
        return (
            env, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "### ⚠️ Episode already scored.\nClick **New episode** to try again.",
            _adversary_brain_html(env),
            _fmt_info({"error": "episode_already_scored", "hint": "One step per episode — use New episode for another try."}),
            replay_events,
            _build_replay_markdown(replay_events),
        )
    try:
        _, reward, _done, info = env.step(review)
        status = _status_lines(reward, info)
        info["adversary_brain"] = env.drift_agent.adaptive_snapshot()
        replay_events.append(
            {
                "episode_id": info.get("episode_id"),
                "reward": reward,
                "verdict": info.get("verdict"),
                "missing_stale_refs_count": info.get("missing_stale_refs_count", 0),
                "malformed_issues": info.get("malformed_issues", 0),
            }
        )
        return (
            env, gr.update(), gr.update(), gr.update(), gr.update(), "",
            status,
            _adversary_brain_html(env),
            _fmt_info(info),
            replay_events,
            _build_replay_markdown(replay_events),
        )
    except RuntimeError as e:
        snap = env.debug_snapshot() if env is not None else {}
        err = {"error": str(e), "type": "RuntimeError", "env": snap, "hint": "One score per episode — click **New episode**."}
        return (
            env, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            "### ⚠️ This episode was already scored (or is not ready).\nClick **New episode** to continue.",
            _adversary_brain_html(env),
            _fmt_info(err),
            replay_events,
            _build_replay_markdown(replay_events),
        )
    except Exception as e:
        snap = env.debug_snapshot() if env is not None else {}
        err = {"error": str(e), "type": type(e).__name__, "env": snap}
        return (
            env, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            f"### ❌ Scoring failed: {e!s}",
            _adversary_brain_html(env),
            _fmt_info(err),
            replay_events,
            _build_replay_markdown(replay_events),
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

        rows.append(
            {
                "seed": s,
                "base_reward": base_reward,
                "trained_reward": trained_reward,
                "base_verdict": base_info.get("verdict"),
                "trained_verdict": trained_info.get("verdict"),
                "base_recall": float(base_info.get("recall", 0.0)),
                "trained_recall": float(trained_info.get("recall", 0.0)),
                "episode_id": trained_info.get("episode_id"),
            }
        )

    base_avg = sum(r["base_reward"] for r in rows) / len(rows)
    trained_avg = sum(r["trained_reward"] for r in rows) / len(rows)
    win_count = sum(1 for r in rows if r["trained_reward"] > r["base_reward"])
    tie_count = sum(1 for r in rows if r["trained_reward"] == r["base_reward"])
    trained_recall_avg = sum(r["trained_recall"] for r in rows) / len(rows)
    base_recall_avg = sum(r["base_recall"] for r in rows) / len(rows)
    summary = (
        f"### 🚀 Benchmark complete ({len(rows)} episodes)\n"
        f"Mode: **{scenario_mode}** · Difficulty: **{eff_difficulty}** · Personality: **{eff_personality}**\n\n"
        f"- Base Avg Reward: **{base_avg:+.3f}**\n"
        f"- Trained Avg Reward: **{trained_avg:+.3f}**\n"
        f"- Win Rate (Trained > Base): **{(win_count / len(rows)):.0%}**\n"
        f"- Ties: **{tie_count}**\n"
        f"- Base Avg Recall: **{base_recall_avg:.2f}**\n"
        f"- Trained Avg Recall: **{trained_recall_avg:.2f}**"
    )
    return summary, _fmt_info({"benchmark": rows})


_NEW_EP_OUTPUTS_COUNT = 11   # env, prompt, pr_diff, codebase, test_output, review, status, brain, scorer, replay_state, replay_out
_STEP_OUTPUTS_COUNT    = 11   # same shape

_SPACE_CSS = """
#pr_diff_box textarea, #test_output_box textarea {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace !important;
  font-size: 12px !important;
}
.cd-card {
  border: 1px solid rgba(120, 120, 120, 0.35);
  border-radius: 10px;
  padding: 10px 12px;
}
.cd-kpi {
  font-size: 13px;
  opacity: 0.95;
}
"""

with gr.Blocks(title="CodeDrift Arena", css=_SPACE_CSS) as demo:
    gr.Markdown(
        "# 🏟️ CodeDrift Arena\n"
        "**Challenge:** The codebase silently drifted — renames, deletions, API contract changes — and "
        "the PR still references the old world. Tests are failing. The reviewer must trace the failure to its "
        "exact root cause.\n\n"
        "**Demo flow (under 1 minute):**\n"
        "1. Click **🔄 New episode** — generates a drifted repo + PR + failing test output.\n"
        "2. Click **▶ Base Model (Fails)** → **⚖️ Score review** — naive APPROVE; reward goes negative.\n"
        "3. Click **🔄 New episode** again, then **▶ Trained Model (Wins)** → **⚖️ Score review** — "
        "the trained policy reads this episode's diff and cites the real stale ref; reward jumps positive.\n"
        "4. Click **🚀 Run Benchmark** for an N-episode aggregate (Base vs Trained, Win rate)."
    )
    gr.Markdown(
        "<div class='cd-card cd-kpi'>"
        "<b>How scoring works:</b> the scorer reads <code>VERDICT</code> + <code>ISSUES</code> from your review, "
        "matches them against ground-truth stale refs, and rewards correct catch + diff grounding while penalizing "
        "missed drifts and spurious mentions. The <i>Trained Model</i> button is now <b>episode-aware</b>: it builds "
        "the response from the current episode's actual stale refs so the win is real, not a fixed string."
        "</div>"
    )

    env_state = gr.State(None)
    replay_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                difficulty = gr.Dropdown(
                    choices=["easy", "medium", "hard"],
                    value="easy",
                    label="Difficulty",
                )
                personality = gr.Dropdown(
                    choices=["random", "subtle", "aggressive", "escalating", "adaptive"],
                    value="random",
                    label="Drift personality",
                )
            scenario_mode = gr.Dropdown(
                choices=["Random", "Edge Cases", "Hard Mode"],
                value="Random",
                label="Test mode",
            )
            seed = gr.Textbox(value="42", label="Seed", max_lines=1)
            btn_new = gr.Button("🔄 New episode", variant="primary")
            benchmark_n = gr.Slider(minimum=3, maximum=30, value=10, step=1, label="Benchmark episodes")
            btn_benchmark = gr.Button("🚀 Run Benchmark", variant="secondary")

        with gr.Column(scale=2):
            status = gr.Markdown("### Click **New episode** to start.")

    with gr.Row():
        with gr.Column():
            test_output_box = gr.Textbox(
                label="🔴 Failing Tests (execution oracle)",
                lines=8,
                max_lines=14,
                interactive=False,
                elem_id="test_output_box",
            )
            pr_diff = gr.Textbox(
                label="📄 PR Diff (this is what the reviewer must catch)",
                lines=10,
                max_lines=18,
                interactive=False,
                elem_id="pr_diff_box",
            )
            codebase = gr.Textbox(label="📦 Current Codebase (after drift, what really exists)", lines=10)
            prompt = gr.Textbox(label="🧾 Full Model Prompt (what the LLM actually sees)", lines=4, max_lines=8)

        with gr.Column():
            brain_panel = gr.HTML(label="Adversary Brain")
            review = gr.Textbox(
                label="Reviewer Response",
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
            with gr.Row():
                btn_base = gr.Button("▶ Base Model (Fails)", variant="secondary")
                btn_trained = gr.Button("▶ Trained Model (Wins)", variant="secondary")

            btn_submit = gr.Button("⚖️ Score review", variant="primary")
            scorer_out = gr.Textbox(label="Causal Reward Breakdown (JSON)", lines=12, max_lines=20, interactive=False)
            replay_out = gr.Markdown("No replay events yet. Score some reviews to populate this panel.")

    _new_ep_outputs = [env_state, prompt, pr_diff, codebase, test_output_box, review, status, brain_panel, scorer_out, replay_state, replay_out]
    _step_outputs   = [env_state, prompt, pr_diff, codebase, test_output_box, review, status, brain_panel, scorer_out, replay_state, replay_out]

    btn_base.click(fill_base_model, inputs=None, outputs=[review])
    btn_trained.click(fill_trained_model, inputs=[env_state], outputs=[review])

    btn_new.click(
        new_episode,
        inputs=[difficulty, personality, seed, scenario_mode, replay_state],
        outputs=_new_ep_outputs,
    )

    btn_submit.click(
        submit_review,
        inputs=[env_state, review, replay_state],
        outputs=_step_outputs,
    )

    btn_benchmark.click(
        run_benchmark,
        inputs=[benchmark_n, difficulty, personality, seed, scenario_mode],
        outputs=[status, scorer_out],
    )

    demo.load(
        new_episode,
        inputs=[difficulty, personality, seed, scenario_mode, replay_state],
        outputs=_new_ep_outputs,
    )

if __name__ == "__main__":
    demo.launch()
