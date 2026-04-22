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
ISSUES: none
REASON: The code looks clean and follows existing patterns. Imports are correct."""

TRAINED_MODEL_RESPONSE = """VERDICT: REQUEST_CHANGES
ISSUES: getUserData is no longer defined in the current codebase. It was renamed to fetchUserData. Line calling getUserData(user_id) will raise a NameError at runtime.
REASON: The PR references a stale function name. Must be updated to fetchUserData before merging."""


def new_episode(difficulty: str, personality: str, seed: str) -> tuple:
    try:
        s = int(seed)
    except ValueError:
        s = 42
    try:
        env = CodeDriftEnv(difficulty=difficulty, personality=personality, seed=s)
        obs = env.reset()
        status = (
            f"### 🏁 Episode started: `{env.episode_id}`\n"
            f"Ground truth stale refs: **{obs.n_stale_refs}** (Hidden from agent)"
        )
        return (
            env,
            obs.prompt,
            obs.pr_diff,
            obs.codebase_context,
            "",
            status,
            _adversary_brain_html(env),
            _fmt_info({"note": "Submit a review to see scorer output.", "episode_id": env.episode_id}),
        )
    except Exception as e:
        err = {"error": str(e), "type": type(e).__name__}
        return (
            None,
            "",
            "",
            "",
            "",
            f"### ❌ Failed to start episode: {e!s}",
            _adversary_brain_html(None),
            _fmt_info(err),
        )


def submit_review(env: CodeDriftEnv | None, review: str) -> tuple:
    if env is None:
        return (
            None,
            "",
            "",
            "",
            "",
            "### ⚠️ No active episode. Click **New episode** first.",
            _adversary_brain_html(None),
            _fmt_info({"error": "no_env"}),
        )
    if not review.strip():
        return (
            env,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "### ⚠️ Paste a non-empty review.",
            gr.update(),
            gr.update(),
        )
    if not env.is_ready_for_step:
        return (
            env,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            "### ⚠️ Episode already scored.\nClick **New episode** to try again.",
            _adversary_brain_html(env),
            _fmt_info(
                {
                    "error": "episode_already_scored",
                    "hint": "One step per episode — use New episode for another try.",
                }
            ),
        )
    try:
        _, reward, _done, info = env.step(review)
        status = _status_lines(reward, info)
        info["adversary_brain"] = env.drift_agent.adaptive_snapshot()
        return env, gr.update(), gr.update(), gr.update(), "", status, _adversary_brain_html(env), _fmt_info(info)
    except Exception as e:
        snap = env.debug_snapshot() if env is not None else {}
        err = {"error": str(e), "type": type(e).__name__, "env": snap}
        return (
            env,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            f"### ❌ Scoring failed: {e!s}",
            _adversary_brain_html(env),
            _fmt_info(err),
        )


with gr.Blocks(title="CodeDrift Arena") as demo:
    gr.Markdown(
        "# 🏟️ CodeDrift Arena\n"
        "**The Challenge:** Today's codebase has changed, but the PR still assumes yesterday's schema. "
        "The reviewer must catch these 'stale' references before they ship and break production.\n\n"
        "**Judge Path:**\n"
        "1. Click **New episode** to generate a drifted codebase + PR.\n"
        "2. Click **▶ Load Base Model** to see how a naive model fails (ships bugs).\n"
        "3. Click **▶ Load Trained Model** to see how our GRPO-trained policy catches it.\n"
        "4. Click **⚖️ Score review** to see the deterministic reward."
    )

    env_state = gr.State(None)

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
            seed = gr.Textbox(value="42", label="Seed", max_lines=1)
            btn_new = gr.Button("🔄 New episode", variant="primary")
        
        with gr.Column(scale=2):
            status = gr.Markdown("### Click **New episode** to start.")

    with gr.Row():
        with gr.Column():
            pr_diff = gr.Textbox(
                label="PR Diff (Review Target)",
                lines=12,
                max_lines=20,
                interactive=False,
                elem_id="pr_diff_box",
            )
            codebase = gr.Textbox(label="Current Codebase State (Reality)", lines=12)
            prompt = gr.Textbox(label="Full Model Prompt (Context)", lines=5, max_lines=10)
        
        with gr.Column():
            brain_panel = gr.HTML(label="Adversary Brain")
            review = gr.Textbox(
                label="Reviewer Response (ISSUES: must cite stale refs)",
                lines=10,
                placeholder="VERDICT: REQUEST_CHANGES\nISSUES: ...\nREASON: ...",
            )
            with gr.Row():
                btn_base = gr.Button("▶ Load Base Model (Fails)", variant="secondary")
                btn_trained = gr.Button("▶ Load Trained Model (Success)", variant="secondary")
            
            btn_submit = gr.Button("⚖️ Score review", variant="primary")
            scorer_out = gr.Textbox(label="Metric Breakdown (JSON)", lines=12, max_lines=20, interactive=False)

    btn_base.click(lambda: BASE_MODEL_RESPONSE, outputs=[review])
    btn_trained.click(lambda: TRAINED_MODEL_RESPONSE, outputs=[review])

    btn_new.click(
        new_episode,
        inputs=[difficulty, personality, seed],
        outputs=[env_state, prompt, pr_diff, codebase, review, status, brain_panel, scorer_out],
    )

    btn_submit.click(
        submit_review,
        inputs=[env_state, review],
        outputs=[env_state, prompt, pr_diff, codebase, review, status, brain_panel, scorer_out],
    )

    demo.load(
        new_episode,
        inputs=[difficulty, personality, seed],
        outputs=[env_state, prompt, pr_diff, codebase, review, status, brain_panel, scorer_out],
    )

if __name__ == "__main__":
    demo.launch()
