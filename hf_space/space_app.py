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


def new_episode(difficulty: str, personality: str, seed: str) -> tuple:
    try:
        s = int(seed)
    except ValueError:
        s = 42
    try:
        env = CodeDriftEnv(difficulty=difficulty, personality=personality, seed=s)
        obs = env.reset()
        status = (
            f"Episode started id={env.episode_id} "
            f"(n_stale_refs shown to trainer only: {obs.n_stale_refs})."
        )
        return (
            env,
            obs.prompt,
            obs.pr_diff,
            obs.codebase_context,
            "",
            status,
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
            f"Failed to start episode: {e!s}",
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
            "Reset an episode first.",
            _fmt_info({"error": "no_env"}),
        )
    if not review.strip():
        return env, gr.update(), gr.update(), gr.update(), gr.update(), "Paste a non-empty review.", gr.update()
    try:
        _, reward, _done, info = env.step(review)
        status = f"Step complete (episode {info.get('episode_id', '')}). Reward: {reward:+.2f}"
        return env, gr.update(), gr.update(), gr.update(), "", status, _fmt_info(info)
    except Exception as e:
        snap = env.debug_snapshot() if env is not None else {}
        err = {"error": str(e), "type": type(e).__name__, "env": snap}
        return (
            env,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            f"Scoring failed: {e!s}",
            _fmt_info(err),
        )


with gr.Blocks(title="CodeDrift Arena") as demo:
    gr.Markdown(
        "## CodeDrift Arena\n"
        "Trainable **code reviewer** vs frozen **drift** on a synthetic repo. "
        "This Space runs the **environment + reward** on CPU (no LLM weights). "
        "Paste any review text and see how `RewardScorer` grades it.\n\n"
        "_Tip: use `VERDICT: APPROVE` or `VERDICT: REQUEST_CHANGES` plus mention stale symbols in `ISSUES`._\n\n"
        "_Rubric: the scorer grades **only** the `ISSUES:` body and explicit `VERDICT:` "
        "(it does not parse the PR diff text for rewards—evidence must appear under `ISSUES:`)._"
    )

    env_state = gr.State(None)

    with gr.Row():
        difficulty = gr.Dropdown(
            choices=["easy", "medium", "hard"],
            value="easy",
            label="Difficulty",
        )
        personality = gr.Dropdown(
            choices=["random", "subtle", "aggressive", "escalating"],
            value="random",
            label="Drift personality",
        )
        seed = gr.Textbox(value="42", label="Seed", max_lines=1)

    btn_new = gr.Button("New episode", variant="primary")
    status = gr.Textbox(label="Status", lines=2, interactive=False)

    prompt = gr.Textbox(label="Prompt (for your LLM)", lines=12, max_lines=40)
    pr_diff = gr.Textbox(label="PR diff", lines=10, max_lines=30)
    codebase = gr.Textbox(label="Current codebase (formatted)", lines=10, max_lines=30)

    review = gr.Textbox(
        label="Reviewer output (paste model completion here)",
        lines=8,
        placeholder="VERDICT: REQUEST_CHANGES\nISSUES: ...\nREASON: ...",
    )
    btn_submit = gr.Button("Score review")

    scorer_out = gr.Textbox(label="Scorer breakdown", lines=16, max_lines=40, interactive=False)

    btn_new.click(
        new_episode,
        inputs=[difficulty, personality, seed],
        outputs=[env_state, prompt, pr_diff, codebase, review, status, scorer_out],
    )

    btn_submit.click(
        submit_review,
        inputs=[env_state, review],
        outputs=[env_state, prompt, pr_diff, codebase, review, status, scorer_out],
    )

    demo.load(
        new_episode,
        inputs=[difficulty, personality, seed],
        outputs=[env_state, prompt, pr_diff, codebase, review, status, scorer_out],
    )
