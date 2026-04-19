"""
Reward curve plotter — generates the training improvement graph for your pitch.

Run after training:
  python utils/plot_curve.py --log_dir ./codedrift_output

Synthetic demo curve:
  python utils/plot_curve.py --demo
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def plot_demo_curve(output_path: str = "reward_curve.png"):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Run: pip install matplotlib numpy")
        return

    np.random.seed(42)
    steps = np.arange(0, 200, 10)

    base = -0.8
    trend = (steps / 200) * 1.5
    noise = np.random.normal(0, 0.15, len(steps))
    rewards = np.clip(base + trend + noise, -1.0, 1.0)

    window = 3
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
    smooth_steps = steps[window - 1 :]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, rewards, alpha=0.3, color="#7F77DD", linewidth=1, label="Episode reward")
    ax.plot(smooth_steps, smoothed, color="#7F77DD", linewidth=2.5, label="Rolling avg (3 ep)")
    ax.axhline(y=0, color="#888780", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axhline(y=-1.0, color="#E24B4A", linewidth=0.8, linestyle=":", alpha=0.4, label="Miss penalty (-1.0)")
    ax.axhline(y=1.0, color="#1D9E75", linewidth=0.8, linestyle=":", alpha=0.4, label="Catch reward (+1.0)")

    ax.set_xlabel("Training steps", fontsize=12)
    ax.set_ylabel("Episode reward", fontsize=12)
    ax.set_title("CodeDrift Arena — GRPO Training Progress", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_ylim(-1.2, 1.2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved reward curve to {output_path}")
    plt.close()


def plot_from_logs(log_dir: str, output_path: str = "reward_curve.png"):
    log_file = Path(log_dir) / "trainer_state.json"
    if not log_file.exists():
        print(f"No trainer_state.json found in {log_dir}. Use --demo for a synthetic curve.")
        return

    with open(log_file) as f:
        state = json.load(f)

    history = state.get("log_history", [])
    steps = [h["step"] for h in history if "rewards/mean" in h]
    rewards = [h["rewards/mean"] for h in history if "rewards/mean" in h]

    if not steps:
        print("No reward logs found. Check TRL logging config.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Run: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, rewards, color="#7F77DD", linewidth=2)
    ax.axhline(y=0, color="#888780", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("CodeDrift Arena — Training Progress")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="Generate synthetic demo curve")
    p.add_argument("--log_dir", default="./codedrift_output")
    p.add_argument("--output", default="reward_curve.png")
    args = p.parse_args()

    if args.demo:
        plot_demo_curve(args.output)
    else:
        plot_from_logs(args.log_dir, args.output)
