"""Matplotlib plots for V2 reward curves and per-pattern accuracy bars.

Headless-safe (Agg backend). Reads JSONL produced by :mod:`utils_v2.metrics`.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils_v2.metrics import load_jsonl  # noqa: E402


def plot_reward_curve(rows: Iterable[dict], out_path: Path) -> Path:
    rows = list(rows)
    if not rows:
        raise ValueError("no rows to plot")
    steps = [int(r.get("step", i)) for i, r in enumerate(rows)]
    total = [float(r.get("reward_total", r.get("reward", 0.0))) for r in rows]
    rc = [float(r.get("reward_root_cause", 0.0)) for r in rows]
    fp = [float(r.get("reward_failure_path", 0.0)) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, total, label="total", linewidth=2)
    ax.plot(steps, rc, label="root_cause", linewidth=1, alpha=0.7)
    ax.plot(steps, fp, label="failure_path", linewidth=1, alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.set_title("V2 reward components over training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_per_pattern_accuracy(rows: Iterable[dict], out_path: Path) -> Path:
    rows = list(rows)
    if not rows:
        raise ValueError("no rows to plot")
    counts: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        pat = str(r.get("pattern", "unknown"))
        counts[pat].append(float(r.get("root_cause_score", 0.0)))
    means = {p: (sum(v) / len(v) if v else 0.0) for p, v in counts.items()}
    patterns = sorted(means.keys())
    values = [means[p] for p in patterns]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(patterns, values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean root_cause_score")
    ax.set_title("V2 per-pattern accuracy")
    ax.tick_params(axis="x", rotation=30)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    p = argparse.ArgumentParser(description="Plot V2 metrics from JSONL.")
    p.add_argument("--input", required=True, help="JSONL file written by MetricsLogger")
    p.add_argument("--out_curve", default="outputs/v2_reward_curve.png")
    p.add_argument("--out_bars", default="outputs/v2_per_pattern.png")
    args = p.parse_args()
    rows = load_jsonl(Path(args.input))
    cp = plot_reward_curve(rows, Path(args.out_curve))
    bp = plot_per_pattern_accuracy(rows, Path(args.out_bars))
    print(f"wrote {cp}\nwrote {bp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
