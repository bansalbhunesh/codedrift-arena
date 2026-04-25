"""JSONL metrics logger with optional W&B passthrough.

Used by ``training_v2.train_v2`` and the generalization eval. Always writes
to a local JSONL file so we have evidence for judges even when W&B is off.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class MetricsLogger:
    output_path: Path
    use_wandb: bool = False
    wandb_run: Any = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.use_wandb and self.wandb_run is None:
            try:
                import wandb  # type: ignore

                self.wandb_run = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "codedrift-arena-v2"),
                    config=self.extra,
                    reinit=True,
                )
            except Exception:
                self.wandb_run = None

    def log(self, step: int, payload: dict[str, Any]) -> None:
        record = {"step": int(step), "ts": time.time(), **payload}
        with self.output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=_json_default) + "\n")
        if self.wandb_run is not None:
            try:
                self.wandb_run.log({k: v for k, v in payload.items() if _wandb_safe(v)}, step=step)
            except Exception:
                pass

    def close(self) -> None:
        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception:
                pass


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
    return str(value)


def _wandb_safe(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool)) or value is None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
