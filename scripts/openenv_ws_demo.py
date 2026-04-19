#!/usr/bin/env python3
"""
Hackathon-friendly demo: one episode over **OpenEnv WebSocket** (persistent session).

Why WebSocket: stateless ``POST /reset`` + ``POST /step`` each create a **new**
environment in openenv-core, so ``step`` cannot follow ``reset`` over plain HTTP.
``/ws`` keeps one ``CodeDriftOpenEnvironment`` for the connection.

Prerequisites:
  pip install -r requirements-server.txt

Terminal 1 (repo root)::

  uvicorn server.app:app --host 0.0.0.0 --port 8000

Terminal 2::

  python scripts/openenv_ws_demo.py

Environment:
  CODEDRIFT_OPENENV_WS  WebSocket URL (default ``ws://127.0.0.1:8000/ws``)

curl (health / metadata only; full episode use WebSocket or in-process)::

  curl -s http://127.0.0.1:8000/health
  curl -s http://127.0.0.1:8000/metadata
  curl -s -X POST http://127.0.0.1:8000/reset -H "Content-Type: application/json" -d "{}"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_WS = os.environ.get("CODEDRIFT_OPENENV_WS", "ws://127.0.0.1:8000/ws")


def _pick_scorer_info(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize observation payload from WS ``WSObservationResponse``."""
    data = payload.get("data") or {}
    obs = data.get("observation")
    if isinstance(obs, dict):
        si = obs.get("scorer_info")
        if isinstance(si, dict):
            return si
    return {}


async def run_episode(ws_url: str, seed: int) -> None:
    try:
        import websockets
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: websockets. Install with: pip install websockets"
        ) from e

    reset_msg = json.dumps({"type": "reset", "data": {"seed": seed}})
    step_msg = json.dumps(
        {
            "type": "step",
            "data": {
                "metadata": {
                    "agent_response": (
                        "VERDICT: REQUEST_CHANGES\n"
                        "ISSUES: stale API references in diff\n"
                        "REASON: Demo review text (tune to episode ground truth).\n"
                    )
                }
            },
        }
    )

    async with websockets.connect(ws_url) as ws:
        await ws.send(reset_msg)
        raw_r = await ws.recv()
        reset_payload = json.loads(raw_r)
        print("--- reset ---")
        print(json.dumps(reset_payload, indent=2)[:4000])

        await ws.send(step_msg)
        raw_s = await ws.recv()
        step_payload = json.loads(raw_s)
        print("--- step ---")
        print(json.dumps(step_payload, indent=2)[:4000])

        data = step_payload.get("data") or {}
        reward = data.get("reward")
        done = data.get("done")
        info = _pick_scorer_info(step_payload)
        print(
            "\nSummary:",
            f"reward={reward!r}",
            f"done={done!r}",
            f"episode_outcome={info.get('episode_outcome')!r}",
            f"verdict={info.get('verdict')!r}",
            f"metric_strip={info.get('metric_strip')!r}",
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ws", default=DEFAULT_WS, help="WebSocket URL (default from CODEDRIFT_OPENENV_WS)")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    asyncio.run(run_episode(args.ws, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
