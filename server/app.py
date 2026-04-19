"""
OpenEnv FastAPI entrypoint for CodeDrift Arena.

Run:
  uvicorn server.app:app --host 0.0.0.0 --port 8000

Requires: pip install openenv-core uvicorn fastapi
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from integrations.codedrift_openenv import OPENENV_AVAILABLE, build_openenv_app

if not OPENENV_AVAILABLE:
    raise SystemExit(
        "openenv-core is not installed. Run: pip install 'openenv-core' uvicorn fastapi"
    )

app = build_openenv_app()
