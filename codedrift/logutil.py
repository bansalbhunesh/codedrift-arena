"""Lightweight stdlib logging helpers (no extra deps)."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter for machine-parsable production logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """
    Idempotent logging setup.

    Env:
      - CODEDRIFT_LOG_LEVEL  (default: INFO)
      - CODEDRIFT_LOG_FORMAT (text|json, default: text)
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.environ.get("CODEDRIFT_LOG_LEVEL", "INFO")).upper()
    numeric = getattr(logging, lvl, logging.INFO)
    fmt = os.environ.get("CODEDRIFT_LOG_FORMAT", "text").strip().lower()
    if fmt == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        root = logging.getLogger()
        root.setLevel(numeric)
        root.handlers.clear()
        root.addHandler(handler)
    else:
        logging.basicConfig(
            level=numeric,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
