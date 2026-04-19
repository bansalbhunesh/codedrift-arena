"""Lightweight stdlib logging helpers (no extra deps)."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Idempotent basicConfig; reads CODEDRIFT_LOG_LEVEL if level is None."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.environ.get("CODEDRIFT_LOG_LEVEL", "INFO")).upper()
    numeric = getattr(logging, lvl, logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(levelname)s %(name)s: %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
