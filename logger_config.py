"""
logger_config.py
─────────────────
Enterprise logging setup for the multi-agent orchestration system.

Strategy
--------
  CRITICAL  unrecoverable failures — agent crash, file write failure
  ERROR     recoverable errors — bad section parse, missing content
  WARNING   unexpected but handled — unknown language tag, empty section
  INFO      every lifecycle event — agent started, file written, task done
  DEBUG     full payloads — code previews, file paths (disabled in prod)

Output
------
  Console  → human-readable, coloured by level
  File     → logs/orchestration_{session_id}.log (plain text, same format)

Usage
-----
  from logger_config import setup_logger
  setup_logger(session_id="abc123")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"


def setup_logger(session_id: str = "default", level: int = logging.INFO) -> None:
    """
    Configure the root logger with:
      - StreamHandler  → stdout console
      - FileHandler    → logs/orchestration_{session_id}.log

    Call once at application startup (main.py).
    All subsequent `logging.getLogger(__name__)` calls inherit this config.
    """
    LOG_DIR.mkdir(exist_ok=True)

    log_file = LOG_DIR / f"orchestration_{session_id}.log"

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # ── Remove any existing handlers (safe for re-entrant calls) ─────────────
    root.handlers.clear()

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(level)
    root.addHandler(console_handler)

    # ── File handler ──────────────────────────────────────────────────────────
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)   # always capture DEBUG to file
    root.addHandler(file_handler)

    logging.info(
        f"Logger initialised | session={session_id} | log_file={log_file}"
    )