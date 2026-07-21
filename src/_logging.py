"""
_logging.py — Centralised logging setup for the water-quality pipeline.

Call setup_logging() once at application startup (or import this module in
your entry-point script). All other src/ modules use logging.getLogger(__name__)
— they never call setup_logging() themselves.

Log routing
-----------
  Console : INFO and above (human-readable progress during normal runs).
  File    : WARNING and above only (keeps logs/system.log focused on issues).

The log file is created lazily when the first WARNING is emitted. The parent
directory is created automatically.

If setup_logging() is never called, Python's default logging config is used
(WARNING to stderr only — safe for library usage).
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_SETUP_DONE = False


def setup_logging(
    log_file:      str | Path | None = None,
    console_level: str               = "INFO",
    file_level:    str               = "WARNING",
    fmt:           str               = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    date_fmt:      str               = "%Y-%m-%d %H:%M:%S",
) -> None:
    """
    Configure the root logger with a console handler and a rotating file handler.

    Parameters
    ----------
    log_file      : Path to the log file. None = no file handler.
                    Default resolved from system_config.json if not given.
    console_level : Minimum level for console output (default "INFO").
    file_level    : Minimum level for file output (default "WARNING").
    fmt           : Log record format string.
    date_fmt      : Datetime format for the %(asctime)s placeholder.
    """
    global _SETUP_DONE
    if _SETUP_DONE:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)   # let handlers filter independently

    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)

    # ── Console handler ────────────────────────────────────────────────────────
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # ── File handler ───────────────────────────────────────────────────────────
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes    = 10 * 1024 * 1024,   # 10 MB
            backupCount = 5,
            encoding    = "utf-8",
        )
        fh.setLevel(getattr(logging, file_level.upper(), logging.WARNING))
        fh.setFormatter(formatter)
        root.addHandler(fh)

    _SETUP_DONE = True


def setup_logging_from_config(config_path: str | Path | None = None) -> None:
    """
    Convenience wrapper: reads logging params from system_config.json.

    Imports src.config lazily so _logging.py can itself be imported early
    without circular-import issues.
    """
    from src.config import get_config
    cfg = get_config(config_path)
    lc  = cfg.logging
    setup_logging(
        log_file      = lc.log_file,
        console_level = lc.console_level,
        file_level    = lc.file_level,
        fmt           = lc.format,
        date_fmt      = lc.date_format,
    )
