"""Process-wide logging for GUI entry points; library imports configure nothing."""

import logging
import os
import threading

from pixelmap.error_reporting import ExceptionWebhookHandler

log = logging.getLogger(__name__)
_lock = threading.Lock()
_warned_values: set[str] = set()


def configure_logging() -> None:
    """Install stderr and webhook handlers once, without changing root logging."""
    raw = os.environ.get("PIXELMAP_LOG_LEVEL", "INFO")
    normalized = raw.strip().upper()
    level = logging.getLevelNamesMapping().get(normalized)
    if level is None and normalized.isdecimal():
        level = int(normalized)
    invalid = level is None
    if invalid:
        level = logging.INFO

    with _lock:
        logger = logging.getLogger("pixelmap")
        logger.setLevel(level)
        logger.propagate = False
        if not any(getattr(h, "_pixelmap_stderr", False) for h in logger.handlers):
            handler = logging.StreamHandler()
            handler._pixelmap_stderr = True
            handler.setFormatter(logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s"
            ))
            logger.addHandler(handler)
        if not any(isinstance(h, ExceptionWebhookHandler) for h in logger.handlers):
            logger.addHandler(ExceptionWebhookHandler())
        for name in ("bokeh", "tornado"):
            logging.getLogger(name).setLevel(logging.ERROR)
        if invalid and raw not in _warned_values:
            _warned_values.add(raw)
            log.warning("Invalid PIXELMAP_LOG_LEVEL=%r; using INFO", raw)
