"""Process-wide logging for GUI entry points; library imports configure nothing."""

import logging
import os

from pixelmap.error_reporting import ExceptionWebhookHandler

log = logging.getLogger(__name__)

_STDERR = logging.StreamHandler()
_STDERR.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
_WEBHOOK = ExceptionWebhookHandler()


def configure_logging() -> None:
    """Install stderr and webhook handlers once, without changing root logging."""
    raw = os.environ.get("PIXELMAP_LOG_LEVEL", "INFO")
    level = logging.getLevelNamesMapping().get(raw.strip().upper())
    logger = logging.getLogger("pixelmap")
    first_install = _STDERR not in logger.handlers
    logger.setLevel(level or logging.INFO)
    # Stops records printing twice when panel serve's bokeh configures the root logger.
    logger.propagate = False
    logger.addHandler(_STDERR)
    logger.addHandler(_WEBHOOK)
    # Preserves the old basicConfig(ERROR) "no warnings" behaviour for these.
    for name in ("bokeh", "tornado"):
        logging.getLogger(name).setLevel(logging.ERROR)
    if level is None and first_install:
        log.warning("Invalid PIXELMAP_LOG_LEVEL=%r; using INFO", raw)
