"""Top level package for the video generation workflow project.

This module configures a default logger so that informational messages emitted
by modules such as ``video_gen.api.server`` appear on the console when the
application is started via ``uvicorn`` or other entrypoints.  The configuration
is intentionally lightweight and only runs when the ``video_gen`` logger does
not yet have any handlers, allowing downstream applications to override the
logging setup if desired.
"""

from __future__ import annotations

import logging

DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def _configure_default_logging() -> None:
    """Ensure the package has a console logger for informational messages."""

    logger = logging.getLogger("video_gen")
    if logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
    logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False


_configure_default_logging()

__all__ = ["DEFAULT_LOG_FORMAT"]
