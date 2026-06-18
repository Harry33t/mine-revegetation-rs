"""Minimal structured logging. `import logging` here resolves to the stdlib
(absolute imports in Py3), not this module."""

from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_configured: set[str] = set()


def get_logger(name: str = "nz_canopy") -> logging.Logger:
    logger = logging.getLogger(name)
    if name not in _configured:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("NZC_LOG_LEVEL", "INFO").upper())
        logger.propagate = False
        _configured.add(name)
    return logger
