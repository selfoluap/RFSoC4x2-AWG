"""Runs the web backend (status page, bench test, live view) inside the device server process.

Both use the same device core, so the web page always shows what the device server did.
"""

from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)


def start_web(host: str | None = None, port: int | None = None):
    """Start the FastAPI backend as a task in the running event loop.

    Returns the task, or None when the web packages are not installed (the device server then
    runs without a web page).
    """
    try:
        import uvicorn

        from backend.backend import app
    except ImportError as error:
        logger.warning("web backend not started: %s", error)
        return None
    config = uvicorn.Config(
        app,
        host=host or os.environ.get("RFSOC_BACKEND_HOST", "127.0.0.1"),
        port=port or int(os.environ.get("RFSOC_BACKEND_PORT", "8001")),
        log_level="info",
    )
    return asyncio.ensure_future(uvicorn.Server(config).serve())
