"""Structured JSON logging via structlog (Phase 1.6).

Usage:
    from app.core.logging import get_logger
    log = get_logger(__name__)
    log.info("planner_done", sub_questions=4)

In routes.py, bind the per-request request_id at the top of the stream:
    bind_request_context(request_id=request_id)
All log lines emitted inside that coroutine then carry the request_id,
making a single run's logs filterable end to end.
"""
import logging
import sys

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent structlog configuration with JSON output."""
    if getattr(configure_logging, "_configured", False):
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (used by existing agent modules) through the same
    # JSON renderer so all pipeline logs share one format.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(message)s")
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    configure_logging._configured = True


def get_logger(name: str):
    configure_logging()
    return structlog.get_logger(name)


def bind_request_context(**bindings) -> None:
    """Bind per-request context (e.g. request_id) into every subsequent log line."""
    structlog.contextvars.bind_contextvars(**bindings)


def unbind_request_context() -> None:
    structlog.contextvars.unbind_contextvars("request_id")
