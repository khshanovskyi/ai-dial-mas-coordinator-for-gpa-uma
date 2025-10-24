import logging
import sys
from typing import Optional


def setup_logging(
        log_level: str = "INFO",
        log_format: Optional[str] = None,
        include_timestamp: bool = True
) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Custom log format string. If None, uses default format.
        include_timestamp: Whether to include timestamp in log messages
    """
    if log_format is None:
        if include_timestamp:
            log_format = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "[%(filename)s:%(lineno)d] - %(message)s"
            )
        else:
            log_format = (
                "%(name)s - %(levelname)s - "
                "[%(filename)s:%(lineno)d] - %(message)s"
            )

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)