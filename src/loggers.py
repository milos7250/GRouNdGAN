import logging
import os
from contextlib import contextmanager

from rich.logging import Console, RichHandler


def __get_handler():
    return RichHandler(
        level=logging.INFO,
        console=Console(width=116, force_terminal=True, no_color=False),
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        log_time_format="[%X]",
    )


FORMAT = "([green]%(name)s[/green]) %(message)s "
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"), format=FORMAT, datefmt="[%X]", handlers=[__get_handler()])


@contextmanager
def with_log_level(logger, level):
    old_level = logger.getEffectiveLevel()
    try:
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(old_level)


def setup_logger(name: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    if os.getenv("RANK", "0") == "0":
        logger.addHandler(__get_handler())
        logger.info(f"Logger {name} initialized on rank {os.getenv('RANK', '0')}")

    return logger
