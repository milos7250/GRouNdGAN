import logging
import os
from contextlib import contextmanager

from rich.logging import Console, RichHandler
from tqdm import tqdm as std_tqdm


def __get_handler() -> RichHandler:
    """
    Get a custom RichHandler for logging.

    Returns
    -------
    handler : RichHandler
        Configured RichHandler instance.
    """
    return RichHandler(
        level=logging.NOTSET,
        console=Console(width=100, force_terminal=True, no_color=False),
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        log_time_format="[%X]",
    )

# Set up basic configuration for unmanaged loggers
FORMAT = "([green]%(name)s[/green]) %(message)s "
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "WARNING"),
    force=True,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[__get_handler()],
)


@contextmanager
def with_log_level(logger, level):
    """
    Context manager to temporarily set the log level of a logger.
    """
    old_level = logger.getEffectiveLevel()
    try:
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(old_level)


def setup_logger(name: str = None) -> logging.Logger:
    """
    Custom function that initializes and returns a logger with a RichHandler.

    Parameters
    ----------
    name  : str, optional
        Name of the logger. If None, the root logger is used.

    Returns
    -------
    logger : logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(os.environ.get("GROUNDGAN_LOGLEVEL", "INFO"))

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    if os.getenv("RANK", "0") == "0":
        logger.addHandler(__get_handler())
        logger.debug(f"Logger {name} initialized on rank {os.getenv('RANK', '0')}")

    return logger

@contextmanager
def tqdm_logging_redirect(loggers: list[logging.Logger] = None, tqdm_class: std_tqdm = None, *tqdm_args, **tqdm_kwargs):
    """
    Context manager to allow logging output to be printed out without interfering with `tqdm` progress bars.

    Parameters
    ----------
    loggers  : list, optional
        List of loggers to redirect. If None, defaults to [logging.root].
    tqdm_class  : optional
        The `tqdm` class to use for progress bars. If None, defaults to standard `tqdm`.
    *tqdm_args, **tqdm_kwargs :
        Additional arguments and keyword arguments to pass to the `tqdm` class.

    Yields
    ------
    pbar : tqdm_class instance
        The progress bar instance created by `tqdm_class`.
    """
    # TODO: currently does not support tqdm.rich, is only tested with standard tqdm
    if loggers is None:
        loggers = [logging.root]
    if tqdm_class is None:
        tqdm_class = std_tqdm
    try:
        with tqdm_class(*tqdm_args, **tqdm_kwargs) as pbar:
            for logger in loggers:
                for handler in logger.handlers:
                    old_emit = handler.emit

                    def new_emit(*args, **kwargs):
                        with pbar.external_write_mode():
                            old_emit(*args, **kwargs)

                    handler.old_emit = old_emit
                    handler.emit = new_emit
            yield pbar
    finally:
        for logger in loggers:
            for handler in logger.handlers:
                if handler.emit.__name__ == "new_emit":
                    handler.emit = handler.old_emit