import logging
import os
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

from matplotlib import pyplot as plt
from rich.console import Console
from rich.logging import RichHandler

if TYPE_CHECKING:
    from collections.abc import Iterator
    from logging import Logger, LogRecord
    from typing import Any

    from tqdm import tqdm as std_tqdm

FORMAT = "([green]%(name)s[/green]) %(message)s "


def __get_handler(rank: str | None = None) -> RichHandler:
    """
    Get a custom RichHandler for logging.

    Parameters
    ----------
    rank : str | None, optional
        Rank of the process in distributed training. If None, no rank is displayed.

    Returns
    -------
    handler : RichHandler
        Configured RichHandler instance.
    """
    handler = RichHandler(
        level=logging.NOTSET,
        # console=Console(width=200, force_terminal=True, no_color=False),
        console=Console(no_color=False),
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
        log_time_format="[%X]",
    )
    handler.formatter = logging.Formatter(
        (rf"[orange1]\[RANK {rank}][/orange1] " if rank else "") + FORMAT, datefmt="[%X]"
    )
    return handler


@contextmanager
def with_log_level(logger: "Logger", level: int) -> "Iterator[None]":
    """
    Context manager to temporarily set the log level of a logger.

    Parameters
    ----------
    logger : Logger
        The logger whose level is to be set.
    level : int
        The log level to set temporarily.
    """
    old_level = logger.getEffectiveLevel()
    try:
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(old_level)


def setup_logger(name: str | None = None) -> "Logger":
    """
    Custom function that initializes and returns a logger with a RichHandler.

    Parameters
    ----------
    name  : str, optional
        Name of the logger. If None, the root logger is used.

    Returns
    -------
    logger : Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    logger.propagate = False
    logger.setLevel(os.environ.get("GROUNDGAN_LOGLEVEL", "INFO"))

    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    if os.environ.get("RANK", "0") == "0" or logger.level <= logging.DEBUG:
        logger.addHandler(__get_handler(os.environ.get("RANK", None)))
        logger.debug(f"Logger {name} initialized.")

    return logger


@contextmanager
def tqdm_logging_redirect(
    loggers: "list[Logger] | None" = None,
    tqdm_class: "type[std_tqdm[Any]] | None" = None,
    *tqdm_args: "Any",
    **tqdm_kwargs: "Any",
) -> "Iterator[std_tqdm[Any]]":
    """
    Context manager to allow logging output to be printed out without interfering with `tqdm` progress bars.

    Parameters
    ----------
    loggers  : list[Logger], optional
        List of loggers to redirect. If None, defaults to [logging.root].
    tqdm_class  : type[std_tqdm], optional
        The `tqdm` class to use for progress bars. If None, defaults to standard `tqdm`.
    *tqdm_args: Any, **tqdm_kwargs: Any
        Additional arguments and keyword arguments to pass to the `tqdm` class.

    Yields
    ------
    pbar : tqdm_class[Any] instance
        The progress bar instance created by `tqdm_class`.
    """
    # TODO: currently does not support tqdm.rich, is only tested with standard tqdm
    if loggers is None:
        loggers = [logging.root]
    if tqdm_class is None:
        tqdm_class = std_tqdm
    try:
        tqdm_kwargs["disable"] = os.environ.get("RANK", "0") != "0" or os.environ.get("GROUNDGAN_NO_TQDM", "0") == "1"
        with tqdm_class(
            *tqdm_args,
            **tqdm_kwargs,
        ) as pbar:
            for logger in loggers:
                for handler in logger.handlers:
                    old_emit = handler.emit

                    def new_emit(record: "LogRecord") -> None:
                        with pbar.external_write_mode():
                            old_emit(record)

                    handler.old_emit = old_emit  # type: ignore
                    handler.emit = new_emit
            yield pbar
    finally:
        for logger in loggers:
            for handler in logger.handlers:
                if handler.emit.__name__ == "new_emit":
                    handler.emit = handler.old_emit  # type: ignore


# Set up basic configuration for unmanaged loggers
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "WARNING"),
    force=True,
    format=FORMAT,
    datefmt="[%X]",
    handlers=[__get_handler()],
)

# Set up specific loggers
setup_logger("optuna")

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*UserWarning: pkg_resources is deprecated as an API.*")
warnings.filterwarnings("ignore", message=".*GPSampler is experimental.*")
warnings.filterwarnings("ignore", message=".*dynamo_pgo force disabled by torch.compiler.config.force_disable_caches*")
warnings.filterwarnings("ignore", message=".*Using an existing study with name .* instead of creating a new one.*")
warnings.filterwarnings("ignore", message=".*Trial [0-9]+ pruned.*")

# Setup plotting variables
plt.rcParams.update({
    "savefig.facecolor": (0.0, 0.0, 0.0, 0.0),
    "axes.facecolor": (0.0, 0.0, 0.0, 0.0),
    "legend.facecolor": (0.0, 0.0, 0.0, 0.1),
    "legend.framealpha": 0.1,
    "savefig.transparent": True,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0,
    "savefig.dpi": 300,
})
