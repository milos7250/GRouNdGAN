#!/usr/bin/env python
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import rich_click as click
except ImportError:
    import click

logger = logging.getLogger(__name__)


def _format_count(count: int) -> str:
    """Format a parameter count using compact terminal-friendly units."""
    for divisor, suffix in ((10**12, "T"), (10**9, "G"), (10**6, "M"), (10**3, "k")):
        if count >= divisor:
            value = f"{count / divisor:.2f}".rstrip("0").rstrip(".")
            return f"{value}{suffix}"
    return f"{count:,}"


def _count_parameters(value: Any) -> int:
    """Return the number of scalar values in tensors contained in ``value``."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.numel()
    if isinstance(value, Mapping):
        return sum(_count_parameters(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_parameters(item) for item in value)
    if isinstance(value, (int, float, bool)):
        return 1
    if value is None:
        return 0
    logger.warning("Ignoring non-tensor value %s of type %s", value, type(value).__name__)
    return 0


def main(checkpoint_path: Path, output_path: Path):
    """Count tensor parameters in each top-level checkpoint entry and save JSON."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading checkpoint from '%s' on %s", checkpoint_path, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise click.ClickException("The checkpoint must contain a dictionary at its top level.")

    parameter_counts = {str(name): _count_parameters(value) for name, value in checkpoint.items()}
    name_width = max((len(name) for name in parameter_counts), default=0)
    count_strings = {name: _format_count(count) for name, count in parameter_counts.items()}
    count_width = max((len(count) for count in count_strings.values()), default=0)
    table = "\n".join(
        f"{name:<{name_width}}  {count_strings[name]:>{count_width}} parameters" for name in parameter_counts
    )
    logger.info("Parameter counts:\n%s", table)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(parameter_counts, indent=2) + "\n")
    logger.info("Successfully saved parameter counts to '%s'", output_path)


@click.command()
@click.option(
    "--checkpoint",
    "checkpoint_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--out",
    "output_path",
    required=True,
    type=click.Path(writable=True, dir_okay=False, path_type=Path),
)
def cli(checkpoint_path: Path, output_path: Path):
    main(checkpoint_path, output_path)


if __name__ == "__main__":
    try:
        from rich.logging import RichHandler

        FORMAT = "%(message)s"
        logging.basicConfig(
            level="INFO",
            format=FORMAT,
            handlers=[RichHandler(rich_tracebacks=True, tracebacks_show_locals=True)],
        )
    except ImportError:
        FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="%H:%M:%S")

    cli()
