#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))  # Allow importing from the project root

from grn_accessor import GRNAccessor

from loggers import setup_logger

logger = setup_logger(Path(__file__).stem)

# Allow running this script directly to filter a GRN CSV file
if __name__ == "__main__":
    try:
        import rich_click as click
    except ImportError:
        import click

    @click.group(
        context_settings={"show_default": True, "help_option_names": ["-h", "--help"]}
    )
    @click.option("--input", "-i", type=click.Path(exists=True), required=True, help="Input GRN CSV file.")
    @click.option("--output", "-o", type=click.Path(), required=True, help="Output filtered bipartite GRN CSV file.")
    @click.option("--tf-col", type=str, default="TF", help="Column name for TFs in the input CSV.")
    @click.option("--target-col", type=str, default="target", help="Column name for targets in the input CSV.")
    @click.option(
        "--importance-col",
        type=str,
        default="importance",
        help="Column name for importance in the input CSV. If not present, set to empty string "
        "and the script will use the order of edges as importance.",
    )
    @click.pass_context
    def cli(ctx: click.Context, input: Path, output: Path, tf_col: str, target_col: str, importance_col: str) -> None:
        """This script processes a GRN CSV file to either filter it into a bipartite graph or convert it into an undirected graph."""
        col_names = {"TF": tf_col, "target": target_col}
        if importance_col != "":
            col_names["importance"] = importance_col
        grn: GRNAccessor = GRNAccessor.from_csv(input, col_names=col_names).grn  # pyright: ignore[reportAssignmentType]
        ctx.obj = grn

    @cli.result_callback()
    def save_result(result: pd.DataFrame, output: Path, *args: tuple[Any], **kwargs: dict[str, Any]) -> None:
        result.to_csv(output, index=False)
        logger.info(f"Saved processed GRN to {output}")

    @cli.command("to-bipartite")
    @click.pass_context
    def to_bipartite(ctx: click.Context) -> pd.DataFrame:
        """Filter the GRN into a bipartite graph based on importance of edges."""
        grn: GRNAccessor = ctx.obj
        return grn.to_bipartite()

    @cli.command("to-undirected")
    @click.pass_context
    def to_undirected(ctx: click.Context) -> pd.DataFrame:
        """Convert the GRN into an undirected graph by symmetrizing the edges."""
        grn: GRNAccessor = ctx.obj
        return grn.to_undirected()

    cli()
