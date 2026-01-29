#!/usr/bin/env python3
from pathlib import Path
from typing import Any

import pandas as pd

from loggers import setup_logger

from .grn_accessor import GRNAccessor

logger = setup_logger(__name__)

# Allow running this script directly to filter a GRN CSV file
if __name__ == "__main__":
    try:
        import rich_click as click
    except ImportError:
        import click

    @click.group()
    @click.option("--input", "-i", type=click.Path(exists=True), required=True, help="Input GRN CSV file")
    @click.option("--output", "-o", type=click.Path(), required=True, help="Output filtered bipartite GRN CSV file")
    @click.option("--tf-col", type=str, default="TF", help="Column name for TFs in the input CSV")
    @click.option("--target-col", type=str, default="target", help="Column name for targets in the input CSV")
    @click.option("--importance-col", type=str, default=None, help="Column name for importance in the input CSV")
    @click.pass_context
    def cli(ctx: click.Context, input: Path, output: Path, tf_col: str, target_col: str, importance_col: str | None) -> None:
        """This script processes a GRN CSV file to either filter it into a bipartite graph or convert it into an undirected graph."""
        col_names = {"TF": tf_col, "target": target_col}
        if importance_col is not None:
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
        filtered_grn = grn.to_bipartite()
        return filtered_grn
    
    @cli.command("to-undirected")
    @click.pass_context
    def to_undirected(ctx: click.Context) -> pd.DataFrame:
        """Convert the GRN into an undirected graph."""
        grn: GRNAccessor = ctx.obj
        undirected_grn = grn.to_undirected()
        return undirected_grn

    cli()
