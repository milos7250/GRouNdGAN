#!/usr/bin/env python
import logging
from pathlib import Path

try:
    import rich_click as click
except ImportError:
    import click

logger = logging.getLogger(__name__)


def main(adata_path: Path, output_path: Path, tfs_path: Path | None = None):
    """
    Infers a GRN using GRNBoost2 and saves it to a CSV file.

    Parameters
    ----------
    adata_path : Path
        Path to the input AnnData file.
    output_path : Path
        Path to save the inferred GRN CSV file.
    tfs_path : Path, optional
        Path to a TSV file containing transcription factors (TFs). If not provided, all genes will be used as TFs.
    """

    import anndata as ad
    import numpy as np
    import pandas as pd
    from arboreto.algo import grnboost2
    from scipy import sparse

    adata = ad.read_h5ad(adata_path)
    x = adata.X.toarray() if sparse.issparse(adata.X) else adata.X  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
    X = np.array(x)
    real_cells_df = pd.DataFrame(X, columns=adata.var_names)

    # Load TFs if provided
    if tfs_path:
        TFs = pd.read_csv(tfs_path, sep="\t")["Symbol"]
        TFs = [tf for tf in TFs if tf in adata.var_names]
        if not TFs:
            raise ValueError("No TFs from the provided list were found in the dataset. Using all genes as TFs.")
    else:
        TFs = "all"

    logger.info(f"Starting GRN inference using {len(TFs) if TFs != 'all' else 'all'} TFs.")
    inferred_grn = grnboost2(real_cells_df, tf_names=TFs, verbose=True)  # pyright: ignore[reportArgumentType]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    inferred_grn.to_csv(output_path, index=False)
    logger.info(f"Successfully saved GRN inferred by GRNBoost2 GRN to {output_path}")


@click.command()
@click.option("--cells", "adata_path", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--out", "output_path", required=True, type=click.Path(writable=True, dir_okay=False, path_type=Path))
@click.option("--tfs", "tfs_path", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
def cli(adata_path: Path, output_path: Path, tfs_path: Path | None = None):
    main(adata_path, output_path, tfs_path)


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
