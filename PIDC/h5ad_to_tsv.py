from pathlib import Path

import click
import pandas as pd
import scanpy as sc


def h5ad_to_tsv(input: Path, output: Path):

    adata = sc.read_h5ad(input)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X # pyright: ignore[reportOptionalMemberAccess,reportAttributeAccessIssue]

    # Transpose to get genes as rows
    df = pd.DataFrame(X.T, index=adata.var_names, columns=adata.obs_names)  # pyright: ignore[reportAttributeAccessIssue, reportOptionalMemberAccess, reportArgumentType, reportCallIssue]

    # Save with gene names in first column (index=True), and sample names as headers
    df.to_csv(output, sep="\t", index=True)
    
if __name__ == "__main__":
    @click.command()
    @click.option("--input", "-i", type=click.Path(exists=True, path_type=Path), required=True, help="Input h5ad file path.")
    @click.option("--output", "-o", type=click.Path(path_type=Path), required=True, help="Output TSV file path.")
    def _main(input: Path, output: Path):
        h5ad_to_tsv(input, output)
    
    _main()