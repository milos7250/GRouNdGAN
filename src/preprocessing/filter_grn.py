#!/usr/bin/env python3
from pathlib import Path

import pandas as pd


class GRN:
    def __init__(self, grn_file: Path) -> None:
        """
        Initializes the GRN object by loading the GRN from a CSV file. The CSV file is expected to have columns:
        'TF', 'target', and 'importance'.
        
        Args:
            grn_file (Path): Path to the GRN CSV file.
        """
        self.grn = self.load_grn(grn_file)
        
    def load_grn(self, grn_file: Path) -> pd.DataFrame:
        """
        Loads a GRN from a CSV file. The CSV file is expected to have columns: 'TF', 'target', and 'importance'.
        
        Args:
            grn_file (Path): Path to the GRN CSV file.
            
        Returns:
            pd.DataFrame: DataFrame containing the GRN sorted by importance.
        """
        return pd.read_csv(
            grn_file,
            dtype={"TF": str, "target": str, "importance": float},
            usecols=["TF", "target", "importance"],
        ).sort_values("importance", ascending=False)
        
    def filtered_bipartite(self) -> pd.DataFrame:
        """
        FIlters the GRN in order to represent it as a bipartite graph. For each gene, the importance of edges where the
        gene is a TF is summed up, as well as the importance of edges where the gene is a target. Genes with higher
        TF importance than target importance are considered TFs, and only edges from these TFs to non-TF targets are
        retained.
        
        Returns:
            pd.DataFrame: Filtered bipartite GRN.
        """
        gene_names = self.grn["TF"].tolist() + self.grn["target"].tolist()
        gene_names = pd.Index(gene_names).unique().sort_values()
        importances = pd.DataFrame(
            {
                "TF": self.grn.groupby("TF")["importance"].sum(),
                "target": self.grn.groupby("target")["importance"].sum(),
            },
            index=gene_names,
        )
        TFs = importances[importances["TF"] > importances["target"]].index
        return self.grn[self.grn["TF"].isin(TFs) & ~self.grn["target"].isin(TFs)]

# Allow running this module directly to filter a GRN CSV file
if __name__ == "__main__":
    import click
    
    @click.command()
    @click.option("--input", "-i", type=click.Path(exists=True), required=True, help="Input GRN CSV file")
    @click.option("--output", "-o", type=click.Path(), required=True, help="Output filtered bipartite GRN CSV file")
    def main(input: Path, output: Path):
        bipartite_grn = GRN(input)
        filtered_grn = bipartite_grn.filtered_bipartite()
        
        filtered_grn.to_csv(output, index=False)
        
    main()