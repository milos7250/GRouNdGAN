from pathlib import Path

import numpy as np
import pandas as pd

from loggers import setup_logger

logger = setup_logger(__name__)


@pd.api.extensions.register_dataframe_accessor("grn")
class GRNAccessor:
    """
    Pandas DataFrame accessor for Gene Regulatory Networks (GRNs).

    This accessor provides methods to manipulate and analyze GRNs represented as DataFrames
    with 'TF', 'target', and 'importance' columns.

    Usage
    -----
    Import to register the accessor:

        from grn_accessor import GRNAccessor

    If `df` is a pandas DataFrame representing a GRN, access methods via `df.grn.method_name()`. Or load from CSV
    directly (see `from_csv()` class method):

        df = GRNAccessor.from_csv("path/to/grn.csv")
        df.grn.method_name()
    """
    def __init__(self, pandas_obj: pd.DataFrame) -> None:
        self._validate(pandas_obj)
        self._obj = GRNAccessor._coalesce_duplicates(pandas_obj)

    @staticmethod
    def from_csv(
        grn_file: Path | str,
        col_names: dict[str, str] | None = {"TF": "TF", "target": "target", "importance": "importance"},
    ) -> pd.DataFrame:
        """
        Loads a GRN from a CSV file. The CSV file is expected to have columns representing 'TF' and'target'.
        Optionally it can include columns representing 'importance', otherwise the file is assumed to be sorted by
        importance.

        Args:
            grn_file (Path): Path to the GRN CSV file.
            col_names (dict[str, str], optional): Dictionary mapping 'TF', 'target', and 'importance' to the actual
                column names in the CSV file. Defaults to {"TF": "TF", "target": "target", "importance": "importance"}.

        Returns:
            pd.DataFrame: DataFrame containing the GRN sorted by importance.
        """
        if col_names is None:
            col_names = {"TF": "TF", "target": "target", "importance": "importance"}
        if "importance" not in col_names:
            df = pd.read_csv(
                grn_file,
                dtype={col_names["TF"]: str, col_names["target"]: str},
                usecols=list(col_names.values()),
            ).rename(columns={v: k for k, v in col_names.items()})
            df["importance"] = np.arange(len(df), 0, -1).astype(float)  # Assign importance based on rows if missing
        else:
            df = (
                pd.read_csv(
                    grn_file,
                    dtype={col_names["TF"]: str, col_names["target"]: str, col_names["importance"]: float},
                    usecols=list(col_names.values()),
                )
                .sort_values(col_names["importance"], ascending=False)
                .rename(columns={v: k for k, v in col_names.items()})
            )
        return df

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        """
        Validates the GRN DataFrame to ensure it contains the required columns with correct data types.
        """
        if not all(col in df.columns for col in ["TF", "target", "importance"]):
            raise AttributeError("GRN must contain 'TF', 'target' and 'importance' columns.")
        if not pd.api.types.is_string_dtype(df["TF"]) or not pd.api.types.is_string_dtype(df["target"]):
            raise AttributeError("'TF' and 'target' columns must be of string type.")
        if "importance" in df.columns and not pd.api.types.is_numeric_dtype(df["importance"]):
            raise AttributeError("'importance' column must be of numeric type.")

    @staticmethod
    def _coalesce_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        """
        Coalesces duplicate edges in the GRN by summing their importance values.

        Returns:
            pd.DataFrame: DataFrame with duplicate edges coalesced.
        """
        if (no_of_duplicates := df.duplicated(subset=["TF", "target"]).sum()) == 0:
            return df
        logger.warning(f"{no_of_duplicates} duplicate edges found in GRN. Coalescing by summing importance values.")
        coalesced = (
            pd.DataFrame(df.groupby(["TF", "target"], as_index=False)["importance"].sum())
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        return coalesced

    @property
    def normalized_importance(self) -> pd.Series:
        """
        Returns a Series containing the normalized importance values of the GRN edges, scaled to sum to 1.

        Returns:
            pd.Series: Series containing normalized importance values.
        """
        return self._obj["importance"] / self._obj["importance"].sum()

    def to_bipartite(self) -> pd.DataFrame:
        """
        FIlters the GRN in order to represent it as a bipartite graph. For each gene, the importance of edges where the
        gene is a TF is summed up, as well as the importance of edges where the gene is a target. Genes with higher
        TF importance than target importance are considered TFs, and only edges from these TFs to non-TF targets are
        retained.

        Returns:
            pd.DataFrame: Filtered bipartite GRN.
        """
        gene_names = self._obj["TF"].tolist() + self._obj["target"].tolist()
        gene_names = pd.Index(gene_names).unique().sort_values()
        importances = pd.DataFrame(
            {
                "TF": self._obj.groupby("TF")["importance"].sum(),
                "target": self._obj.groupby("target")["importance"].sum(),
            },
            index=gene_names,
        )
        TFs = importances[importances["TF"] > importances["target"]].index
        filtered = self._obj[self._obj["TF"].isin(TFs) & ~self._obj["target"].isin(TFs)]
        targets = filtered["target"].unique()
        logger.info(f"Filtered GRN to bipartite graph with {len(TFs)} TFs and {len(targets)} targets.")
        filtered = filtered.reset_index(drop=True)
        return filtered

    def intersect(self, other: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the intersection of this GRN with another GRN. The intersection contains edges that are present in
        both GRNs, with importance values averaged.

        Args:
            other (GRN): Another GRN object to intersect with.

        Returns:
            pd.DataFrame: DataFrame containing the intersected GRN.
        """
        self._validate(other)
        first = self._obj.copy()
        second = other.grn._obj.copy()
        sum_importance = first["importance"].sum()
        first["normalized_importance"] = first.grn.normalized_importance
        second["normalized_importance"] = second.grn.normalized_importance
        merged = pd.merge(
            first,
            second,
            on=["TF", "target"],
            suffixes=("_1", "_2"),
            validate="one_to_one",
        )
        merged["importance"] = np.nanmean(merged[["normalized_importance_1", "normalized_importance_2"]].to_numpy(), axis=1)
        merged.drop(columns=["normalized_importance_1", "normalized_importance_2"], inplace=True)
        merged["importance"] = merged["importance"] / merged["importance"].sum() * sum_importance
        merged = merged.sort_values("importance", ascending=False).reset_index(drop=True)
        return merged

    def union(self, other: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the union of this GRN with another GRN. The union contains all edges from both GRNs, with importance
        values averaged for edges present in both GRNs.

        Args:
            other (GRN): Another GRN object to union with.

        Returns:
            pd.DataFrame: DataFrame containing the unioned GRN.
        """
        self._validate(other)
        first = self._obj.copy()
        second = other.grn._obj.copy()
        sum_importance = first["importance"].sum()
        first["normalized_importance"] = first.grn.normalized_importance
        second["normalized_importance"] = second.grn.normalized_importance
        print(first)
        print(second)
        merged = pd.merge(
            first,
            second,
            on=["TF", "target"],
            how="outer",
            suffixes=("_1", "_2"),
            validate="one_to_one",
        )
        merged["importance"] = np.nanmean(merged[["normalized_importance_1", "normalized_importance_2"]].to_numpy(), axis=1)
        merged.drop(columns=["normalized_importance_1", "normalized_importance_2"], inplace=True)
        merged["importance"] = merged["importance"] / merged["importance"].sum() * sum_importance
        merged = merged.sort_values("importance", ascending=False).reset_index(drop=True)
        return merged

    def to_undirected(self) -> pd.DataFrame:
        """
        Converts the directed GRN into an undirected GRN by repeating each edge in both directions.

        Returns:
            pd.DataFrame: DataFrame containing the undirected GRN.
        """
        reversed_grn = self._obj.rename(columns={"TF": "target", "target": "TF"})
        undirected_grn = pd.concat([self._obj, reversed_grn], ignore_index=True)
        undirected_grn = undirected_grn.sort_values("importance", ascending=False).reset_index(drop=True)
        return undirected_grn
