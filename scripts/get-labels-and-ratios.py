#!/usr/bin/env python3
import sys
from pathlib import Path


def conditional_labels_and_ratios(adata_file: str | Path) -> tuple[int, list[float]]:
    """
    Provides the number of classes and their ratios for conditional GAN tests, extracted from the training data.
    """
    # Get the number of classes and their ratios from the training data
    import anndata as ad

    adata = ad.read_h5ad(adata_file, backed="r")
    num_classes = adata.uns["clusters_no"]
    # Need to sort the label ratios according to the cluster labels (assuming they are integers starting from 0)
    label_ratios: list[float] = [adata.uns["cluster_ratios"][str(i)] for i in range(num_classes)]

    return num_classes, label_ratios

if __name__ == "__main__":
    num_classes, label_ratios = conditional_labels_and_ratios(sys.argv[1])
    print(f"number of classes = {num_classes}")
    print(f"label ratios = {' '.join(map(str, label_ratios))}")