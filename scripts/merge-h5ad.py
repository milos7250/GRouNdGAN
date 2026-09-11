#!/usr/bin/env python3
"""
A simple script to merge multiple h5ad files into a single h5ad file.
Files are merged by concatenating the observations (cells) and keeping the union of the variables (genes).

Usage: 
    python merge_h5ad.py output_file.h5ad input_file1.h5ad input_file2.h5ad ...
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ["NUMBA_CACHE_DIR"] = "./tmp/numba"
os.environ["MPLCONFIGDIR"] = "./tmp/matplotlib"

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix

# Verify that output file does not already exist
if Path(sys.argv[1]).exists():
    raise FileExistsError(f"Output file already exists: {sys.argv[1]}")

# Verify that the input files exist
for f in sys.argv[2:]:
    if not Path(f).exists():
        raise FileNotFoundError(f"File not found: {f}")

adatas = [sc.read_h5ad(f) for f in sys.argv[2:]]

genes = [adata.var_names for adata in adatas]
obs_col_intersection = set(adatas[0].obs.columns).intersection(*[adata.obs.columns for adata in adatas[1:]])
obs_col_intersection = sorted(obs_col_intersection)


def get_columns(adata: ad.AnnData):
    return {dtype: adata.obs.select_dtypes(include=dtype).columns for dtype in ["object", "category", "number"]}


column_dtypes = defaultdict(set)

for adata in adatas:
    for dtype, columns in get_columns(adata).items():
        for column in columns:
            column_dtypes[column].add(dtype)

column_defaults = {}

for column, dtypes in column_dtypes.items():
    if len(dtypes) > 1:
        for adata in adatas:
            if column in adata.obs.columns:
                adata.obs[column] = adata.obs[column].astype("object")
        column_defaults[column] = "unknown"
    else:
        column_defaults[column] = np.nan if dtypes.copy().pop() == "number" else "unknown"

column_defaults["label"] = "Unknown"

for adata in adatas:
    for col in set(obs_col_intersection).difference(adata.obs.columns):
        adata.obs[col] = column_defaults[col]
    adata.obs = adata.obs[obs_col_intersection]

    for col in obs_col_intersection:
        if column_dtypes[col] == {"number"}:
            continue
        adata.obs[col] = adata.obs[col].astype(str).astype("category")

adata_outer = ad.concat(adatas, join="outer")
adata_outer.obs_names_make_unique()

for adata in adatas:
    for var in adata.var.columns:
        if var not in adata_outer.var.columns:
            adata_outer.var.loc[adata.var_names, var] = adata.var[var]
        elif not adata_outer.var[var].equals(adata.var[var]):
            raise ValueError(f"Variable '{var}' has different values in different datasets.")


# Merge spliced, unspliced, and ambiguous genes if they exist by adding ambiguous counts to spliced counts, and then removing unspliced and ambiguous genes from the dataset.
# This is done to simplify the dataset for downstream tasks.
if 'gene_type' in adata_outer.var.columns:
    gene_types = adata_outer.var['gene_type'].unique()
    if set(gene_types) == {'spliced', 'unspliced', 'ambiguous'}:
        spliced_genes = adata_outer.var_names[adata_outer.var['gene_type'] == 'spliced']
        ambiguous_genes = spliced_genes.str.replace(r'$', '-A', regex=True)
        X = adata_outer[:, spliced_genes].X + adata_outer[:, ambiguous_genes].X
        adata_outer = adata_outer[:, ~adata_outer.var_names.str.endswith('-A') & ~adata_outer.var_names.str.endswith('-U')].copy(filename=Path(sys.argv[1]))
        adata_outer.X = X

adata_outer.X = csr_matrix(adata_outer.X)

# Sort obs columns alphabetically to make reproducible
adata_outer.obs = adata_outer.obs.reindex(sorted(adata_outer.obs.columns), axis=1)
adata_outer.var = adata_outer.var.reindex(sorted(adata_outer.var.columns), axis=1)

Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
adata_outer.write(Path(sys.argv[1]), compression="gzip")
