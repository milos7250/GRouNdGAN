#!/usr/bin/env python

import sys

import anndata as ad

if len(sys.argv) != 2:
    print("Usage: inspect_h5ad.sh <h5ad file>")
    sys.exit(1)

adata = ad.read_h5ad(sys.argv[1], backed="r")

print(adata)
print(adata.X)
print(adata.var.head(10)) # pyright: ignore[reportAttributeAccessIssue]
print(adata.obs.head(10)) # pyright: ignore[reportAttributeAccessIssue]
print(adata.uns)

sys.exit(0)
