"""Tests for mask generation in CausalGenerator.

Verifies that the scipy.sparse-based mask construction produces connectivity
indices identical to the original dense torch.Tensor approach.
"""

import itertools
import pickle

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from tests.resources.constants import CAUSAL_GRAPH_FILE

from .. import pytestmark  # pyright: ignore[reportUnusedImport]  # noqa: F401


def _build_masks_dense(
    causal_graph: dict[int, set[int]],
    num_tfs: int,
    tfs: list[int],
    genes: list[int],
    noise_per_gene: int,
    depth_per_gene: int,
    width_scale_per_gene: int,
) -> list[torch.Tensor]:
    """Reproduce the original dense-tensor mask construction and return connectivity tensors.

    Returns a list of connectivity tensors (one per generator block) in the format
    expected by SparseLinear: shape (2, nnz) with row 0 = output indices, row 1 = input indices.
    """
    regulators = list(itertools.chain.from_iterable(causal_graph.values()))
    num_genes = len(genes)
    num_noises = num_genes * noise_per_gene
    hidden_dims = (len(regulators) + num_noises) * width_scale_per_gene

    input_mask = torch.zeros(num_tfs, hidden_dims, dtype=torch.int64)
    hidden_mask = torch.zeros(hidden_dims, hidden_dims, dtype=torch.int64)
    output_mask = torch.zeros(hidden_dims, num_genes, dtype=torch.int64)

    prev_gene_hidden_dims = 0
    for gene, gene_regulators in causal_graph.items():
        gene_idx = genes.index(gene)
        curr_gene_hidden_dims = width_scale_per_gene * (len(gene_regulators) + noise_per_gene)
        for gene_regulator in gene_regulators:
            gene_regulator_idx = tfs.index(gene_regulator)
            input_mask[
                gene_regulator_idx,
                prev_gene_hidden_dims : prev_gene_hidden_dims + curr_gene_hidden_dims,
            ] = 1

        noise_mask = torch.zeros(noise_per_gene, hidden_dims, dtype=torch.int64)
        noise_mask[:, prev_gene_hidden_dims : prev_gene_hidden_dims + curr_gene_hidden_dims] = 1
        input_mask = torch.cat([input_mask, noise_mask])

        hidden_mask[
            prev_gene_hidden_dims : prev_gene_hidden_dims + curr_gene_hidden_dims,
            prev_gene_hidden_dims : prev_gene_hidden_dims + curr_gene_hidden_dims,
        ] = 1

        output_mask[
            prev_gene_hidden_dims : prev_gene_hidden_dims + curr_gene_hidden_dims,
            gene_idx,
        ] = 1

        prev_gene_hidden_dims += curr_gene_hidden_dims

    masks = [input_mask, *([hidden_mask] * depth_per_gene), output_mask]

    return [torch.nonzero(mask.T).T for mask in masks]


def _build_masks_sparse(
    causal_graph: dict[int, set[int]],
    num_tfs: int,
    tfs: list[int],
    genes: list[int],
    noise_per_gene: int,
    depth_per_gene: int,
    width_scale_per_gene: int,
) -> list[torch.Tensor]:
    """Reproduce the new scipy.sparse-based mask construction and return connectivity tensors.

    Returns a list of connectivity tensors in the same format as _build_masks_dense.
    """
    regulators = list(itertools.chain.from_iterable(causal_graph.values()))
    num_genes = len(genes)
    num_noises = num_genes * noise_per_gene
    hidden_dims = (len(regulators) + num_noises) * width_scale_per_gene

    input_rows: list[np.ndarray] = []
    input_cols: list[np.ndarray] = []
    hidden_rows: list[np.ndarray] = []
    hidden_cols: list[np.ndarray] = []
    output_rows: list[np.ndarray] = []
    output_cols: list[np.ndarray] = []

    prev_gene_hidden_dims = 0
    for gene_idx, (gene, gene_regulators) in enumerate(causal_graph.items()):
        curr_gene_hidden_dims = width_scale_per_gene * (len(gene_regulators) + noise_per_gene)
        start = prev_gene_hidden_dims
        end = prev_gene_hidden_dims + curr_gene_hidden_dims

        for gene_regulator in gene_regulators:
            gene_regulator_idx = tfs.index(gene_regulator)
            input_rows.append(np.full(curr_gene_hidden_dims, gene_regulator_idx, dtype=np.int64))
            input_cols.append(np.arange(start, end, dtype=np.int64))

        noise_row_offset = num_tfs + gene_idx * noise_per_gene
        for noise_idx in range(noise_per_gene):
            input_rows.append(np.full(curr_gene_hidden_dims, noise_row_offset + noise_idx, dtype=np.int64))
            input_cols.append(np.arange(start, end, dtype=np.int64))

        block = np.arange(start, end, dtype=np.int64)
        br, bc = np.meshgrid(block, block, indexing="ij")
        hidden_rows.append(br.flatten())
        hidden_cols.append(bc.flatten())

        output_rows.append(block)
        output_cols.append(np.full(curr_gene_hidden_dims, gene_idx, dtype=np.int64))

        prev_gene_hidden_dims = end

    num_input_rows = num_tfs + num_noises

    input_mask = sp.csr_matrix(
        (
            np.ones(sum(len(r) for r in input_rows), dtype=bool),
            (np.concatenate(input_rows), np.concatenate(input_cols)),
        ),
        shape=(num_input_rows, hidden_dims),
    )
    hidden_mask = sp.csr_matrix(
        (
            np.ones(sum(len(r) for r in hidden_rows), dtype=bool),
            (np.concatenate(hidden_rows), np.concatenate(hidden_cols)),
        ),
        shape=(hidden_dims, hidden_dims),
    )
    output_mask = sp.csr_matrix(
        (
            np.ones(sum(len(r) for r in output_rows), dtype=bool),
            (np.concatenate(output_rows), np.concatenate(output_cols)),
        ),
        shape=(hidden_dims, num_genes),
    )

    masks = [input_mask, *([hidden_mask] * depth_per_gene), output_mask]

    connectivities = []
    for mask in masks:
        coo = mask.tocoo()
        conn = torch.stack([
            torch.as_tensor(coo.col, dtype=torch.long),
            torch.as_tensor(coo.row, dtype=torch.long),
        ])
        connectivities.append(conn)
    return connectivities


def _sort_connectivity(conn: torch.Tensor) -> torch.Tensor:
    """Sort connectivity tensor (2, nnz) by output index, then input index, for comparison."""
    np_conn = conn.numpy()
    order = np.lexsort((np_conn[1], np_conn[0]))
    return conn[:, torch.as_tensor(order, dtype=torch.long)]


def _graph_info(causal_graph: dict[int, set[int]]):
    genes = list(causal_graph.keys())
    regulators = list(itertools.chain.from_iterable(causal_graph.values()))
    tfs = list(set(regulators))
    return genes, regulators, tfs


class TestMaskConnectivity:
    @pytest.fixture
    def small_causal_graph(self) -> dict[int, set[int]]:
        return {
            10: {1, 2, 3},
            11: {2, 3, 4},
            12: {1, 4, 5},
            13: {3, 5, 6},
            14: {1, 2, 6},
        }

    @pytest.fixture
    def full_causal_graph(self) -> dict[int, set[int]]:
        with open(CAUSAL_GRAPH_FILE, "rb") as f:
            return pickle.load(f)

    @pytest.mark.parametrize(
        "noise_per_gene, depth_per_gene, width_scale_per_gene",
        [(1, 2, 1), (1, 3, 2), (2, 1, 1)],
        ids=["default", "deep-wide", "multi-noise"],
    )
    def test_connectivity_matches_small(
        self,
        small_causal_graph: dict[int, set[int]],
        noise_per_gene: int,
        depth_per_gene: int,
        width_scale_per_gene: int,
    ) -> None:
        genes, _, tfs = _graph_info(small_causal_graph)
        num_tfs = len(tfs)

        kwargs = {
            "causal_graph": small_causal_graph,
            "num_tfs": num_tfs,
            "tfs": tfs,
            "genes": genes,
            "noise_per_gene": noise_per_gene,
            "depth_per_gene": depth_per_gene,
            "width_scale_per_gene": width_scale_per_gene,
        }

        dense_conns = _build_masks_dense(**kwargs)
        sparse_conns = _build_masks_sparse(**kwargs)

        assert len(dense_conns) == len(sparse_conns)
        for i, (dense_conn, sparse_conn) in enumerate(zip(dense_conns, sparse_conns)):
            assert dense_conn.shape == sparse_conn.shape, (
                f"Block {i}: shape mismatch dense={dense_conn.shape} vs sparse={sparse_conn.shape}"
            )
            dense_sorted = _sort_connectivity(dense_conn)
            sparse_sorted = _sort_connectivity(sparse_conn)
            assert torch.equal(dense_sorted, sparse_sorted), f"Block {i}: connectivity mismatch after sorting"

    def test_connectivity_matches_full(self, full_causal_graph: dict[int, set[int]]) -> None:
        genes, _, tfs = _graph_info(full_causal_graph)
        num_tfs = len(tfs)

        kwargs = {
            "causal_graph": full_causal_graph,
            "num_tfs": num_tfs,
            "tfs": tfs,
            "genes": genes,
            "noise_per_gene": 1,
            "depth_per_gene": 2,
            "width_scale_per_gene": 1,
        }

        dense_conns = _build_masks_dense(**kwargs)
        sparse_conns = _build_masks_sparse(**kwargs)

        assert len(dense_conns) == len(sparse_conns)
        for i, (dense_conn, sparse_conn) in enumerate(zip(dense_conns, sparse_conns)):
            assert dense_conn.shape == sparse_conn.shape, (
                f"Block {i}: shape mismatch dense={dense_conn.shape} vs sparse={sparse_conn.shape}"
            )
            dense_sorted = _sort_connectivity(dense_conn)
            sparse_sorted = _sort_connectivity(sparse_conn)
            assert torch.equal(dense_sorted, sparse_sorted), f"Block {i}: connectivity mismatch after sorting"
