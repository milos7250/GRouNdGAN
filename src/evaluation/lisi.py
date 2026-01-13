# CODE TAKEN FROM: HarmonyPy (https://github.com/slowkow/harmonypy)
# LISI - The Local Inverse Simpson Index
# Copyright (C) 2018  Ilya Korsunsky
#               2019  Kamil Slowikowski <kslowikowski@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections.abc import Collection

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def compute_lisi(
    X: np.ndarray | sparse.csr_matrix,
    metadata: pd.DataFrame,
    label_colnames: Collection[str],
    perplexity: int = 30,
) -> np.ndarray:
    """
    Compute the Local Inverse Simpson Index (LISI) for each label column in the metadata.

    LISI measures the diversity of labels in the local neighborhood of each sample,
    based on distances in the feature space `X`. It is commonly used to evaluate
    dataset mixing (e.g., integration of multiple batches or experimental conditions).

    A LISI score close to:
        - 1 indicates homogeneity (neighbors mostly from one label category),
        - N (number of categories) indicates high diversity (neighbors from all categories).

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features), representing the embedding space (e.g., PCA, UMAP).

    metadata : pd.DataFrame
        DataFrame containing categorical metadata for each sample (e.g., batch or cell type).

    label_colnames : Collection[str]
        List of column names in `metadata` for which to compute LISI.

    perplexity : float, optional
        Perplexity parameter influencing the neighborhood size, by default 30.

    Returns
    -------
    np.ndarray
        A matrix of shape (n_samples, len(label_colnames)) with LISI scores per label.
    """
    n_cells = metadata.shape[0]
    n_labels = len(label_colnames)
    # We need at least 3 * n_neigbhors to compute the perplexity
    knn = NearestNeighbors(n_neighbors=perplexity * 3, algorithm="kd_tree").fit(X)
    distances, indices = knn.kneighbors(X)
    # Don't count yourself
    indices = indices[:, 1:]
    distances = distances[:, 1:]
    # Save the result
    lisi_df = np.zeros((n_cells, n_labels))
    for i, label in enumerate(label_colnames):
        labels = pd.Categorical(metadata[label])
        n_categories = len(labels.categories)
        simpson = compute_simpson(distances.T, indices.T, labels, n_categories, perplexity)
        lisi_df[:, i] = 1 / simpson
    return lisi_df


def compute_simpson(
    distances: np.ndarray,
    indices: np.ndarray,
    labels: pd.Categorical,
    n_categories: int,
    perplexity: float,
    tol: float = 1e-5,
) -> np.ndarray:
    """
    Compute the Simpson's diversity index for each sample using a locally scaled probability distribution.

    This function computes the effective number of categories in each sample's neighborhood,
    based on the similarity-weighted label distribution. It is internally used by `compute_lisi`.

    Parameters
    ----------
    distances : np.ndarray
        Array of shape (n_neighbors, n_samples) with distances to each sample's neighbors.

    indices : np.ndarray
        Array of shape (n_neighbors, n_samples) with indices of nearest neighbors for each sample.

    labels : pd.Categorical
        Categorical labels for all samples, aligned with the rows of `X`.

    n_categories : int
        Number of unique categories in the label.

    perplexity : float
        Perplexity value, used to set the target entropy for neighborhood probability distribution.

    tol : float, optional
        Tolerance for the entropy convergence, by default 1e-5.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples,) with the Simpson's index per sample.
        Values closer to 1 indicate less diversity, higher values indicate more.
    """
    n = distances.shape[1]
    p = np.zeros(distances.shape[0])
    simpson = np.zeros(n)
    logU = np.log(perplexity)
    # Loop through each cell.
    for i in range(n):
        beta = 1
        betamin = -np.inf
        betamax = np.inf
        # Compute h_diff
        p = np.exp(-distances[:, i] * beta)
        p_sum = np.sum(p)
        if p_sum == 0:
            h = 0
            p = np.zeros(distances.shape[0])
        else:
            h = np.log(p_sum) + beta * np.sum(distances[:, i] * p) / p_sum
            p = p / p_sum
        h_diff = h - logU
        n_tries = 50
        for _ in range(n_tries):
            # Stop when we reach the tolerance
            if abs(h_diff) < tol:
                break
            # Update beta
            if h_diff > 0:
                betamin = beta
                if not np.isfinite(betamax):
                    beta *= 2
                else:
                    beta = (beta + betamax) / 2
            else:
                betamax = beta
                if not np.isfinite(betamin):
                    beta /= 2
                else:
                    beta = (beta + betamin) / 2
            # Compute Hdiff
            p = np.exp(-distances[:, i] * beta)
            p_sum = np.sum(p)
            if p_sum == 0:
                h = 0
                p = np.zeros(distances.shape[0])
            else:
                h = np.log(p_sum) + beta * np.sum(distances[:, i] * p) / p_sum
                p = p / p_sum
            h_diff = h - logU
        # distancesefault value
        if h == 0:
            simpson[i] = -1
        # Simpson's index
        for label_category in labels.categories:
            ix = indices[:, i]
            q = labels[ix] == label_category
            if np.any(q):
                p_sum = np.sum(p[q])
                simpson[i] += p_sum * p_sum
    return simpson
