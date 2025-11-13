import typing

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


class MMD:
    """
    Maximum Mean Discrepancy (MMD) class for computing distribution similarity
    between real and generated samples using Gaussian kernels.
    """

    def __init__(self, real_cells: np.ndarray, device: str = "cpu"):
        """
        Initialize the MMD class with scale and weight parameters based on the median
        nearest neighbor distance among real cells.

        Parameters
        ----------
        real_cells : np.ndarray
            A NumPy array representing real cell data (cells x features).
        device : str
            Device to store tensors ('cpu' or 'cuda').
        """
        self.device = device
        n_neighbors = 25
        med = np.ones(20)

        for ii in range(1, 20):
            sample = real_cells[np.random.randint(real_cells.shape[0] - 1, size=real_cells.shape[0]), :]
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(sample)
            distances, _ = nbrs.kneighbors(sample)
            med[ii] = np.median(distances[:, 1:n_neighbors])  # exclude self-distance

        med = np.median(med)
        scales = torch.tensor([med / 2, med, med * 2], dtype=torch.float32, device=self.device)
        weights = torch.ones(len(scales), dtype=torch.float32, device=self.device)

        # Reshape for broadcasting
        self.scales = scales.view(-1, 1, 1)
        self.weights = weights.view(-1, 1, 1)

    def squared_distance(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise squared Euclidean distances between rows of X and Y.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (n, d).
        Y : torch.Tensor
            Input tensor of shape (m, d).

        Returns
        -------
        torch.Tensor
            A tensor of shape (n, m) representing squared distances.
        """
        # X is nxd, Y is mxd, returns nxm matrix of all pairwise Euclidean distances
        # broadcasted subtraction, a square, and a sum.
        X = X.to(self.device)
        Y = Y.to(self.device)
        r = X.unsqueeze(1)  # shape: (n, 1, d)
        return torch.sum((r - Y) ** 2, dim=-1)  # shape: (n, m)

    def gaussian_kernel(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the multi-scale Gaussian kernel between two datasets.

        Parameters
        ----------
        a : torch.Tensor
            Input tensor of shape (n, d).
        b : torch.Tensor
            Input tensor of shape (m, d).

        Returns
        -------
        torch.Tensor
            A tensor of shape (n, m) representing the Gaussian kernel matrix.
        """
        numerator = self.squared_distance(a, b).unsqueeze(0)  # shape: (1, n, m)
        kernel = torch.sum(self.weights * torch.exp(-numerator / (self.scales**2)), dim=0)
        return kernel

    def compute(
        self,
        a: typing.Union[np.ndarray, torch.Tensor],
        b: typing.Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the Maximum Mean Discrepancy (MMD) between two samples.

        Parameters
        ----------
        a : np.ndarray or torch.Tensor
            First sample of shape (n, d).
        b : np.ndarray or torch.Tensor
            Second sample of shape (m, d).

        Returns
        -------
        torch.Tensor
            The MMD score between the two distributions.
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a, dtype=torch.float32, device=self.device)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b, dtype=torch.float32, device=self.device)

        return (
            self.gaussian_kernel(a, a).mean()
            + self.gaussian_kernel(b, b).mean()
            - 2 * self.gaussian_kernel(a, b).mean()
        )
