from typing import TYPE_CHECKING

import scanpy as sc
from scipy import sparse
from torch import are_deterministic_algorithms_enabled, from_numpy  # pyright: ignore[reportUnknownVariableType]
from torch.cuda import is_available as is_cuda_available
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from torch import Tensor


class SCDataset(Dataset["tuple[Tensor, Tensor]"]):
    def __init__(self, path: "Path") -> None:
        """
        Create a dataset from the h5ad processed data. Use the
        preprocessing/preprocess.py script to create the h5ad train,
        test, and validation files.

        Parameters
        ----------
        path : Union[str, bytes, os.PathLike]
            Path to the h5ad file.
        """
        data = sc.read_h5ad(path)
        # data = sc.read_h5ad(path, backed="r") # for larger-than-memory datasets, unsure what performance impact is

        if not isinstance(data.X, sparse.csr_matrix):
            raise ValueError("The data matrix is not in sparse csr format. Please preprocess the data accordingly.")
        else:
            self.cells = data.X
        if "cluster" not in data.obs:
            raise ValueError("Cluster labels not found in the data. Please preprocess the data accordingly.")
        else:
            self.clusters = from_numpy(data.obs["cluster"].to_numpy(dtype=int))

    def __getitem__(self, index: int) -> "tuple[Tensor, Tensor]":
        """
        Parameters
        ----------
        index : int

        Returns
        -------
        Tuple[Tensor, Tensor]
            Gene expression, Cluster label Tensor tuple.
        """
        cells = from_numpy(self.cells[index].toarray())
        return (cells.squeeze(), self.clusters[index].squeeze())

    def __getitems__(self, indices: list[int]) -> "list[tuple[Tensor, Tensor]]":
        """
        Parameters
        ----------
        indices : list[int]

        Returns
        -------
        Tuple[Tensor, Tensor]
            Gene expression, Cluster label Tensor tuple.
        """
        cells = from_numpy(self.cells[indices].toarray())
        return [(cell.squeeze(), cluster.squeeze()) for cell, cluster in zip(cells, self.clusters[indices])]

    def get_all_cells(self) -> "tuple[Tensor, Tensor]":
        """
        Returns
        -------
        Tuple[Tensor, Tensor]
            All gene expression and cluster label Tensors.
        """
        return from_numpy(self.cells.toarray()), self.clusters

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            Number of samples (cells).
        """
        return self.cells.shape[0]  # pyright: ignore[reportOptionalSubscript]


class SCDataLoader(DataLoader["tuple[Tensor, Tensor]"]):
    """
    Subclass of DataLoader to allow type hinting of the returned data.
    """

    dataset: SCDataset  # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(self, dataset: SCDataset, *args: "Any", **kwargs: "Any") -> None:
        super().__init__(dataset, *args, **kwargs)


def get_loader(
    file_path: "Path",
    batch_size: int | None = None,
    shuffle: bool = False,
    drop_last: bool = False,
    deterministic: bool | None = None,
) -> SCDataLoader:
    """
    Provides an IterableLoader over a scRNA-seq Dataset read from given h5ad file.

    Parameters
    ----------
    file_path : Path
        Path to the h5ad file.
    batch_size : int | None
        Training batch size. If not specified, the entire dataset
        is returned at each load. Default is None.
    shuffle : bool
        Whether to shuffle the data in the loader. Default is False.
    drop_last : bool
        Whether to drop the last incomplete batch. Default is False.
    deterministic : bool | None
        Whether to use deterministic data loading. Default is based on calling torch.are_deterministic_algorithms_enabled().

    Returns
    -------
    DataLoader
        Iterable data loader over the dataset.
    """
    deterministic = deterministic or are_deterministic_algorithms_enabled()
    dataset = SCDataset(file_path)

    # return the whole dataset if batch size if not specified
    if batch_size is None:
        batch_size = len(dataset)

    if not deterministic:
        return SCDataLoader(
            dataset,
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=is_cuda_available(),
            num_workers=2,
            persistent_workers=True,
        )
    else:
        import random

        import numpy
        import torch
        
        def seed_worker(_) -> None:
            worker_seed = torch.initial_seed() % 2**32
            numpy.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)
        
        return SCDataLoader(
            dataset,
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=is_cuda_available(),
            num_workers=2,
            worker_init_fn=seed_worker,
            generator=g,
        )
