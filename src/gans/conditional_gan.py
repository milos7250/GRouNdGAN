from abc import ABC
from typing import TYPE_CHECKING

import torch

from .gan import GAN

if TYPE_CHECKING:
    from torch import Tensor


class ConditionalGAN(GAN, ABC):
    def __init__(
        self,
        genes_no: int,
        batch_size: int,
        latent_dim: int,
        gen_layers: list[int],
        crit_layers: list[int],
        num_classes: int,
        label_ratios: list[float],
        device: str | None = None,
        library_size: int | None = 20000,
    ) -> None:
        """
        Conditional single-cell RNA-seq GAN using the conditioning method by concatenation.

        Parameters
        ----------
        genes_no
            Number of genes in the dataset.
        batch_size
            Training batch size.
        latent_dim
            Dimension of the latent space from which the noise vector is sampled.
        gen_layers
            List of integers corresponding to the number of neurons of each generator layer.
        crit_layers
            List of integers corresponding to the number of neurons of each critic layer.
        num_classes
            Number of classes in the dataset.
        label_ratios
            List containing the ratio of each class in the dataset.
        device
            Specifies to train on 'cpu' or 'cuda'. Only 'cuda' is supported for training the
            GAN but 'cpu' can be used for inference, by default "cuda" if torch.cuda.is_available() else"cpu".
        library_size
            Total number of counts per generated cell, by default 20000.
        """
        self.num_classes = num_classes

        super().__init__(
            genes_no,
            batch_size,
            latent_dim,
            gen_layers,
            crit_layers,
            device,
            library_size,
        )

        self.label_ratios = torch.nn.Buffer(
            torch.tensor(label_ratios, device=device), persistent=True
        )  # After super().__init__() to ensure self.device is set

    @staticmethod
    def sample_pseudo_labels(batch_size: int, cluster_ratios: "Tensor") -> "Tensor":
        """
        Randomly samples cluster labels following a multinomial distribution.

        Parameters
        ----------
        batch_size
            The number of samples to generate (normally equal to training batch size).
        cluster_ratios
            Tensor containing the parameters of the multinomial distribution
            (ex: Tensor([0.5, 0.3, 0.2]) for 3 clusters with occurence
            probabilities of  0.5, 0.3, and 0.2 for clusters 0, 1, and 2, respectively).

        Returns
        -------
        Tensor
            Tensor containing a batch of samples cluster labels.
        """
        cluster_ratios = 1 - cluster_ratios
        mn_logits = torch.tile(-torch.log(cluster_ratios), (batch_size, 1))
        labels = torch.multinomial(mn_logits, 1)

        return labels.flatten()
