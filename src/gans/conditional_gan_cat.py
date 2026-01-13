from typing import TYPE_CHECKING

import numpy as np
import torch

from gans.conditional_gan import ConditionalGAN
from networks.critic import Critic
from networks.generator import Generator

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from torch import Tensor


class ConditionalCatGAN(ConditionalGAN):
    def __init__(
        self,
        genes_no: int,
        batch_size: int,
        latent_dim: int,
        gen_layers: list[int],
        crit_layers: list[int],
        num_classes: int,
        label_ratios: list[float],
        device: str | None = "cuda" if torch.cuda.is_available() else "cpu",
        library_size: int | None = 20000,
    ) -> None:
        """
        Conditional single-cell RNA-seq GAN using the conditioning method by concatenation.

        Parameters
        ----------
        genes_no : int
            Number of genes in the dataset.
        batch_size : int
            Training batch size.
        latent_dim : int
            Dimension of the latent space from which the noise vector is sampled.
        gen_layers : list[int]
            List of integers corresponding to the number of neurons of each generator layer.
        crit_layers : list[int]
            List of integers corresponding to the number of neurons of each critic layer.
        num_classes : int
            Number of classes in the dataset.
        label_ratios : list[float]
            List containing the ratio of each class in the dataset.
        device : str | None, optional
            Specifies to train on 'cpu' or 'cuda'. Only 'cuda' is supported for training the
            GAN but 'cpu' can be used for inference, by default "cuda" if torch.cuda.is_available() else"cpu".
        library_size : int | None, optional
            Total number of counts per generated cell, by default 20000.
        """

        self.num_classes = num_classes
        self.label_ratios = torch.tensor(label_ratios, device=device)

        super(ConditionalCatGAN, self).__init__(
            genes_no,
            batch_size,
            latent_dim,
            gen_layers,
            crit_layers,
            device,
            library_size,
        )

    def _build_model(self) -> None:
        """Initializes the Generator and Critic."""
        self.gen = Generator(
            self.latent_dim + self.num_classes,
            self.genes_no,
            self.gen_layers,
            self.library_size,
        ).to(self.device)
        self.crit = Critic(self.genes_no + self.num_classes, self.critic_layers).to(self.device)

    def _cat_one_hot_labels(self, cells: "Tensor", labels: "Tensor") -> "Tensor":
        """
        Concatenates one-hot encoded labels to a tensor.

        Parameters
        ----------
        cells : Tensor
            Tensor to which to concatenate one-hot encoded class labels.
        labels : Tensor
            Class labels to concatenate.

        Returns
        -------
        Tensor
            Tensor with one-hot encoded labels concatenated at the tail.
        """
        one_hot = torch.nn.functional.one_hot(labels, self.num_classes)
        return torch.cat((cells.float(), one_hot.float()), 1)

    def _critic_step(self, real_cells: "Tensor", real_labels: "Tensor") -> tuple["Tensor", "Tensor", "Tensor"]:
        """
        Performs a forward pass of the critic on real and fake cells.

        Parameters
        ----------
        real_cells : Tensor
            Tensor containing a batch of real cells.
        real_labels : Tensor
            Tensor containing the class labels of the real cells.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Critic's scores on fake cells, critic's scores on real cells, and the generated fake cells (generated without tracking gradients).
        """
        with torch.no_grad():
            fake_noise = self._generate_noise(len(real_cells), self.latent_dim, self.device)
            fake_cells = self.gen(self._cat_one_hot_labels(fake_noise, real_labels))

        crit_fake_pred = self.crit(self._cat_one_hot_labels(fake_cells, real_labels))
        crit_real_pred = self.crit(self._cat_one_hot_labels(real_cells, real_labels))

        return crit_fake_pred, crit_real_pred, fake_cells

    def _generator_step(self) -> tuple["Tensor", "Tensor", "Tensor"]:
        """
        Performs a forward pass of the generator and critic and computes the generator loss.

        Returns
        -------
        Tensor
            Generator's loss for the current batch, critic's scores on fake cells, and the generated fake cells.
        """
        fake_noise = self._generate_noise(self.batch_size, self.latent_dim, device=self.device)
        fake_labels = self._sample_pseudo_labels(self.batch_size, self.label_ratios).to(self.device)

        fake = self.gen(self._cat_one_hot_labels(fake_noise, fake_labels))
        crit_fake_pred = self.crit(self._cat_one_hot_labels(fake, fake_labels))

        gen_loss = self._generator_loss(crit_fake_pred)

        return gen_loss, crit_fake_pred, fake

    def generate_cells(
        self,
        cells_no: int,
        checkpoint: "Path | None" = None,
        class_: int | None = None,
        **kwargs: "Any",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Generate cells from the Conditional GAN model.

        Parameters
        ----------
        cells_no : int
            Number of cells to generate.
        checkpoint : Path | None
            Path to the saved trained model, by default None.
        class_: int | None = None
            Class of the cells to generate. If None, cells with the same ratio per class
            will be generated.
        kwargs : Any
            Additional keyword arguments (not used).

        Returns
        -------
        tuple[np.ndarray, np.ndarray | None]
            Gene expression matrix of generated cells and their corresponding class labels.
        """
        if checkpoint is not None:
            self._load(checkpoint)

        batch_no = int(np.ceil(cells_no / self.batch_size))
        fake_cells = []
        fake_labels = []
        for _ in range(batch_no):
            noise = self._generate_noise(self.batch_size, self.latent_dim, self.device)
            if class_ is None:
                labels = self._sample_pseudo_labels(self.batch_size, self.label_ratios).to(self.device)
            else:
                label_ratios = torch.zeros(self.num_classes).to(self.device)
                label_ratios[class_] = 0.99
                labels = self._sample_pseudo_labels(self.batch_size, label_ratios).to(self.device)
            fake_cells.append(self.gen(self._cat_one_hot_labels(noise, labels)).cpu().detach().numpy())
            fake_labels.append(labels.cpu().detach().numpy())

        return (
            np.concatenate(fake_cells)[:cells_no],
            np.concatenate(fake_labels)[:cells_no],
        )
