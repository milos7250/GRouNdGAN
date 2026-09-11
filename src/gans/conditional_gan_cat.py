from typing import TYPE_CHECKING

import numpy as np
import torch

from networks.critic import Critic
from networks.generator import Generator

from .conditional_gan import ConditionalGAN

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from torch import Tensor


class ConditionalCatGAN(ConditionalGAN):
    def _build_model(self) -> None:
        """Initializes the Generator and Critic."""
        self.gen = Generator(
            self.latent_dim + self.num_classes,
            self.genes_no,
            self.gen_layers,
            self.library_size,
        ).to(self.device)
        self.crit = Critic(self.genes_no + self.num_classes, self.crit_layers).to(self.device)

    def cat_one_hot_labels(self, cells: "Tensor", labels: "Tensor") -> "Tensor":
        """
        Concatenates one-hot encoded labels to a tensor.

        Parameters
        ----------
        cells
            Tensor to which to concatenate one-hot encoded class labels.
        labels
            Class labels to concatenate.

        Returns
        -------
        Tensor
            Tensor with one-hot encoded labels concatenated at the tail.
        """
        one_hot = torch.nn.functional.one_hot(labels, self.num_classes)
        return torch.cat((cells.float(), one_hot.float()), 1)

    def _generate_gen_and_crit_data(self) -> tuple["Tensor", "Tensor"]:
        """
        Generates a batch of noise and corresponding fake cells for logging the model graph.

        Returns
        -------
        tuple[Tensor, Tensor]
            A batch of noise and the corresponding generated fake cells.
        """
        with torch.no_grad():
            gen_noise = self.generate_noise(self.batch_size, self.latent_dim, self.device)
            gen_labels = self.sample_pseudo_labels(self.batch_size, self.label_ratios).to(self.device)
            gen_data = self.cat_one_hot_labels(gen_noise, gen_labels)
            gen_cells = self.gen(gen_data)
            crit_data = self.cat_one_hot_labels(gen_cells, gen_labels)
        return gen_data, crit_data

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
        cells_no
            Number of cells to generate.
        checkpoint
            Path to the saved trained model, by default None.
        class_
            Class of the cells to generate. If None, cells with the same ratio per class
            will be generated.
        kwargs
            Additional keyword arguments (not used).

        Returns
        -------
        tuple[np.ndarray, np.ndarray | None]
            Gene expression matrix of generated cells and their corresponding class labels.
        """
        if checkpoint is not None:
            self.load(checkpoint)

        batch_no = int(np.ceil(cells_no / self.batch_size))
        was_training = self.gen.training
        fake_cells = []
        fake_labels = []
        self.gen.eval()
        with torch.inference_mode():
            for _ in range(batch_no):
                noise = self.generate_noise(self.batch_size, self.latent_dim, self.device)
                if class_ is None:
                    labels = self.sample_pseudo_labels(self.batch_size, self.label_ratios).to(self.device)
                else:
                    label_ratios = torch.zeros(self.num_classes).to(self.device)
                    label_ratios[class_] = 0.99
                    labels = self.sample_pseudo_labels(self.batch_size, label_ratios).to(self.device)
                fake_cells.append(self.gen(self.cat_one_hot_labels(noise, labels)).cpu().detach().numpy())
                fake_labels.append(labels.cpu().detach().numpy())
        self.gen.train(was_training)

        return (
            np.concatenate(fake_cells)[:cells_no],
            np.concatenate(fake_labels)[:cells_no],
        )
