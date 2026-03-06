from typing import TYPE_CHECKING

import numpy as np
import torch

from networks.critic import ConditionalCritic
from networks.generator import ConditionalGenerator

from .conditional_gan import ConditionalGAN

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from torch import Tensor


class ConditionalProjGAN(ConditionalGAN):
    def _build_model(self) -> None:
        """Initializes the Generator and Critic."""
        self.gen = ConditionalGenerator(
            self.latent_dim,
            self.genes_no,
            self.num_classes,
            self.gen_layers,
            self.library_size,
        ).to(self.device)

        self.crit = ConditionalCritic(self.genes_no, self.crit_layers, self.num_classes).to(self.device)

    def _generate_gen_and_crit_data(self) -> tuple[tuple["Tensor", "Tensor"], tuple["Tensor", "Tensor"]]:
        """
        Generates a batch of noise and corresponding fake cells for logging the model graph.

        Returns
        -------
        tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]
            A batch of noise and corresponding labels, and a batch of generated fake cells and corresponding labels.
        """
        with torch.no_grad():
            gen_noise = self.generate_noise(self.batch_size, self.latent_dim, self.device)
            gen_labels = self.sample_pseudo_labels(self.batch_size, self.label_ratios).to(self.device)
            gen_inputs = (gen_noise, gen_labels)
            gen_cells = self.gen(gen_noise, gen_labels)
            crit_inputs = (gen_cells, gen_labels)
        return gen_inputs, crit_inputs
    
    def generate_cells(
        self,
        cells_no: int,
        checkpoint: "Path | None" = None,
        class_: int | None = None,
        **kwargs: "Any",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate cells from the Conditional GAN model.

        Parameters
        ----------
        cells_no : int
            Number of cells to generate.
        checkpoint : Path | None, optional
            Path to the saved trained model. Default is None.
        class_: int | None, optional
            Class of the cells to generate. If None, cells with the same ratio per class
            will be generated. Default is None.
        **kwargs: Any
            Additional keyword arguments to pass to the generator (not used here).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
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
                fake_cells.append(self.gen(noise, labels).cpu().detach().numpy())
                fake_labels.append(labels.cpu().detach().numpy())
        self.gen.train(was_training)

        return (
            np.concatenate(fake_cells)[:cells_no],
            np.concatenate(fake_labels)[:cells_no],
        )
