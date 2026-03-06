from typing import TYPE_CHECKING

import torch

from .conditional_gan import ConditionalGANTrainer

if TYPE_CHECKING:
    from pathlib import Path

    from optuna import Trial
    from torch import Tensor
    
    from gans import ConditionalProjGAN

    from .dicts import GANTrainingArgs, SummaryArgs


class ConditionalProjGANTrainer(ConditionalGANTrainer):
    def __init__(
        self,
        gan: "ConditionalProjGAN",
        train_file: "Path",
        valid_file: "Path",
        training_args: "GANTrainingArgs",
        summary_args: "SummaryArgs",
        output_dir: "Path",
        trial: "Trial | None" = None,
    ) -> None:
        super().__init__(gan, train_file, valid_file, training_args, summary_args, output_dir, trial)
        self.gan = gan

    def _get_gradient(self, real: "Tensor", fake: "Tensor", real_labels: "Tensor") -> "Tensor":
        # Mix real and fake cells together
        epsilon = torch.rand(len(real), 1, device=self.gan.device)
        interpolates = real * epsilon + fake * (1 - epsilon)
        interpolates.requires_grad_(True)

        # Calculate the critic's scores on the mixed data
        critic_interpolates = self.modules["crit"](interpolates, real_labels)

        # Take the gradient of the scores with respect to the data
        gradient = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(critic_interpolates, device=self.gan.device),
            create_graph=True,
        )[0]
        return gradient  # noqa: RET504
    
    def _generator_step(self) -> tuple["Tensor", "Tensor", "Tensor"]:
        """
        Performs a forward pass of the generator and critic and computes the generator loss.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Generator's loss for the current batch, critic's scores on fake cells, and the generated fake cells.
        """
        fake_noise = self.gan.generate_noise(self.gan.batch_size, self.gan.latent_dim, device=self.gan.device)
        fake_labels = self.gan.sample_pseudo_labels(self.gan.batch_size, self.gan.label_ratios).to(self.gan.device)

        fake = self.gan.gen(fake_noise, fake_labels)
        crit_fake_pred = self.gan.crit(fake, fake_labels)

        gen_loss = self._generator_loss(crit_fake_pred)

        return gen_loss, crit_fake_pred, fake
    
    def _critic_step(self, real_cells: "Tensor", real_labels: "Tensor") -> tuple["Tensor", "Tensor", "Tensor"]:
        with torch.no_grad():
            fake_noise = self.gan.generate_noise(len(real_cells), self.gan.latent_dim, self.gan.device)
            fake_cells = self.modules["gen"](fake_noise, real_labels)

        crit_fake_pred = self.modules["crit"](fake_cells, real_labels)
        crit_real_pred = self.modules["crit"](real_cells, real_labels)

        return crit_fake_pred, crit_real_pred, fake_cells

