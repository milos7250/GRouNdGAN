import os
from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING
from warnings import catch_warnings, filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch._inductor.select_algorithm
from optuna import TrialPruned
from torch.cuda import empty_cache as empty_cuda_cache
from torch.cuda import is_available as is_cuda_available
from torch.distributed import barrier  # pyright: ignore[reportUnknownVariableType]
from torch.distributed import is_initialized as is_ddp_initialized
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm.rich import tqdm
from umap import UMAP

from evaluation.data_quality import compute_RF_AUROC
from loggers import setup_logger, tqdm_logging_redirect
from networks.critic import Critic
from networks.generator import Generator
from sc_dataset import get_loader

if TYPE_CHECKING:
    from typing import Any

    from optuna import Trial
    from torch import Tensor

    from sc_dataset import SCDataLoader


class GAN:
    def __init__(
        self,
        genes_no: int,
        batch_size: int,
        latent_dim: int,
        gen_layers: list[int],
        crit_layers: list[int],
        device: str | None = None,
        library_size: int | None = 20000,
    ) -> None:
        """
        Non-conditional single-cell RNA-seq GAN.

        Parameters
        ----------
        genes_no : int
            Number of genes in the dataset.
        batch_size : int
            Training batch size.
        latent_dim : int
            Dimension of the latent space from which the noise vector is sampled.
        gen_layers : list[int]
            list of integers corresponding to the number of neurons of each generator layer.
        crit_layers : list[int]
            list of integers corresponding to the number of neurons of each critic layer.
        device : str | None, optional
            Specifies to train on 'cpu' or 'cuda'. Only 'cuda' is supported for training the
            GAN but 'cpu' can be used for inference, by default "cuda" if is_cuda_available() else"cpu".
        library_size : int | None, optional
            Total number of counts per generated cell, by default 20000.
        """
        empty_cuda_cache()

        self.genes_no = genes_no
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.gen_layers = gen_layers
        self.critic_layers = crit_layers
        self.device = device if device else ("cuda" if is_cuda_available() else "cpu")
        self.library_size = library_size

        self._build_model()

        self.step: int = 0
        self.gen_opt: Optimizer | None = None
        self.crit_opt: Optimizer | None = None
        self.gen_lr_scheduler: ExponentialLR | SequentialLR | None = None
        self.crit_lr_scheduler: ExponentialLR | SequentialLR | None = None
        self.umap: UMAP | None = None
        self.real_embedding: np.ndarray | None = None

    @staticmethod
    def _generate_noise(batch_size: int, latent_dim: int, device: str) -> "Tensor":
        """
        Function for creating noise vectors: Given the dimensions (batch_size, latent_dim).

        Parameters
        ----------
        batch_size : int
            The number of samples to generate (normally equal to training batch size).
        latent_dim : int
            Dimension of the latent space to sample from.
        device : str
            The device type.

        Returns
        -------
        Tensor
            A tensor filled with random numbers from the standard normal distribution.
        """
        return torch.randn(batch_size, latent_dim, device=device)

    @staticmethod
    def _set_exponential_lr(
        optimizer: Optimizer,
        alpha_0: float,
        alpha_final: float,
        max_steps: int,
        warmup_percent: float = 0.05,
    ) -> ExponentialLR | SequentialLR:
        """
        Sets up exponentially decaying learning rate scheduler to be used
        with the optimizer.

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer for which to create an exponential learning rate scheduler.
        alpha_0 : float
            Initial learning rate.
        alpha_final : float
            Final learning rate.
        max_steps : int
            Total number of training steps. When current_step=max_steps, alpha_final
            will be set as the learning rate.
        warmup_percent : float, optional
            Percentage of total steps to use for learning rate warmup (default: 0.0).

        Returns
        -------
        ExponentialLR | SequentialLR
            Learning rate scheduler. Call the step() function on this
            scheduler in the training loop.
        """
        warmup_steps = int(max_steps * warmup_percent)
        exponential_steps = max_steps - warmup_steps

        # Find the decay rate of the exponential learning rate
        decay_rate = (alpha_final / alpha_0) ** (1 / exponential_steps)
        exponential_sched = ExponentialLR(optimizer, gamma=decay_rate)

        if warmup_steps > 0:
            warmup_sched = LinearLR(
                optimizer=optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            return SequentialLR(optimizer, [warmup_sched, exponential_sched], milestones=[warmup_steps])
        else:
            return exponential_sched

    @staticmethod
    @torch.compile(fullgraph=True)  # pyright: ignore[reportUnknownMemberType]
    def _critic_loss(
        crit_fake_pred: "Tensor",
        crit_real_pred: "Tensor",
        gp: "Tensor",
        c_lambda: float,
    ) -> "Tensor":
        """
        Compute critic's loss given the its scores on real and fake cells,
        the gradient penalty, and gradient penalty regularization hyper-parameter.

        Parameters
        ----------
        crit_fake_pred : Tensor
            Critic's score on fake cells.
        crit_real_pred : Tensor
            Critic's score on real cells.
        gp : Tensor
            Unweighted gradient penalty
        c_lambda : float
            Regularization hyper-parameter to be used with the gradient penalty
            in the WGAN loss.

        Returns
        -------
        Tensor
            Critic's loss for the current batch.
        """
        return torch.nansum(torch.stack([torch.mean(crit_fake_pred) - torch.mean(crit_real_pred), c_lambda * gp]))

    @staticmethod
    @torch.compile(fullgraph=True)  # pyright: ignore[reportUnknownMemberType]
    def _generator_loss(crit_fake_pred: "Tensor") -> "Tensor":
        """
        Compute the generator loss from the critic's score of the generated cells.

        Parameters
        ----------
        crit_fake_pred : Tensor
            The critic's score on fake generated cells.

        Returns
        -------
        Tensor
            Generator's loss value for the current batch.
        """
        return -1.0 * torch.mean(crit_fake_pred)

    @torch.compiler.set_stance("force_eager")  # gradients not supported in compiled mode
    def _get_gradient(
        self,
        real: "Tensor",
        fake: "Tensor",
    ) -> "Tensor":
        """
        Compute the gradient of the critic's scores with respect to interpolations
        of real and fake cells.

        Parameters
        ----------
        real : Tensor
            A batch of real cells.
        fake : Tensor
            A batch of fake cells.
        epsilon : Tensor
            A vector of the uniformly random proportions of real/fake per interpolated cells.

        Returns
        -------
        Tensor
            Gradient of the critic's score with respect to interpolated data.
        """

        # Mix real and fake cells together
        epsilon = torch.rand(len(real), 1, device=self.device)
        interpolates = real * epsilon + fake * (1 - epsilon)
        interpolates.requires_grad_(True)

        # Calculate the critic's scores on the mixed data
        critic_interpolates = self.crit(interpolates)

        # Take the gradient of the scores with respect to the data
        gradient = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(critic_interpolates, device=self.device),
            create_graph=True,
        )[0]
        return gradient

    @staticmethod
    def _gradient_penalty(gradient: "Tensor") -> "Tensor":
        """
        Compute the gradient penalty given a gradient.

        Parameters
        ----------
        gradient : Tensor
            The gradient of the critic's score with respect to
            the interpolated data.

        Returns
        -------
        Tensor
            Gradient penalty of the given gradient.
        """
        gradient = gradient.view(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=1, dtype=gradient.dtype)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        return torch.mean((gradient_norm - 1) ** 2)  # pyright: ignore[reportUnknownArgumentType]

    def generate_cells(
        self,
        cells_no: int,
        checkpoint: Path | None = None,
        **kwargs: "Any",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Generate cells from the GAN model.

        Parameters
        ----------
        cells_no : int
            Number of cells to generate.
        checkpoint : Path | None, optional
            Path to the saved trained model, by default None.
        kwargs : Any
            Additional keyword arguments (not used).

        Returns
        -------
        tuple[np.ndarray, np.ndarray | None]
            Tuple of Gene expression matrix of generated cells and None (dummy labels).
        """
        if checkpoint:
            self._load(checkpoint)

        # find how many batches to generate
        batch_no = int(np.ceil(cells_no / self.batch_size))

        fake_cells = []
        was_training = self.gen.training
        self.gen.eval()
        with torch.no_grad():
            for _ in range(batch_no):
                noise = self._generate_noise(self.batch_size, self.latent_dim, self.device)
                fake_cells.append(self.gen(noise).cpu().detach().numpy())
        self.gen.train(was_training)

        return np.concatenate(fake_cells)[:cells_no], None

    def _save(self, path: Path) -> None:
        """
        Saves the model.

        Parameters
        ----------
        path : Path
            Directory to save the model.
        """

        if is_ddp_initialized() and os.environ.get("RANK", "0") != "0":
            return

        output_dir = path / "checkpoints"
        output_dir.mkdir(parents=True, exist_ok=True)

        if is_ddp_initialized():
            state_dict = {
                "generator_state_dict": self.gen.module.state_dict(),  # pyright: ignore[reportAttributeAccessIssue]
                "critic_state_dict": self.crit.module.state_dict(),  # pyright: ignore[reportAttributeAccessIssue]
            }
        else:
            state_dict = {
                "generator_state_dict": self.gen.state_dict(),
                "critic_state_dict": self.crit.state_dict(),
            }

        torch.save(
            state_dict
            | {
                "step": self.step,
                "generator_optimizer_state_dict": self.gen_opt.state_dict() if self.gen_opt else None,
                "critic_optimizer_state_dict": self.crit_opt.state_dict() if self.crit_opt else None,
                "generator_lr_scheduler": self.gen_lr_scheduler.state_dict() if self.gen_lr_scheduler else None,
                "critic_lr_scheduler": self.crit_lr_scheduler.state_dict() if self.crit_lr_scheduler else None,
            },
            output_dir / f"step_{self.step}.pth",
        )

    def _load(
        self,
        path: Path,
        mode: str | None = "inference",
    ) -> None:
        """
        Loads a saved model (.pth file).

        Parameters
        ----------
        path : Path
            Path to the saved model.
        mode : str | None, optional
            Specify if the loaded model is used for 'inference' or 'training', by default "inference".

        Raises
        ------
        ValueError
            If a mode other than 'inference' or 'training' is specified.
        RuntimeError
            If training mode is specified but the optimizers or learning rate schedulers are not initialized.
        """

        checkpoint = torch.load(path, map_location=torch.device(self.device))

        self.gen.load_state_dict(checkpoint["generator_state_dict"])
        self.crit.load_state_dict(checkpoint["critic_state_dict"])

        if mode == "inference":
            self.gen.eval()
            self.crit.eval()

        elif mode == "training":
            self.gen.train()
            self.crit.train()

            self.step = checkpoint["step"] + 1

            if not self.gen_opt or not self.crit_opt or not self.gen_lr_scheduler or not self.crit_lr_scheduler:
                raise RuntimeError(
                    "Generator and critic optimizers and learning rate schedulers must be initialized "
                    "before loading in training mode."
                )

            self.gen_opt.load_state_dict(checkpoint["generator_optimizer_state_dict"])
            self.crit_opt.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.gen_lr_scheduler.load_state_dict(checkpoint["generator_lr_scheduler"])
            self.crit_lr_scheduler.load_state_dict(checkpoint["critic_lr_scheduler"])

        else:
            raise ValueError("mode should be 'inference' or 'training'")

    def _build_model(self) -> None:
        """Instantiates the Generator and Critic."""
        self.gen = Generator(self.latent_dim, self.genes_no, self.gen_layers, self.library_size).to(self.device)
        self.crit = Critic(self.genes_no, self.critic_layers).to(self.device)

    def _get_loaders(
        self,
        train_file: Path,
        validation_file: Path,
    ) -> tuple["SCDataLoader", "SCDataLoader"]:
        """
        Gets training and validation DataLoaders for training.

        Parameters
        ----------
        train_file : Path
            Path to training files.
        validation_file : Path
            Path to validation files.

        Returns
        -------
        tuple[SCDataLoader, SCDataLoader]
            Train and Validation Dataloaders.
        """
        return get_loader(train_file, self.batch_size, shuffle=True, drop_last=True), get_loader(
            validation_file,
            batch_size=2000,  # batch size for validation chosen to balance speed and memory, can be adjusted
            shuffle=False,
            drop_last=False,
        )

    def log_tensorboard_graph(self, output_dir: Path) -> None:
        """
        Adds the model graph to TensorBoard.

        Parameters
        ----------
        output_dir : Path
            Directory to save the tfevents.
        """
        if is_ddp_initialized():
            # Only log on the master node
            if os.environ.get("RANK", "0") != "0":
                return

        was_training = (self.gen.training, self.crit.training)
        self.gen.eval()
        self.crit.eval()

        with torch.no_grad():
            gen_data = self._generate_noise(self.batch_size, self.latent_dim, self.device)
            crit_data = self.gen(gen_data)

        with catch_warnings():
            filterwarnings("ignore", message=".*Trace had nondeterministic nodes.*")
            filterwarnings(
                "ignore",
                message=".*the traced function does not match the corresponding output of the Python function.*",
            )
            filterwarnings(
                "ignore", message=r".*The \.grad attribute of a Tensor that is not a leaf Tensor is being accessed*"
            )
            with SummaryWriter(f"{output_dir}/TensorBoard/model/generator") as w:
                w.add_graph(self.gen, gen_data, use_strict_trace=False)
            with SummaryWriter(f"{output_dir}/TensorBoard/model/critic") as w:
                w.add_graph(self.crit, crit_data, use_strict_trace=False)

        self.gen.train(was_training[0])
        self.crit.train(was_training[1])

    def _update_tensorboard(
        self,
        loss_dict: dict[str, float],
        output_dir: Path,
        summary_writer: SummaryWriter | None = None,
    ) -> None:
        """
        Updates the TensorBoard summary logs.

        Parameters
        ----------
        loss_dict : dict[str, float]
            Dictionary containing the losses to log.
        output_dir : Path
            Directory to save the tfevents.
        summary_writer : SummaryWriter | None, optional
            SummaryWriter instance to use for logging. If None, a new SummaryWriter
            will be created, by default None.
        """

        # Only update on the master node
        if is_ddp_initialized() and os.environ.get("RANK", "0") != "0":
            return

        with (
            summary_writer
            if summary_writer
            else SummaryWriter(output_dir / "TensorBoard/", filename_suffix=f".step{self.step}") as w
        ):
            for key, value in loss_dict.items():
                w.add_scalar(key, value, self.step)

    def _generate_umap_plots(
        self,
        valid_loader: "SCDataLoader",
        output_dir: Path,
        max_cells: int | None = None,
    ) -> None:
        """
        Generates UMAP plots during training.

        Parameters
        ----------
        valid_loader : SCDataLoader
            Validation set DataLoader.
        output_dir : Path
            Directory to save the UMAP plots.
        max_cells : int | None
            Maximum number of cells to use from the validation set for UMAP
        """
        no_of_cells = min(max_cells, len(valid_loader.dataset)) if max_cells else len(valid_loader.dataset)

        fake_cells = self.generate_cells(no_of_cells)[0]

        # Only generate on the master node,
        if is_ddp_initialized() and os.environ.get("RANK", "0") != "0":
            return

        umap_path = output_dir / "UMAP"
        umap_path.mkdir(parents=True, exist_ok=True)

        if self.umap is None or self.real_embedding is None:
            self.umap = UMAP(random_state=42, min_dist=0.0, n_jobs=1)
            self.umap.fit(valid_loader.dataset.cells)  # ensure UMAP is fitted only once to preserve comparability
            self.real_embedding = np.array(self.umap.transform(valid_loader.dataset.cells))

        real_embedding = self.real_embedding
        fake_embedding = np.array(self.umap.transform(fake_cells))
        extent = np.array([
            [
                min(min(real_embedding[:, 0]), min(fake_embedding[:, 0])),
                max(max(real_embedding[:, 0]), max(fake_embedding[:, 0])),
            ],
            [
                min(min(real_embedding[:, 1]), min(fake_embedding[:, 1])),
                max(max(real_embedding[:, 1]), max(fake_embedding[:, 1])),
            ],
        ])
        margin = np.array(extent[:, 1] - extent[:, 0]) * 0.05  # 5% margin
        extent[:, 0] -= margin[0]
        extent[:, 1] += margin[1]

        plt.clf()
        scatter_fig = plt.figure(figsize=(5, 5))

        plt.scatter(
            real_embedding[:, 0],
            real_embedding[:, 1],
            c="blue",
            label="real",
            alpha=0.1,
        )

        plt.scatter(
            fake_embedding[:, 0],
            fake_embedding[:, 1],
            c="red",
            label="generated",
            alpha=0.1,
        )

        plt.grid(True)
        plt.xlim(extent[0, 0], extent[0, 1])
        plt.ylim(extent[1, 0], extent[1, 1])
        plt.title(f"UMAP Projection of Real and Generated Cells at Step {self.step}")
        plt.legend(loc="lower left", numpoints=1, ncol=2, fontsize=8, bbox_to_anchor=(0, 0))

        plt.savefig(umap_path / f"step_{self.step}_scatter.jpg")

        hexbin_fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].hexbin(
            real_embedding[:, 0], real_embedding[:, 1], mincnt=1, linewidths=0.0, extent=extent.flatten(), cmap="Reds"
        )
        ax[0].set_title("Real Cells")
        plt.colorbar(ax[0].collections[0], ax=ax[0])

        ax[1].hexbin(
            fake_embedding[:, 0], fake_embedding[:, 1], mincnt=1, linewidths=0.0, extent=extent.flatten(), cmap="Reds"
        )
        ax[1].set_title("Generated Cells")
        plt.colorbar(ax[1].collections[0], ax=ax[1])
        plt.suptitle(f"UMAP Histograms at Step {self.step}")

        plt.savefig(umap_path / f"step_{self.step}_hexbin.jpg")

        H_real, xedges, yedges = np.histogram2d(real_embedding[:, 0], real_embedding[:, 1], bins=80, range=extent)
        H_fake, _, _ = np.histogram2d(fake_embedding[:, 0], fake_embedding[:, 1], bins=80, range=extent)
        H_real[H_real == 0] = np.nan
        H_rel = H_real / (H_real + H_fake)  # relative density difference
        X, Y = np.meshgrid(xedges, yedges)
        v_bound = np.nanmax(np.abs(H_rel - 0.5))

        hist_diff_fig = plt.figure(figsize=(5, 5))
        plt.pcolormesh(X, Y, H_rel.T, shading="auto", cmap="coolwarm", vmin=0.5 - v_bound, vmax=0.5 + v_bound)

        plt.title(f"UMAP Histogram Relative Abundance of Real Cells at Step {self.step}")

        plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)  # shrink fig so cbar is visible
        # make new ax object for the cbar
        cbar_ax = hist_diff_fig.add_axes((0.87, 0.15, 0.02, 0.7))  # x, y, width, height
        plt.colorbar(cax=cbar_ax)

        plt.savefig(umap_path / f"step_{self.step}_hist_real_relative.jpg")

        # TODO: REMOVE
        # with open(umap_path / f"step_{self.step}-embeddings.pkl", "wb") as f:
        #     import pickle

        #     pickle.dump({"real": real_embedding, "fake": fake_embedding}, f)

        with SummaryWriter(output_dir / "TensorBoard/UMAP", filename_suffix=f".step{self.step}") as w:
            w.add_figure("UMAP Scatter", scatter_fig, self.step)
            w.add_figure("UMAP Histogram", hexbin_fig, self.step)
            w.add_figure("UMAP Histogram Relative Abundance of Real Cells", hist_diff_fig, self.step)

        plt.close("all")

    def _get_validation_loss(
        self,
        valid_loader: "SCDataLoader",
        c_lambda: float,
    ) -> dict[str, float]:
        """
        Computes the validation loss over the entire validation set.

        Parameters
        ----------
        valid_loader : SCDataLoader
            Validation set DataLoader.
        c_lambda : float
            Regularization hyper-parameter for gradient penalty.

        Returns
        -------
        dict[str, float]
            Dictionary containing average generator loss, critic loss and gradient penalty
            over the entire validation set.
        """
        total_gen_total_loss = torch.tensor(0.0, device=self.device)
        total_crit_real_loss = torch.tensor(0.0, device=self.device)
        total_crit_fake_loss = torch.tensor(0.0, device=self.device)
        total_crit_gp_loss = torch.tensor(0.0, device=self.device)
        total_crit_total_loss = torch.tensor(0.0, device=self.device)
        total_batches = torch.tensor(0, device=self.device)

        was_training = (self.gen.training, self.crit.training)
        self.gen.eval(), self.crit.eval()  # pyright: ignore[reportUnusedExpression]

        for real_cells, real_labels in valid_loader:
            with torch.no_grad():
                real_cells = real_cells.to(self.device)
                crit_fake_pred, crit_real_pred, fake_cells = self._critic_step(real_cells, real_labels)

            gradient = self._get_gradient(real_cells, fake_cells)
            gp = self._gradient_penalty(gradient)

            with torch.no_grad():
                gen_loss = self._generator_loss(crit_fake_pred)
                crit_loss = self._critic_loss(crit_fake_pred, crit_real_pred, gp, c_lambda=c_lambda)

                total_gen_total_loss += gen_loss.detach().clone()
                total_crit_real_loss += (-crit_real_pred.mean()).detach().clone()
                total_crit_fake_loss += (crit_fake_pred.mean()).detach().clone()
                total_crit_gp_loss += c_lambda * gp.detach().clone()
                total_crit_total_loss += crit_loss.detach().clone()
                total_batches += 1

        avg_gen_total_loss = total_gen_total_loss / total_batches
        avg_crit_real_loss = total_crit_real_loss / total_batches
        avg_crit_fake_loss = total_crit_fake_loss / total_batches
        avg_crit_gp_loss = total_crit_gp_loss / total_batches
        avg_crit_total_loss = total_crit_total_loss / total_batches

        self.gen.train(was_training[0]), self.crit.train(was_training[1])  # pyright: ignore[reportUnusedExpression]
        return {
            "val_gen_loss": avg_gen_total_loss.item(),
            "val_gen_total_loss": avg_gen_total_loss.item(),
            "val_crit_real_loss": avg_crit_real_loss.item(),
            "val_crit_fake_loss": avg_crit_fake_loss.item(),
            "val_crit_gp_loss": avg_crit_gp_loss.item(),
            "val_crit_total_loss": avg_crit_total_loss.item(),
            "val_total_loss": (avg_gen_total_loss + avg_crit_total_loss).item(),
        }

    def _critic_step(self, real_cells: "Tensor", real_labels: "Tensor") -> tuple["Tensor", "Tensor", "Tensor"]:
        """
        Performs a forward pass of the critic on real and fake cells.

        Parameters
        ----------
        real_cells : Tensor
            Tensor containing a batch of real cells.
        real_labels : Tensor
            Tensor containing a batch of real labels (corresponding to real_cells). Not used in non-conditional GAN.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Critic's scores on fake cells, critic's scores on real cells, and the generated fake cells (generated without tracking gradients).
        """
        with torch.no_grad():
            fake_noise = self._generate_noise(len(real_cells), self.latent_dim, self.device)
            fake_cells = self.gen(fake_noise)

        crit_fake_pred = self.crit(fake_cells)
        crit_real_pred = self.crit(real_cells)

        return crit_fake_pred, crit_real_pred, fake_cells

    def _train_critic(self, real_cells: "Tensor", real_labels: "Tensor", c_lambda: float) -> dict[str, float]:
        """
        Trains the critic for one iteration.

        Parameters
        ----------
        real_cells : Tensor
            Tensor containing a batch of real cells.
        real_labels : Tensor
            Tensor containing a batch of real labels (corresponding to real_cells).
        c_lambda : float
            Regularization hyper-parameter for gradient penalty.

        Returns
        -------
        dict[str, float]
            The computed critic loss and gradient penalty.

        Raises
        ------
        RuntimeError
            If the critic optimizer is not initialized.
        """
        if self.crit_opt is None:
            raise RuntimeError("Critic optimizer is not initialized.")

        self.crit_opt.zero_grad(set_to_none=True)

        crit_fake_pred, crit_real_pred, fake_cells = self._critic_step(real_cells, real_labels)
        gradient = self._get_gradient(real_cells, fake_cells.clone())
        gp = self._gradient_penalty(gradient)

        total_loss = self._critic_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)
        losses = {
            "crit_fake_loss": crit_fake_pred.mean().item(),
            "crit_real_loss": -crit_real_pred.mean().item(),
            "crit_gp_loss": c_lambda * gp.item(),
            "crit_total_loss": total_loss.item(),
        }

        # Update gradients
        total_loss.backward()

        # Update optimizer
        self.crit_opt.step()

        return losses

    def _generator_step(self) -> tuple["Tensor", "Tensor", "Tensor"]:
        """
        Performs a forward pass of the generator and critic and computes the generator loss.

        Returns
        -------
        Tensor
            Generator's loss for the current batch, critic's scores on fake cells, and the generated fake cells.
        """
        fake_noise = self._generate_noise(self.batch_size, self.latent_dim, device=self.device)

        fake = self.gen(fake_noise)
        crit_fake_pred = self.crit(fake)

        gen_loss = self._generator_loss(crit_fake_pred)

        return gen_loss, crit_fake_pred, fake

    def _train_generator(self) -> dict[str, float]:
        """
        Trains the generator for one iteration.

        Returns
        -------
        dict[str, float]
            Dictionary containing only 1 item, the generator loss.

        Raises
        ------
        RuntimeError
            If the generator optimizer is not initialized.
        """
        if self.gen_opt is None:
            raise RuntimeError("Generator optimizer is not initialized.")

        self.gen_opt.zero_grad(set_to_none=True)

        gen_loss, _, _ = self._generator_step()
        losses = {"gen_loss": gen_loss.item(), "gen_total_loss": gen_loss.item()}
        gen_loss.backward()

        # Update weights
        self.gen_opt.step()

        return losses

    def _training_step(
        self, real_cells: "Tensor", real_labels: "Tensor", critic_iter: int, c_lambda: float
    ) -> dict[str, float]:
        """
        Performs one training step: multiple critic updates followed by one generator update.

        Parameters
        ----------
        real_cells : Tensor
            Tensor containing a batch of real cells.
        real_labels : Tensor
            Tensor containing a batch of real labels (corresponding to real_cells). Not used in non-conditional GAN.
        critic_iter : int
            Number of training iterations of the critic for each iteration on the generator.
        c_lambda : float
            Regularization hyper-parameter for gradient penalty.

        Returns
        -------
        dict[str, float]
            Dictionary containing the computed losses.

        Raises
        ------
        RuntimeError
            If the critic or generator learning rate schedulers are not initialized.
        """
        if self.crit_lr_scheduler is None or self.gen_lr_scheduler is None:
            raise RuntimeError("Learning rate schedulers are not initialized.")

        losses = {}
        if self.step != 0:
            crit_losses = []
            for _ in range(critic_iter):
                torch.compiler.cudagraph_mark_step_begin()
                crit_losses.append(self._train_critic(real_cells, real_labels, c_lambda))

            losses |= {k: float(np.array([dic[k] for dic in crit_losses]).mean()) for k in crit_losses[0].keys()}
            # Update learning rate
            self.crit_lr_scheduler.step()
        else:
            losses |= {
                "crit_total_loss": np.inf,
                "crit_fake_loss": np.inf,
                "crit_real_loss": np.inf,
                "crit_gp_loss": np.inf,
            }

        torch.compiler.cudagraph_mark_step_begin()
        losses |= self._train_generator()
        self.gen_lr_scheduler.step()

        return losses

    def train(
        self,
        *,
        train_files: Path,
        valid_files: Path,
        critic_iter: int,
        max_steps: int,
        c_lambda: float,
        beta1: float,
        beta2: float,
        gen_alpha_0: float,
        gen_alpha_final: float,
        crit_alpha_0: float,
        crit_alpha_final: float,
        checkpoint: Path | None = None,
        output_dir: Path = Path("output"),
        summary_freq: int = 5000,
        plt_freq: int = 10000,
        save_freq: int = 10000,
        rf_auroc_freq: int = 0,
        trial: "Trial | None" = None,
        **kwargs: "Any",
    ) -> float:
        """
        Method for training the GAN.

        Parameters
        ----------
        train_files : str
            Path to training set files (TFrecords supported for now).
        valid_files : str
            Path to validation set files (TFrecords supported for now).
        critic_iter : int
            Number of training iterations of the critic for each iteration on the generator.
        max_steps : int
            Maximum number of steps to train the GAN.
        c_lambda : float
            Regularization hyper-parameter for gradient penalty.
        beta1 : float
            Coefficients used for computing running averages of gradient in the optimizer.
        beta2 : float
            Coefficient used for computing running averages of gradient squares in the optimizer.
        gen_alpha_0 : float
            Generator's initial learning rate value.
        gen_alpha_final : float
            Generator's final learning rate value.
        crit_alpha_0 : float
            Critic's initial learning rate value.
        crit_alpha_final : float
            Critic's final learning rate value.
        checkpoint : Path | None, optional
            Path to a trained model; if specified, the checkpoint is be used to resume training, by default None.
        output_dir : str, optional
            Directory to which plots, tfevents, and checkpoints will be saved, by default "output".
        summary_freq : int | None, optional
            Period between summary logs to TensorBoard, by default 5000. Set to 0 to disable.
        plt_freq : int | None, optional
            Period between UMAP plots, by default 10000. Set to 0 to disable.
        save_freq : int | None, optional
            Period between saves of the model, by default 10000. Set to 0 to disable.
        rf_auroc_freq : int | None, optional
            Period between random forest AUROC calculations, by default 0 (disabled).
        trial : Trial | None, optional
            Optuna trial object for hyperparameter optimization, by default None.
        **kwargs : Any
            Additional keyword arguments (not used).

        Returns
        -------
        float
            The final random forest AUROC score if rf_auroc_freq > 0, else the total validation loss.
        """
        # Configure logger
        logger = setup_logger("gan.train")
        if kwargs:
            logger.warning(f"Unused arguments passed to gan.train(): {kwargs}")

        def should_run(freq: int) -> bool:
            return (freq > 0 and self.step % freq == 0 and self.step > 0) or (self.step == max_steps)

        loader, valid_loader = self._get_loaders(train_files, valid_files)
        loader_gen = iter(loader)

        # Instantiate optimizers
        self.gen_opt = AdamW(
            filter(lambda p: p.requires_grad, self.gen.parameters()),
            lr=torch.tensor(gen_alpha_0, device=self.device),
            betas=(beta1, beta2),
            amsgrad=True,
            fused=True,
        )

        self.crit_opt = AdamW(
            self.crit.parameters(),
            lr=torch.tensor(crit_alpha_0, device=self.device),
            betas=(beta1, beta2),
            amsgrad=True,
            fused=True,
        )

        # Exponential Learning Rate
        self.gen_lr_scheduler = self._set_exponential_lr(self.gen_opt, gen_alpha_0, gen_alpha_final, max_steps, 0.05)
        self.crit_lr_scheduler = self._set_exponential_lr(
            self.crit_opt, crit_alpha_0, crit_alpha_final, max_steps, 0.05
        )

        if checkpoint is not None:
            if not Path(checkpoint).exists():
                logger.warning(f"Checkpoint {checkpoint} does not exist.")
            else:
                self._load(checkpoint, mode="training")
                return 0.0

        self.gen.train()
        self.crit.train()

        logger.info("Saving model graph...")
        with torch.compiler.set_stance("force_eager"):
            self.log_tensorboard_graph(output_dir)

        torch._inductor.select_algorithm.PRINT_AUTOTUNE = False  # to suppress autotune printing
        if is_ddp_initialized():
            self.gen = DDP(torch.compile(self.gen, fullgraph=True))
            self.crit = DDP(torch.compile(self.crit, fullgraph=True))
        else:
            self._generator_step = torch.compile(self._generator_step, fullgraph=True)
            self._critic_step = torch.compile(self._critic_step, fullgraph=True)

        if self.device == "cpu":
            logger.warning("Training on CPU is not supported and will be very slow.")

        # Main training loop
        losses: list[dict[str, float]] = []
        loss_dict: dict[str, float] = {}
        rf_auroc = 1.0
        summary_writer = SummaryWriter(output_dir / "TensorBoard/")
        torch.set_float32_matmul_precision("high")
        logger.info("Starting training...")
        with (
            tqdm_logging_redirect(loggers=[logger], tqdm_class=tqdm, desc="Training GAN", total=max_steps) as pbar,
            # torch.profiler.profile(
            #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            #     schedule=torch.profiler.schedule(wait=10, warmup=10, active=3, repeat=1),
            #     profile_memory=True,
            #     with_stack=True,
            #     with_flops=True,
            #     record_shapes=True,
            # ) as prof,
        ):
            pbar.update(self.step)
            while self.step <= max_steps:
                time_start = time_ns()
                try:
                    real_cells, real_labels = next(loader_gen)
                except StopIteration:
                    loader_gen = iter(loader)
                    real_cells, real_labels = next(loader_gen)

                real_cells = real_cells.to(self.device)
                real_labels = real_labels.flatten().to(self.device)

                iter_losses = self._training_step(real_cells, real_labels, critic_iter, c_lambda)
                iter_losses["total_loss"] = np.nansum([
                    iter_losses["gen_total_loss"],
                    iter_losses["crit_total_loss"],
                ])  # gp is already included in crit_loss
                losses.append(iter_losses)

                if should_run(save_freq):
                    self._save(output_dir)
                    logger.info(f"Step {self.step}: Saved checkpoint to {output_dir}")

                # Log and visualize progress
                if should_run(summary_freq):
                    val_loss = self._get_validation_loss(valid_loader, c_lambda)
                    val_loss_display_names = {
                        "val_gen_loss": "Validation Generator Loss",
                        "val_gen_total_loss": "Validation Generator Total Loss",
                        "val_crit_real_loss": "Validation Critic Real Loss",
                        "val_crit_fake_loss": "Validation Critic Fake Loss",
                        "val_crit_gp_loss": "Validation Critic Gradient Penalty Loss",
                        "val_crit_total_loss": "Validation Critic Total Loss",
                        "val_total_loss": "Validation Total Loss",
                    }
                    val_loss = {val_loss_display_names.get(k, k): v for k, v in val_loss.items()}
                    loss_dict_display_names = {
                        "gen_loss": "Generator Loss",
                        "gen_total_loss": "Generator Total Loss",
                        "crit_fake_loss": "Critic Fake Loss",
                        "crit_real_loss": "Critic Real Loss",
                        "crit_gp_loss": "Critic Gradient Penalty Loss",
                        "crit_total_loss": "Critic Total Loss",
                        "total_loss": "Total Loss",
                    }
                    # add loss keys without custom display names
                    loss_dict_display_names |= {
                        k: k for k in losses[-1].keys() if k not in loss_dict_display_names.keys()
                    }
                    loss_dict = {
                        v: float(np.nanmean([iter_losses[k] for iter_losses in losses[-summary_freq:]]))
                        for k, v in loss_dict_display_names.items()
                    } | val_loss

                    learning_rates_dict = {
                        "Generator LR": self.gen_lr_scheduler.get_last_lr()[0].item(),  # pyright: ignore[reportAttributeAccessIssue]
                        "Critic LR": self.crit_lr_scheduler.get_last_lr()[0].item(),  # pyright: ignore[reportAttributeAccessIssue]
                        "Generator Avg Abs Weight": torch.cat([
                            v.flatten() for k, v in self.gen.named_parameters() if "_lsn" not in k
                        ])
                        .abs()
                        .mean()
                        .item(),
                        "Critic Avg Abs Weight": torch.cat([v.flatten() for v in self.crit.parameters()])
                        .abs()
                        .mean()
                        .item(),
                    }

                    self._update_tensorboard(loss_dict | learning_rates_dict, output_dir, summary_writer)
                    logger.info(f"Step {self.step}:\n" + pd.Series(loss_dict).to_string(float_format="{:.2g}".format))
                    logger.debug(
                        f"Step {self.step}:\n" + pd.Series(learning_rates_dict).to_string(float_format="{:.2g}".format)
                    )

                    if trial:
                        # Allow trial pruning before reaching the end of training
                        if trial.should_prune():
                            raise TrialPruned()

                if should_run(rf_auroc_freq):
                    logger.info(f"Step {self.step}: Computing Random Forest AUROC...")
                    fake_cells = self.generate_cells(len(valid_loader.dataset))[0]

                    if not is_ddp_initialized() or os.environ.get("RANK") == "0":
                        rf_auroc, fig = compute_RF_AUROC(valid_loader.dataset.cells, fake_cells)
                        rf_auroc_dir = output_dir / "RF_AUROC"
                        rf_auroc_dir.mkdir(parents=True, exist_ok=True)
                        fig.savefig(rf_auroc_dir / f"step_{self.step}.jpg")
                        with SummaryWriter(
                            output_dir / "TensorBoard/RF_AUROC", filename_suffix=f".step{self.step}"
                        ) as w:
                            w.add_figure("Random Forest AUROC", fig, self.step)
                            w.add_scalar("AUROC", rf_auroc, self.step)

                        logger.info(f"Step {self.step}: Computed Random Forest AUROC: {rf_auroc:.3f}")
                        if trial:
                            trial.report(rf_auroc, self.step)
                            if trial.should_prune():  # Allow trial pruning before reaching the end of training
                                raise TrialPruned()

                if should_run(plt_freq):
                    logger.info(f"Step {self.step}: Generating UMAP plots...")
                    self._generate_umap_plots(valid_loader, output_dir)
                    logger.info(f"Step {self.step}: Generated and saved UMAP plots to {output_dir}")

                if is_ddp_initialized():
                    barrier()
                time_end = time_ns()
                logger.debug(
                    f"Step {self.step}/{max_steps} completed in {(time_end - time_start) // 1_000_000:.0f} milliseconds"
                )
                pbar.update(min(self.step, 1))  # don't update on step 0
                self.step += 1
                # prof.step()

        # logger.info(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=100))
        # pd.DataFrame(map(vars, prof.key_averages())).to_excel("profiler_stats.xlsx")
        # # prof.export_memory_timeline("memtrace.html") # Gives unexpected errors
        # prof.export_chrome_trace("trace.json")

        if rf_auroc_freq > 0:
            ret = rf_auroc
        elif loss_dict:
            ret = loss_dict["Validation Total Loss"]
        else:
            ret = float("inf")
        return ret
