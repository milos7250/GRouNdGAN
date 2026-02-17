import os
from pathlib import Path
from time import time_ns
from typing import TYPE_CHECKING
from warnings import filterwarnings

import numpy as np
import pandas as pd
import torch
import torch._inductor.select_algorithm
from optuna import TrialPruned
from torch.cuda import is_available as is_cuda_available
from torch.distributed import barrier  # pyright: ignore[reportUnknownVariableType]
from torch.distributed import is_initialized as is_ddp_initialized
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm.rich import tqdm

from evaluation.data_quality import compute_RF_AUROC
from gans.gan import GAN
from loggers import setup_logger, tqdm_logging_redirect
from networks.critic import Critic
from networks.generator import Generator
from networks.labeler import Labeler
from networks.masked_causal_generator import CausalGenerator

if TYPE_CHECKING:
    from typing import Any

    from optuna import Trial
    from torch import Tensor
    from torch.nn import Buffer

filterwarnings("ignore", message=".*rich is experimental/alpha.*")


class CausalGAN(GAN):
    def __init__(
        self,
        genes_no: int,
        batch_size: int,
        latent_dim: int,
        noise_per_gene: int,
        depth_per_gene: int,
        width_per_gene: int,
        cc_latent_dim: int,
        cc_layers: list[int],
        cc_pretrained_checkpoint: Path,
        crit_layers: list[int],
        causal_graph: dict[int, set[int]],
        labeler_layers: list[int],
        device: str | None = None,
        library_size: int | None = 20000,
    ) -> None:
        """
        Causal single-cell RNA-seq GAN (TODO: find a unique name).

        Parameters
        ----------
        genes_no : int
            Number of genes in the dataset.
        batch_size : int
            Training batch size.
        latent_dim : int
            Dimension of the latent space from which the noise vector used by the causal controller is sampled.
        noise_per_gene : int
            Dimension of the latent space from which the noise vectors used by target generators is sampled.
        depth_per_gene : int
            Depth of the target generator networks.
        width_per_gene : int
            The width scale used for the target generator networks.
        cc_latent_dim : int
            Dimension of the latent space from which the noise vector to the causal controller is sampled.
        cc_layers : list[int]
            list of integers corresponding to the number of neurons of each causal controller layer.
        cc_pretrained_checkpoint : Path
            Path to the  pretrained causal controller.
        crit_layers : list[int]
            list of integers corresponding to the number of neurons of each critic layer.
        causal_graph : dict[int, set[int]]
            The causal graph is a dictionary representing the TRN to impose. It has the following format:
            {target gene index: {TF1 index, TF2 index, ...}}. This causal graph has to be acyclic and bipartite.
            A TF cannot be regulated by another TF.
            Invalid: {1: {2, 3, {4, 6}}, ...} - a regulator (TF) is regulated by another regulator (TF)
            Invalid: {1: {2, 3, 4}, 2: {4, 3, 5}, ...} - a regulator (TF) is also regulated
            Invalid: {4: {2, 3}, 2: {4, 3}} - contains a cycle

            Valid causal graph example: {1: {2, 3, 4}, 6: {5, 4, 2}, ...}
        labeler_layers : list[int]
            list of integers corresponding to the width of each labeler layer.
        device : str | None, optional
            Specifies to train on 'cpu' or 'cuda'. Only 'cuda' is supported for training the
            GAN but 'cpu' can be used for inference, by default "cuda" if torch.cuda.is_available() else"cpu".
        library_size : int | None, optional
            Total number of counts per generated cell, by default 20000.
        """

        self.causal_controller = Generator(
            z_input=cc_latent_dim,
            output_cells_dim=genes_no,
            gen_layers=cc_layers,
            library_size=None,
        )

        device = device if device else ("cuda" if is_cuda_available() else "cpu")

        checkpoint = torch.load(cc_pretrained_checkpoint, map_location=torch.device(device))
        self.causal_controller.load_state_dict(checkpoint["generator_state_dict"], strict=False)

        self.noise_per_gene = noise_per_gene
        self.depth_per_gene = depth_per_gene
        self.width_per_gene = width_per_gene
        self.causal_graph = causal_graph
        self.labeler_layers = labeler_layers
        super().__init__(
            genes_no,
            batch_size,
            latent_dim,
            [],
            crit_layers,
            device=device,
            library_size=library_size,
        )

    def _build_model(self) -> None:
        """Instantiates the Generator and Critic."""
        self.gen = CausalGenerator(
            self.latent_dim,
            self.noise_per_gene,
            self.depth_per_gene,
            self.width_per_gene,
            self.causal_controller,
            self.causal_graph,
            self.library_size,
            self.device,
        ).to(self.device)
        self.gen.freeze_causal_controller()

        self.crit = Critic(self.genes_no, self.critic_layers).to(self.device)

        # the number of genes and TFs are resolved by the causal generator during its instantiation
        self.labeler = Labeler(self.gen.num_genes, self.gen.num_tfs, self.labeler_layers).to(self.device)
        self.antilabeler = Labeler(self.gen.num_genes, self.gen.num_tfs, self.labeler_layers).to(self.device)

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
                "labeler_state_dict": self.labeler.module.state_dict(),  # pyright: ignore[reportAttributeAccessIssue]
                "antilabeler_state_dict": self.antilabeler.module.state_dict(),  # pyright: ignore[reportAttributeAccessIssue]
            }
        else:
            state_dict = {
                "generator_state_dict": self.gen.state_dict(),
                "critic_state_dict": self.crit.state_dict(),
                "labeler_state_dict": self.labeler.state_dict(),
                "antilabeler_state_dict": self.antilabeler.state_dict(),
            }

        torch.save(
            state_dict
            | {
                "step": self.step,
                "generator_optimizer_state_dict": self.gen_opt.state_dict() if self.gen_opt else None,
                "critic_optimizer_state_dict": self.crit_opt.state_dict() if self.crit_opt else None,
                "labeler_optimizer_state_dict": self.labeler_opt.state_dict() if self.labeler_opt else None,
                "antilabeler_optimizer_state_dict": self.antilabeler_opt.state_dict() if self.antilabeler_opt else None,
                "generator_lr_scheduler": self.gen_lr_scheduler.state_dict() if self.gen_lr_scheduler else None,
                "critic_lr_scheduler": self.crit_lr_scheduler.state_dict() if self.crit_lr_scheduler else None,
            },
            path / f"checkpoints/step_{self.step}.pth",
        )

    def _load(
        self,
        path: Path,
        mode: str | None = "inference",
    ) -> None:
        """
        Loads a saved causal GAN model (.pth file). Inference mode only loads the generator and critic.
        Initialization mode loads the model only for weight initialization of the generator, critic and
        labellers (optimizer states are not loaded). Training mode loads the model for training from
        checkpoint with optimizer states.

        Parameters
        ----------
        path : Path
            Path to the saved model.
        mode : str | None, optional
            Specify if the loaded model is used for 'inference', 'initialization', or 'training', by default "inference".

        Raises
        ------
        ValueError
            If a mode other than 'inference', 'initialization', or 'training' is specified.
        RuntimeError
            If training mode is specified but the optimizers or learning rate schedulers are not initialized.
        """

        checkpoint = torch.load(path, map_location=torch.device(self.device))

        self.gen.load_state_dict(checkpoint["generator_state_dict"])
        self.crit.load_state_dict(checkpoint["critic_state_dict"])

        if mode == "inference":
            # The causal GAN performs better when using batch stats (model.train() mode)
            self.gen.train()
            self.crit.train()

        elif mode == "initialization":
            self.labeler.load_state_dict(checkpoint["labeler_state_dict"])
            self.antilabeler.load_state_dict(checkpoint["antilabeler_state_dict"])

            self.gen.train()
            self.crit.train()
            self.labeler.train()
            self.antilabeler.train()

        elif mode == "training":
            self.gen.train()
            self.crit.train()

            if (
                not self.gen_opt
                or not self.crit_opt
                or not self.gen_lr_scheduler
                or not self.crit_lr_scheduler
                or not self.labeler_opt
                or not self.antilabeler_opt
            ):
                raise RuntimeError(
                    "Generator, critic, labeler, and antilabeler optimizers and generator and critic learning rate"
                    "schedulers must be initialized before loading a training checkpoint."
                )

            self.step = checkpoint["step"] + 1
            self.gen_opt.load_state_dict(checkpoint["generator_optimizer_state_dict"])
            self.crit_opt.load_state_dict(checkpoint["critic_optimizer_state_dict"])
            self.gen_lr_scheduler.load_state_dict(checkpoint["generator_lr_scheduler"])
            self.crit_lr_scheduler.load_state_dict(checkpoint["critic_lr_scheduler"])
            self.labeler.load_state_dict(checkpoint["labeler_state_dict"])
            self.antilabeler.load_state_dict(checkpoint["antilabeler_state_dict"])
            self.labeler_opt.load_state_dict(checkpoint["labeler_optimizer_state_dict"])
            self.antilabeler_opt.load_state_dict(checkpoint["antilabeler_optimizer_state_dict"])

        else:
            raise ValueError("mode should be 'inference', 'initialization', or 'training'")

    def _antilabeler_step(self, cells: "Tensor", genes: "Tensor", tfs: "Tensor") -> tuple["Tensor", "Tensor"]:
        """
        Performs a forward pass of the antilabeler and computes the antilabeler loss.

        Parameters
        ----------
        cells : Tensor
            Tensor containing a batch of cells.
        genes : Tensor
            Tensor containing the indices of the genes in the causal graph.
        tfs : Tensor
            Tensor containing the indices of the TFs in the causal graph.

        Returns
        -------
        tuple[Tensor, Tensor]
            Antilabeler's loss for the current batch and the predicted TFs.
        """
        predicted_tfs = self.antilabeler(cells[:, genes])
        antilabeler_loss = self.mse(predicted_tfs, cells[:, tfs])

        return antilabeler_loss, predicted_tfs

    def _labeler_step(self, cells: "Tensor", genes: "Tensor", tfs: "Tensor") -> tuple["Tensor", "Tensor"]:
        """
        Performs a forward pass of the labeler and computes the labeler loss.

        Parameters
        ----------
        cells : Tensor
            Tensor containing a batch of cells.
        genes : Tensor
            Tensor containing the indices of the genes in the causal graph.
        tfs : Tensor
            Tensor containing the indices of the TFs in the causal graph.

        Returns
        -------
        Tensor
            Labeler's loss for the current batch and the predicted TFs.
        """
        predicted_tfs = self.labeler(cells[:, genes])
        labeler_loss = self.mse(predicted_tfs, cells[:, tfs])

        return labeler_loss, predicted_tfs

    def _train_labelers(self, real_cells: "Tensor") -> dict[str, float]:
        """
        Trains the labeler (on real and fake) and anti-labeler (on fake only).

        Parameters
        ----------
        real_cells : Tensor
            Tensor containing a batch of real cells.

        Returns
        -------
        dict[str, float]
            dictionary containing the labeler and anti-labeler losses.
        """
        torch.compiler.cudagraph_mark_step_begin()

        with torch.no_grad():
            fake_noise = self._generate_noise(self.batch_size, self.latent_dim, self.device)
            fake = self.gen(fake_noise)

        if is_ddp_initialized():
            genes: Buffer = self.gen.module.genes_tensor  # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue]
            tfs: Buffer = self.gen.module.tfs_tensor  # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue]
        else:
            genes: Buffer = self.gen.genes_tensor  # pyright: ignore[reportAssignmentType]
            tfs: Buffer = self.gen.tfs_tensor  # pyright: ignore[reportAssignmentType]

        losses = {}
        # train anti-labeler
        self.antilabeler_opt.zero_grad(set_to_none=True)
        antilabeler_loss, _ = self._antilabeler_step(fake, genes, tfs)
        losses["antilabeler_loss"] = antilabeler_loss.item()
        antilabeler_loss.backward()
        self.antilabeler_opt.step()

        # train labeler on fake data
        self.labeler_opt.zero_grad(set_to_none=True)
        labeler_fake_loss, _ = self._labeler_step(fake, genes, tfs)
        losses["labeler_fake_loss"] = labeler_fake_loss.item()
        labeler_fake_loss.backward()
        self.labeler_opt.step()

        # train labeler on real data
        self.labeler_opt.zero_grad(set_to_none=True)
        labeler_real_loss, _ = self._labeler_step(real_cells, genes, tfs)
        losses["labeler_real_loss"] = labeler_real_loss.item()
        labeler_real_loss.backward()
        self.labeler_opt.step()

        return losses

    def _train_generator(self) -> dict[str, float]:
        """
        Trains the causal generator for one iteration.

        Returns
        -------
        dict[str, float]
            dictionary containing the generator, labeler and anti-labeler losses.

        Raises
        ------
        RuntimeError
            If the generator optimizer is not initialized.
        """
        if not self.gen_opt:
            raise RuntimeError("Generator optimizer not initialized.")

        self.gen_opt.zero_grad(set_to_none=True)

        if is_ddp_initialized():
            genes: Buffer = self.gen.module.genes_tensor  # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue]
            tfs: Buffer = self.gen.module.tfs_tensor  # pyright: ignore[reportAssignmentType, reportAttributeAccessIssue]
        else:
            genes: Buffer = self.gen.genes_tensor  # pyright: ignore[reportAssignmentType]
            tfs: Buffer = self.gen.tfs_tensor  # pyright: ignore[reportAssignmentType]

        gen_loss, _, fake = self._generator_step()

        labeler_loss, _ = self._labeler_step(fake, genes, tfs)
        antilabeler_loss, _ = self._antilabeler_step(fake, genes, tfs)

        total_loss = torch.nansum(torch.stack([gen_loss, labeler_loss, antilabeler_loss]))
        losses = {
            "gen_loss": gen_loss.item(),
            "gen_labeler_loss": labeler_loss.item(),
            "gen_antilabeler_loss": antilabeler_loss.item(),
            "gen_total_loss": total_loss.item(),
        }

        total_loss.backward()  # only total_loss needs to be backpropagated as it includes labeler and anti-labeler losses

        # Update weights
        self.gen_opt.step()

        return losses

    # FIXME: A lot of code duplication here with the parent train() method.
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
        labeler_alpha: float,
        antilabeler_alpha: float,
        labeler_training_interval: int,
        checkpoint: Path | None = None,
        starting_checkpoint: Path | None = None,
        output_dir: Path = Path("output"),
        summary_freq: int = 5000,
        plt_freq: int = 10000,
        save_freq: int = 10000,
        rf_auroc_freq: int = 0,
        trial: "Trial | None" = None,
        **kwargs: "Any",
    ) -> float:
        """
        Method for training the causal GAN.

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
        labeler_alpha : float
            Labeler's learning rate value.
        antilabeler_alpha : float
            Anti-labeler's learning rate value.
        labeler_training_interval: int
            The number of steps after which the labeler and anti-labeler are trained.
            If 20, the labeler and anti-labeler will be trained every 20 steps.
        checkpoint : Path | None, optional
            Path to a trained model; if specified, the checkpoint is be used to resume training, by default None.
        starting_checkpoint : Path | None, optional
            Path to a trained model; if specified, the checkpoint is be used to initialize the generator, critic,
            labeler and anti-labeler, by default None.
        output_dir : Path | None, optional
            Directory to which plots, tfevents, and checkpoints will be saved, by default "output".
        summary_freq : int | None, optional
            Period between summary logs to TensorBoard, by default 5000.
        plt_freq : int | None, optional
            Period between t-SNE plots, by default 10000.
        save_freq : int | None, optional
            Period between saves of the model, by default 10000.
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
        logger = setup_logger("causal_gan.train")
        if kwargs:
            logger.warning(f"Unused arguments passed to gan.train(): {kwargs}")

        def should_run(freq: int, /) -> bool:
            return (freq > 0 and self.step % freq == 0 and self.step > 0) or (self.step == max_steps)

        loader, valid_loader = self._get_loaders(train_files, valid_files)
        loader_gen = iter(loader)

        # Instantiate optimizers
        self.gen_opt = AdamW(
            filter(lambda p: p.requires_grad, self.gen.parameters()),
            lr=torch.tensor(gen_alpha_0),
            betas=(beta1, beta2),
            amsgrad=True,
            fused=True,
        )

        self.crit_opt = AdamW(
            filter(lambda p: p.requires_grad, self.crit.parameters()),
            lr=torch.tensor(crit_alpha_0),
            betas=(beta1, beta2),
            amsgrad=True,
            fused=True,
        )

        self.labeler_opt = AdamW(
            filter(lambda p: p.requires_grad, self.labeler.parameters()),
            lr=torch.tensor(labeler_alpha),
            betas=(beta1, beta2),
            amsgrad=True,
            fused=True,
        )

        self.antilabeler_opt = AdamW(
            filter(lambda p: p.requires_grad, self.antilabeler.parameters()),
            lr=torch.tensor(antilabeler_alpha),
            betas=(beta1, beta2),
            amsgrad=True,
            fused=True,
        )

        # for the labeler and anti-labeler
        self.mse = torch.nn.MSELoss()

        # Exponential Learning Rate
        self.gen_lr_scheduler = self._set_exponential_lr(self.gen_opt, gen_alpha_0, gen_alpha_final, max_steps, 0.05)
        self.crit_lr_scheduler = self._set_exponential_lr(
            self.crit_opt, crit_alpha_0, crit_alpha_final, max_steps, 0.00
        )

        if checkpoint is not None:
            self._load(checkpoint, mode="training")
        elif starting_checkpoint is not None:
            self._load(starting_checkpoint, mode="initialization")

        self.gen.train()
        self.crit.train()
        self.labeler.train()
        self.antilabeler.train()

        logger.info("Saving model graph...")
        with torch.compiler.set_stance("force_eager"):
            self.log_tensorboard_graph(output_dir)

        torch._inductor.select_algorithm.PRINT_AUTOTUNE = False  # to suppress autotune printing
        if is_ddp_initialized():
            logger.info("Distributed Data Parallel (DDP) training active, compiling generator and critic modules")
            self.gen = DDP(torch.compile(self.gen, fullgraph=True, mode="max-autotune-no-cudagraphs"))
            self.crit = DDP(torch.compile(self.crit, fullgraph=True, mode="max-autotune-no-cudagraphs"))
            self.labeler = DDP(torch.compile(self.labeler, fullgraph=True, mode="max-autotune-no-cudagraphs"))
            self.antilabeler = DDP(torch.compile(self.antilabeler, fullgraph=True, mode="max-autotune-no-cudagraphs"))
        else:
            logger.info("Single-device training active, compiling generator and critic step functions.")
            self._generator_step = torch.compile(
                self._generator_step, fullgraph=True, mode="max-autotune-no-cudagraphs"
            )
            self._critic_step = torch.compile(self._critic_step, fullgraph=True, mode="max-autotune-no-cudagraphs")
            self._labeler_step = torch.compile(self._labeler_step, fullgraph=True, mode="max-autotune-no-cudagraphs")
            self._antilabeler_step = torch.compile(
                self._antilabeler_step, fullgraph=True, mode="max-autotune-no-cudagraphs"
            )

        if self.device == "cpu":
            logger.warning("Training on CPU is not supported and will be very slow.")

        # Main training loop
        losses = []
        loss_dict: dict[str, float] = {}
        rf_auroc = np.inf
        summary_writers = {
            "stats": SummaryWriter(output_dir / "TensorBoard/"),
            "umap": SummaryWriter(output_dir / "TensorBoard/", filename_suffix=".UMAP"),
            "rf_auroc": SummaryWriter(output_dir / "TensorBoard/", filename_suffix=".RF_AUROC"),
        }
        torch.set_float32_matmul_precision("high")
        logger.info("Starting training...")
        with (
            tqdm_logging_redirect(
                loggers=[logger], tqdm_class=tqdm, desc="Training Causal GAN", total=max_steps
            ) as pbar,
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

                if should_run(labeler_training_interval):
                    iter_losses |= self._train_labelers(real_cells)
                else:
                    iter_losses["labeler_fake_loss"] = float("nan")
                    iter_losses["labeler_real_loss"] = float("nan")
                    iter_losses["antilabeler_loss"] = float("nan")

                iter_losses["total_loss"] = np.nansum([
                    iter_losses["gen_total_loss"],
                    iter_losses["crit_total_loss"],
                ])  # labeler and anti-labeler losses are already included in gen loss, gp is already in crit loss
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
                        "gen_labeler_loss": "Generator Labeler Loss",
                        "gen_antilabeler_loss": "Generator Anti-labeler Loss",
                        "gen_total_loss": "Generator Total Loss",
                        "crit_total_loss": "Critic Total Loss",
                        "crit_fake_loss": "Critic Fake Loss",
                        "crit_real_loss": "Critic Real Loss",
                        "crit_gp_loss": "Critic Gradient Penalty Loss",
                        "labeler_fake_loss": "Labeler Loss on Fake",
                        "labeler_real_loss": "Labeler Loss on Real",
                        "antilabeler_loss": "Anti-labeler Loss",
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
                        "Labeler Avg Abs Weight": torch.cat([v.flatten() for v in self.labeler.parameters()])
                        .abs()
                        .mean()
                        .item(),
                        "Anti-labeler Avg Abs Weight": torch.cat([v.flatten() for v in self.antilabeler.parameters()])
                        .abs()
                        .mean()
                        .item(),
                    }

                    self._update_tensorboard(loss_dict | learning_rates_dict, output_dir, summary_writers["stats"])
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

                        summary_writers["rf_auroc"].add_figure("Random Forest AUROC", fig, self.step)
                        summary_writers["stats"].add_scalar("AUROC", rf_auroc, self.step)

                        logger.info(f"Step {self.step}: Computed Random Forest AUROC: {rf_auroc:.3f}")
                        if trial:
                            trial.report(rf_auroc, self.step)
                            if trial.should_prune():  # Allow trial pruning before reaching the end of training
                                raise TrialPruned()

                        if self.step > 50_000 and rf_auroc > 0.99:
                            logger.info(f"Step {self.step}: Early stopping as RF AUROC > 0.99 (AUROC: {rf_auroc:.3f})")
                            break

                if should_run(plt_freq):
                    logger.info(f"Step {self.step}: Generating UMAP plots...")
                    self._generate_umap_plots(valid_loader, output_dir, summary_writer=summary_writers["umap"])
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
        # pd.DataFrame(map(vars, prof.key_averages())).to_excel(output_dir / "profiler_stats.xlsx") # Needs openpyxl installed
        # prof.export_memory_timeline("memtrace.html") # Gives unexpected errors
        # prof.export_chrome_trace(output_dir / "trace.json")

        return rf_auroc
