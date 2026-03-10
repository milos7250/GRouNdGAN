from pathlib import Path
from typing import TYPE_CHECKING, overload
from warnings import catch_warnings, filterwarnings

import numpy as np
import scanpy as sc
import torch
from scipy.sparse import csr_matrix
from torch.cuda import empty_cache as empty_cuda_cache
from torch.cuda import is_available as is_cuda_available
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
from umap import UMAP

from loggers import setup_logger
from networks.critic import Critic
from networks.generator import Generator
from training.dicts import GANTrainingArgs

if TYPE_CHECKING:
    from typing import Any

    from torch import Tensor


class GAN:
    training_args_class = GANTrainingArgs

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
        self.crit_layers = crit_layers
        self.device = device if device else ("cuda" if is_cuda_available() else "cpu")
        self.library_size = library_size

        self._build_model()

        self.step: int = 0
        self.optimizers: dict[str, Optimizer] = {}
        self.schedulers: dict[str, LRScheduler] = {}
        self.umap: UMAP | None = None
        self.real_embedding: np.ndarray | None = None

    def _build_model(self) -> None:
        """Instantiates the Generator and Critic."""
        self.gen = Generator(self.latent_dim, self.genes_no, self.gen_layers, self.library_size).to(self.device)
        self.crit = Critic(self.genes_no, self.crit_layers).to(self.device)

    @staticmethod
    def generate_noise(batch_size: int, latent_dim: int, device: str) -> "Tensor":
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

    def generate_cells(
        self,
        cells_no: int,
        checkpoint: Path | None = None,
        *args: "Any",
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
        *args : Any
            Additional positional arguments (not used).
        kwargs : Any
            Additional keyword arguments (not used).

        Returns
        -------
        tuple[np.ndarray, np.ndarray | None]
            Tuple of Gene expression matrix of generated cells and None (dummy labels).
        """
        if checkpoint:
            self.load(checkpoint)

        # find how many batches to generate
        batch_no = int(np.ceil(cells_no / self.batch_size))

        fake_cells = []
        was_training = self.gen.training
        self.gen.eval()
        with torch.inference_mode():
            for _ in range(batch_no):
                noise = self.generate_noise(self.batch_size, self.latent_dim, self.device)
                fake_cells.append(self.gen(noise).cpu().detach().numpy())
        self.gen.train(was_training)

        return np.concatenate(fake_cells)[:cells_no], None

    @overload
    def generate_h5ad(self, cells_no: int, *, checkpoint: Path | None = None) -> sc.AnnData: ...
    @overload
    def generate_h5ad(self, cells_no: int, *, gene_names: list[str], checkpoint: Path | None = None) -> sc.AnnData: ...
    @overload
    def generate_h5ad(
        self, cells_no: int, *, reference_dataset: Path, checkpoint: Path | None = None
    ) -> sc.AnnData: ...
    @overload
    def generate_h5ad(self, cells_no: int, save_path: Path, *, checkpoint: Path | None = None) -> None: ...
    @overload
    def generate_h5ad(
        self, cells_no: int, save_path: Path, *, gene_names: list[str], checkpoint: Path | None = None
    ) -> None: ...
    @overload
    def generate_h5ad(
        self, cells_no: int, save_path: Path, *, reference_dataset: Path, checkpoint: Path | None = None
    ) -> None: ...
    def generate_h5ad(
        self,
        cells_no: int,
        save_path: Path | None = None,
        *,
        gene_names: list[str] | None = None,
        reference_dataset: Path | None = None,
        checkpoint: Path | None = None,
    ) -> sc.AnnData | None:
        """
        Generates an h5ad file containing the generated cells.

        Parameters
        ----------
        cells_no : int
            Number of cells to generate.
        save_path : Path | None, optional
            Path to save the generated h5ad file, by default None, iin which case the AnnData object will not be saved to disk.
        gene_names : list[str] | None, optional
            List of gene names to use as variable names in the generated h5ad file. If None, gene names will be taken from the reference dataset, by default None.
        reference_dataset : Path | None, optional
            Path to the reference dataset h5ad file to take gene names from if gene_names is None, by default None.
        checkpoint : Path | None, optional
            Path to the saved trained model, by default None.

        Returns
        -------
        sc.AnnData
            An AnnData object containing the generated cells.
        """
        generated_cells = self.generate_cells(
            cells_no,
            checkpoint,
        )[0]
        generated_cells = csr_matrix(generated_cells)

        generated_h5ad = sc.AnnData(generated_cells)
        generated_h5ad.obs_names = np.repeat("fake", generated_h5ad.shape[0]).tolist()
        generated_h5ad.obs_names_make_unique()

        # Add variable names
        if gene_names and reference_dataset:
            setup_logger(__name__).warning(
                "Both gene_names and reference_dataset provided. gene_names will be used for variable names."
            )
        if gene_names:
            generated_h5ad.var_names = gene_names
        elif reference_dataset:
            train_var_names = sc.read_h5ad(reference_dataset, backed="r").var_names
            generated_h5ad.var_names = train_var_names.tolist()

        if save_path:
            generated_h5ad.write(save_path)
            return None
        else:
            return generated_h5ad

    def save(self, path: Path) -> None:
        """
        Saves the model.

        Parameters
        ----------
        path : Path
            Path to save the model. The model will be saved in .pth format.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "gen_state_dict": self.gen.state_dict(),
                "crit_state_dict": self.crit.state_dict(),
            },
            path.with_suffix(".pth"),
        )

    def load(self, path: Path) -> None:
        """
        Loads a saved model (.pth file).

        Parameters
        ----------
        path : Path
            Path to the saved model .pth file.
        """

        checkpoint = torch.load(path, map_location=torch.device(self.device))

        self.gen.load_state_dict(checkpoint["gen_state_dict"])
        self.crit.load_state_dict(checkpoint["crit_state_dict"])

    def _generate_gen_and_crit_data(
        self,
    ) -> tuple["Tensor", "Tensor"] | tuple[tuple["Tensor", ...], tuple["Tensor", ...]]:
        """
        Generates a batch of noise and corresponding fake cells for logging the model graph. Must work as
        input to summary_writer.add_graph(self.gen, input_to_model=gen_data) and
        summary_writer.add_graph(self.crit, input_to_model=crit_data).

        Returns
        -------
        tuple[Tensor, Tensor]
            A batch of noise (gen_data) and the corresponding generated fake cells (crit_data).
        """
        with torch.no_grad():
            gen_data = self.generate_noise(self.batch_size, self.latent_dim, self.device)
            crit_data = self.gen(gen_data)
        return gen_data, crit_data

    @torch.compiler.set_stance("force_eager")
    def log_tensorboard_graph(self, output_dir: Path) -> None:
        """
        Adds the model graph to TensorBoard.

        Parameters
        ----------
        output_dir : Path
            Directory to save the tfevents.
        """
        was_training = (self.gen.training, self.crit.training)
        self.gen.eval()
        self.crit.eval()

        gen_data, crit_data = self._generate_gen_and_crit_data()

        with catch_warnings():
            filterwarnings("ignore", message=".*Trace had nondeterministic nodes.*")
            filterwarnings(
                "ignore",
                message=".*the traced function does not match the corresponding output of the Python function.*",
            )
            filterwarnings(
                "ignore", message=r".*The \.grad attribute of a Tensor that is not a leaf Tensor is being accessed*"
            )
            with SummaryWriter(f"{output_dir}/TensorBoard/", filename_suffix=f".gen_graph.step{self.step}") as w:
                w.add_graph(self.gen, gen_data, use_strict_trace=False)
            with SummaryWriter(f"{output_dir}/TensorBoard/", filename_suffix=f".crit_graph.step{self.step}") as w:
                w.add_graph(self.crit, crit_data, use_strict_trace=False)

        self.gen.train(was_training[0])
        self.crit.train(was_training[1])
