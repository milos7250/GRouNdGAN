from pathlib import Path

import torch
from torch.cuda import is_available as is_cuda_available

from networks.critic import Critic
from networks.generator import Generator
from networks.labeler import Labeler
from networks.masked_causal_generator import CausalGenerator

from .gan import GAN


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
        self.causal_controller.load_state_dict(checkpoint["gen_state_dict"], strict=False)

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

        self.crit = Critic(self.genes_no, self.crit_layers).to(self.device)

        # the number of genes and TFs are resolved by the causal generator during its instantiation
        self.labeler = Labeler(self.gen.num_genes, self.gen.num_tfs, self.labeler_layers).to(self.device)
        self.antilabeler = Labeler(self.gen.num_genes, self.gen.num_tfs, self.labeler_layers).to(self.device)

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
                "labeler_state_dict": self.labeler.state_dict(),
                "antilabeler_state_dict": self.antilabeler.state_dict(),
            },
            path.with_suffix(".pth"),
        )

    def load(
        self,
        path: Path,
    ) -> None:
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
        self.labeler.load_state_dict(checkpoint["labeler_state_dict"])
        self.antilabeler.load_state_dict(checkpoint["antilabeler_state_dict"])
