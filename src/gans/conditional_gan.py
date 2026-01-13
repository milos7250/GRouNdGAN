import os
from abc import ABC
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from torch.distributed import is_initialized as is_ddp_initialized
from torch.utils.tensorboard import SummaryWriter
from umap import UMAP

from gans.gan import GAN

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from torch import Tensor

    from sc_dataset import SCDataLoader


class ConditionalGAN(GAN, ABC):
    def __init__(self, *args: "Any", **kwargs: "Any") -> None:
        self.num_classes: int
        self.label_ratios: "Tensor"

    @staticmethod
    def _sample_pseudo_labels(batch_size: int, cluster_ratios: "Tensor") -> "Tensor":
        """
        Randomly samples cluster labels following a multinomial distribution.

        Parameters
        ----------
        batch_size : int
            The number of samples to generate (normally equal to training batch size).
        cluster_ratios : Tensor
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

    def _generate_umap_plots(
        self,
        valid_loader: "SCDataLoader",
        output_dir: "Path",
        max_cells: int | None = None,
    ) -> None:
        """
        Generate t-SNE plot during training.

        Parameters
        ----------
        valid_loader : SCDataLoader
            Validation set DataLoader.
        output_dir : Path
            Directory to save the t-SNE plots.
        """
        no_of_cells = min(max_cells, len(valid_loader.dataset)) if max_cells else len(valid_loader.dataset)

        fake_cells, fake_labels = self.generate_cells(no_of_cells)
        if not fake_labels:
            raise ValueError("Cannot generate UMAP plots without class labels from the Conditional GAN.")

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
        real_labels = valid_loader.dataset.clusters.numpy()
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

        colormap = cm.get_cmap("nipy_spectral")
        colors = [colormap(i) for i in np.linspace(0, 1, self.num_classes)]

        plt.clf()
        scatter_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        for i in range(self.num_classes):
            mask = real_labels[:] == i

            ax1.scatter(
                real_embedding[mask, 0],
                real_embedding[mask, 1],
                c=colors[i],
                marker="o",
                label="real_" + str(i),
            )

        ax1.legend(loc="lower left", numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
        ax1.set_title("Real cells")
        ax1.set_xlim(extent[0])
        ax1.set_ylim(extent[1])

        for i in range(self.num_classes):
            mask = fake_labels[:] == i
            ax2.scatter(
                fake_embedding[mask, 0],
                fake_embedding[mask, 1],
                c=colors[i],
                marker="o",
                label="generated_" + str(i),
            )

        ax2.legend(loc="lower left", numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
        ax2.set_title("Generated cells")
        ax2.set_xlim(extent[0])
        ax2.set_ylim(extent[1])

        scatter_fig.suptitle(f"UMAP Projection of Real and Generated Cells at Step {self.step}")
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

        H_real, xedges, yedges = np.histogram2d(real_embedding[:, 0], real_embedding[:, 1], bins=100, range=extent)
        H_fake, _, _ = np.histogram2d(fake_embedding[:, 0], fake_embedding[:, 1], bins=100, range=extent)
        H_diff = H_real - H_fake
        X, Y = np.meshgrid(xedges, yedges)
        v_bound = np.max(np.abs(H_diff))

        hist_diff_fig = plt.figure(figsize=(5, 5))

        H_diff[H_diff == 0] = np.nan
        plt.pcolormesh(X, Y, H_diff.T, shading="auto", cmap="coolwarm", vmin=-v_bound, vmax=v_bound)

        plt.title(f"UMAP Histogram Difference (Real - Generated) at Step {self.step}")

        plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)  # shrink fig so cbar is visible
        # make new ax object for the cbar
        cbar_ax = hist_diff_fig.add_axes((0.87, 0.15, 0.02, 0.7))  # x, y, width, height
        plt.colorbar(cax=cbar_ax)

        plt.savefig(umap_path / f"step_{self.step}_hist_diff.jpg")

        with SummaryWriter(output_dir / "TensorBoard/UMAP", filename_suffix=f".step{self.step}") as w:
            w.add_figure("UMAP Scatter", scatter_fig, self.step)
            w.add_figure("UMAP Histogram", hexbin_fig, self.step)
            w.add_figure("UMAP Histogram Difference", hist_diff_fig, self.step)

        plt.close("all")
