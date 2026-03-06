from abc import ABC
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt

from .gan import GANTrainer

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from optuna import Trial

    from gans.conditional_gan import ConditionalGAN

    from .dicts import GANTrainingArgs, SummaryArgs


class ConditionalGANTrainer(GANTrainer, ABC):
    def __init__(
        self,
        gan: "ConditionalGAN",
        train_file: "Path",
        valid_file: "Path",
        training_args: "GANTrainingArgs",
        summary_args: "SummaryArgs",
        output_dir: "Path",
        trial: "Trial | None" = None,
    ) -> None:
        super().__init__(gan, train_file, valid_file, training_args, summary_args, output_dir, trial)
        self.gan = gan

    def _init_umap(self) -> None:
        """Precompute UMAP embeddings for the validation set to speed up UMAP plotting during training."""
        super()._init_umap()
        self.real_labels = self.loaders["valid"].dataset.clusters.numpy()
    
    def _generate_umap_figures(self, fake_embedding: np.ndarray, fake_labels: np.ndarray | None) -> tuple["Figure", "Figure", "Figure"]:
        real_embedding = self.real_embedding
        real_labels = self.real_labels
        
        if fake_labels is None:
            raise ValueError("fake_labels cannot be None for ConditionalGANTrainer._generate_umap_figures")
        
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

        colormap = plt.get_cmap("nipy_spectral")
        colors = [colormap(i) for i in np.linspace(0, 1, self.gan.num_classes)]

        plt.clf()
        scatter_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1: Axes
        ax2: Axes

        for i in range(self.gan.num_classes):
            mask = real_labels[:] == i

            ax1.scatter(
                real_embedding[mask, 0],
                real_embedding[mask, 1],
                color=colors[i],
                marker="o",
                label="real_" + str(i),
            )

        ax1.legend(loc="lower left", numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
        ax1.set_title("Real cells")
        ax1.set_xlim(extent[0])
        ax1.set_ylim(extent[1])

        for i in range(self.gan.num_classes):
            mask = fake_labels[:] == i
            ax2.scatter(
                fake_embedding[mask, 0],
                fake_embedding[mask, 1],
                color=colors[i],
                marker="o",
                label="generated_" + str(i),
            )

        ax2.legend(loc="lower left", numpoints=1, ncol=3, fontsize=8, bbox_to_anchor=(0, 0))
        ax2.set_title("Generated cells")
        ax2.set_xlim(extent[0])
        ax2.set_ylim(extent[1])

        scatter_fig.suptitle(f"UMAP Projection of Real and Generated Cells at Step {self.step}")

        hexbin_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1: Axes
        ax2: Axes
        
        ax1.hexbin(
            real_embedding[:, 0], real_embedding[:, 1], mincnt=1, linewidths=0.0, extent=extent.flatten(), cmap="Reds" # pyright: ignore[reportArgumentType]
        )
        ax1.set_title("Real Cells")
        plt.colorbar(ax1.collections[0], ax=ax1)

        ax2.hexbin(
            fake_embedding[:, 0], fake_embedding[:, 1], mincnt=1, linewidths=0.0, extent=extent.flatten(), cmap="Reds" # pyright: ignore[reportArgumentType]
        )
        ax2.set_title("Generated Cells")
        plt.colorbar(ax2.collections[0], ax=ax2)
        plt.suptitle(f"UMAP Histograms at Step {self.step}")

        H_real, xedges, yedges = np.histogram2d(real_embedding[:, 0], real_embedding[:, 1], bins=80, range=extent)
        H_fake, _, _ = np.histogram2d(fake_embedding[:, 0], fake_embedding[:, 1], bins=80, range=extent)
        H_real[H_real == 0] = np.nan
        H_rel = H_real / (H_real + H_fake)  # relative density difference
        X, Y = np.meshgrid(xedges, yedges)
        v_bound = np.nanmax(np.abs(H_rel - 0.5))

        hist_rel_abun_fig = plt.figure(figsize=(5, 5))
        plt.pcolormesh(X, Y, H_rel.T, shading="auto", cmap="coolwarm", vmin=0.5 - v_bound, vmax=0.5 + v_bound)

        plt.title("UMAP Histogram Relative Abundance of Real Cells")

        plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)  # shrink fig so cbar is visible
        # make new ax object for the cbar
        cbar_ax = hist_rel_abun_fig.add_axes((0.87, 0.15, 0.02, 0.7))  # x, y, width, height
        plt.colorbar(cax=cbar_ax)
        
        return scatter_fig, hexbin_fig, hist_rel_abun_fig