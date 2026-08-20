from src.loggers import setup_logger

logger = setup_logger("common_scatterfig")

from pathlib import Path
from typing import TYPE_CHECKING

import rich_click as click

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.figure import Figure
    from scipy import sparse


def read_datasets(
    test_cells_path: Path, orig_fake_cells_path: Path, improved_fake_cells_path: Path
) -> tuple["sparse.csr_matrix", "sparse.csr_matrix", "sparse.csr_matrix"]:
    """
    Load real and simulated (fake) gene expression datasets.

    Parameters
    ----------
    test_cells_path
        Path to the H5AD file containing real cell data.
    orig_fake_cells_path
        Path to the H5AD file containing original simulated gene expression data.
    improved_fake_cells_path
        Path to the H5AD file containing improved simulated gene expression data.

    Returns
    -------
    Tuple["sparse.csr_matrix", "sparse.csr_matrix", "sparse.csr_matrix"]
        A tuple containing:
        - real_cells.X : NumPy array of real gene expression data
        - orig_fake_cells.X : NumPy array of original simulated gene expression data, truncated to match real data row count
        - improved_fake_cells.X : NumPy array of improved simulated gene expression data, truncated to match real data row count
    """
    import scanpy as sc
    from scipy import sparse

    real_cells = sc.read_h5ad(test_cells_path)
    orig_fake_cells = sc.read_h5ad(orig_fake_cells_path)
    improved_fake_cells = sc.read_h5ad(improved_fake_cells_path)

    real_cells = sparse.csr_matrix(real_cells.X)
    orig_fake_cells = sparse.csr_matrix(orig_fake_cells.X)
    improved_fake_cells = sparse.csr_matrix(improved_fake_cells.X)

    no_of_cells = int(min(real_cells.shape[0], orig_fake_cells.shape[0], improved_fake_cells.shape[0]))  # pyright: ignore[reportOptionalSubscript,reportUnknownArgumentType]
    real_cells = real_cells[:no_of_cells, :]
    orig_fake_cells = orig_fake_cells[:no_of_cells, :]
    improved_fake_cells = improved_fake_cells[:no_of_cells, :]

    return real_cells, orig_fake_cells, improved_fake_cells


def get_UMAP_embeddings(
    real: "sparse.csr_matrix",
    orig_fake: "sparse.csr_matrix",
    improved_fake: "sparse.csr_matrix",
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """
    Compute UMAP embeddings for real and fake cell data.

    Parameters
    ----------
    real
        A NumPy array of real cell data with shape (n_real_cells, n_features).
    orig_fake
        A NumPy array of original fake/generated cell data with shape (n_fake_cells, n_features).
    improved_fake
        A NumPy array of improved fake/generated cell data with shape (n_fake_cells, n_features).

    Returns
    -------
    tuple["np.ndarray", "np.ndarray", "np.ndarray"]
        A tuple containing the 2D UMAP embeddings of the real and fake data,
        in the form (real_embedding, orig_fake_embedding, improved_fake_embedding).
    """
    import numpy as np
    from umap import UMAP

    umap = UMAP(random_state=42, min_dist=0.0, n_jobs=1)
    umap.fit(real)  # ensure UMAP is fitted only once to preserve comparability
    real_embedding = np.array(umap.transform(real))
    orig_fake_embedding = np.array(umap.transform(orig_fake))
    improved_fake_embedding = np.array(umap.transform(improved_fake))

    return real_embedding, orig_fake_embedding, improved_fake_embedding


def plot_UMAP(
    real_embedding: "np.ndarray",
    orig_fake_embedding: "np.ndarray",
    improved_fake_embedding: "np.ndarray",
    legend_location: str = "lower left",
) -> "Figure":
    """
    Perform UMAP embedding on real and fake cell data and save a scatter plot.

    Parameters
    ----------
    real
        A NumPy array of real cell data with shape (n_real_cells, n_features).
    orig_fake
        A NumPy array of original fake/generated cell data with shape (n_fake_cells, n_features).
    improved_fake
        A NumPy array of improved fake/generated cell data with shape (n_fake_cells, n_features).
    legend_location
        Location of the plot legend. Defaults to ``"lower left"``.
    output_dir

    Returns
    -------
    Figure
        A matplotlib Figure object for the scatter plot
    """

    import matplotlib.pyplot as plt
    import numpy as np

    extent = np.array([
        [
            min(min(real_embedding[:, 0]), min(orig_fake_embedding[:, 0]), min(improved_fake_embedding[:, 0])),
            max(max(real_embedding[:, 0]), max(orig_fake_embedding[:, 0]), max(improved_fake_embedding[:, 0])),
        ],
        [
            min(min(real_embedding[:, 1]), min(orig_fake_embedding[:, 1]), min(improved_fake_embedding[:, 1])),
            max(max(real_embedding[:, 1]), max(orig_fake_embedding[:, 1]), max(improved_fake_embedding[:, 1])),
        ],
    ])
    margin = np.array(extent[:, 1] - extent[:, 0]) * 0.05  # 5% margin
    extent[:, 0] -= margin[0]
    extent[:, 1] += margin[1]

    plt.clf()
    scatter_fig = plt.figure(figsize=(5, 5))
    
    # Interweave points for cycled colors
    points = np.empty(
        (real_embedding.shape[0] + orig_fake_embedding.shape[0] + improved_fake_embedding.shape[0], real_embedding.shape[1]), dtype=real_embedding.dtype
    )
    points[0::3] = real_embedding
    points[1::3] = orig_fake_embedding
    points[2::3] = improved_fake_embedding
    colors = np.array(["blue", "red", "green"] * (points.shape[0] // 3 + 1))[: points.shape[0]]

    plt.scatter(
        points[:, 0],
        points[:, 1],
        c=colors,
        s=3,
        edgecolor="none",
    )

    # Create artificial legend handles for the scatter plot
    a1 = plt.scatter(
        [],
        [],
        c="blue",
        label="real",
        s=3,
        edgecolor="none",
    )
    a2 = plt.scatter(
        [],
        [],
        c="red",
        label="original generated",
        s=3,
        edgecolor="none",
    )
    a3 = plt.scatter(
        [],
        [],
        c="green",
        label="improved generated",
        s=3,
        edgecolor="none",
    )
    plt.legend(handles=[a1, a2, a3], loc=legend_location, ncol=1, fontsize=8).set(zorder=5)

    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.set_axisbelow(True)
    plt.xlim(extent[0, 0], extent[0, 1])
    plt.ylim(extent[1, 0], extent[1, 1])
    plt.title("UMAP Projection of Real and Generated Cells")

    return scatter_fig


def main(
    real_cells_path: Path,
    orig_fake_cells_path: Path,
    improved_fake_cells_path: Path,
    output_dir: Path,
    legend_location: str = "lower left",
) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "savefig.dpi": 600,
        "legend.facecolor": (1.0, 1.0, 1.0, 0.0),
    })

    logger.info("Reading datasets")
    real_cells, orig_fake_cells, improved_fake_cells = read_datasets(
        real_cells_path, orig_fake_cells_path, improved_fake_cells_path
    )

    logger.info("Computing UMAP embeddings")
    real_embedding, orig_fake_embedding, improved_fake_embedding = get_UMAP_embeddings(
        real_cells, orig_fake_cells, improved_fake_cells
    )

    logger.info("Plotting UMAP scatter plot")
    scatter_fig = plot_UMAP(
        real_embedding,
        orig_fake_embedding,
        improved_fake_embedding,
        legend_location=legend_location,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    scatter_fig.savefig(output_dir / "UMAP Scatter.png")
    scatter_fig.savefig(output_dir / "UMAP Scatter.pdf")
    scatter_fig.axes[0].set_rasterization_zorder(1.1)
    scatter_fig.savefig(output_dir / "UMAP Scatter Rasterized.pdf")
    logger.info(f"UMAP scatter plots saved to '{output_dir}'")
    plt.close("all")


@click.command()
@click.option(
    "--real",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        path_type=Path,
    ),
    required=True,
    help="Path to the H5AD file containing real cell data.",
)
@click.option(
    "--orig",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        path_type=Path,
    ),
    required=True,
    help="Path to the H5AD file containing original simulated gene expression data.",
)
@click.option(
    "--improved",
    type=click.Path(
        exists=True,
        dir_okay=False,
        file_okay=True,
        readable=True,
        path_type=Path,
    ),
    required=True,
    help="Path to the H5AD file containing improved simulated gene expression data.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=True, file_okay=False, writable=True, path_type=Path),
    required=True,
    help="Directory where the evaluation results will be saved.",
)
@click.option(
    "--legend-location",
    type=click.Choice(
        [
            "best",
            "upper right",
            "upper left",
            "lower left",
            "lower right",
            "right",
            "center left",
            "center right",
            "lower center",
            "upper center",
            "center",
        ],
        case_sensitive=False,
    ),
    default="best",
    show_default=True,
    help="Location of the plot legend.",
)
def cli(real: Path, orig: Path, improved: Path, out: Path, legend_location: str) -> None:
    """
    Plot UMAP embeddings of real test cells and both original and improved generated cells, saving the results to the specified output directory.
    """
    main(real, orig, improved, out, legend_location)


if __name__ == "__main__":
    cli()
