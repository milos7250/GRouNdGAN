#!/usr/bin/env python3
"""Plot UMAP embeddings of real and generated cells, showing the smaller of two disjoint regions as a zoomed inset to reduce whitespace."""

import sys
from pathlib import Path

# Add parent folder to pythonpath so `src` is importable when run as a script.
sys.path.insert(0, Path(__file__).resolve().parent.parent.as_posix())

from src.loggers import setup_logger

logger = setup_logger("inset_scatterfig")

from typing import TYPE_CHECKING

import rich_click as click

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.figure import Figure
    from scipy import sparse


_CATEGORIES: tuple[str, ...] = ("real", "original generated", "improved generated")
_CATEGORY_COLORS: tuple[str, ...] = ("blue", "red", "green")
_CORNER_LOCATIONS: tuple[str, ...] = ("upper right", "upper left", "lower right", "lower left")
_LEGEND_LOCATIONS: tuple[str, ...] = (
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
)
_INSET_LOCATIONS: tuple[str, ...] = ("best", "upper right", "upper left", "lower right", "lower left", "center")


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
    tuple["sparse.csr_matrix", "sparse.csr_matrix", "sparse.csr_matrix"]
        A tuple containing:
        - real_cells.X : sparse matrix of real gene expression data
        - orig_fake_cells.X : sparse matrix of original simulated data, truncated to match real data row count
        - improved_fake_cells.X : sparse matrix of improved simulated data, truncated to match real data row count
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
        A sparse matrix of real cell data with shape (n_real_cells, n_features).
    orig_fake
        A sparse matrix of original fake/generated cell data with shape (n_fake_cells, n_features).
    improved_fake
        A sparse matrix of improved fake/generated cell data with shape (n_fake_cells, n_features).

    Returns
    -------
    tuple["np.ndarray", "np.ndarray", "np.ndarray"]
        The 2D UMAP embeddings of the real and fake data, in the form
        (real_embedding, orig_fake_embedding, improved_fake_embedding).
    """
    import numpy as np
    from umap import UMAP

    umap = UMAP(random_state=42, min_dist=0.0, n_jobs=1)
    umap.fit(real)  # ensure UMAP is fitted only once to preserve comparability
    real_embedding = np.array(umap.transform(real))
    orig_fake_embedding = np.array(umap.transform(orig_fake))
    improved_fake_embedding = np.array(umap.transform(improved_fake))

    return real_embedding, orig_fake_embedding, improved_fake_embedding


def _interleave(points_per_category: list["np.ndarray"]) -> tuple["np.ndarray", "np.ndarray"]:
    """
    Round-robin interleave points from each category so colours cycle on the scatter plot.

    Parameters
    ----------
    points_per_category
        List of ``(n_i, 2)`` arrays, one per category in registration order.

    Returns
    -------
    tuple["np.ndarray", "np.ndarray"]
        ``(interleaved_points, category_index)`` where ``interleaved_points`` has shape
        ``(sum(n_i), 2)`` and ``category_index`` maps each row back to its source category.
    """
    import numpy as np

    counts = [int(p.shape[0]) for p in points_per_category]
    total = sum(counts)
    if total == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.int64)
    width = int(points_per_category[0].shape[1])
    dtype = np.result_type(*points_per_category)
    out = np.empty((total, width), dtype=dtype)
    cat = np.empty(total, dtype=np.int64)
    idx = [0] * len(points_per_category)
    pos = 0
    while pos < total:
        progressed = False
        for k in range(len(points_per_category)):
            if idx[k] < counts[k]:
                out[pos] = points_per_category[k][idx[k]]
                cat[pos] = k
                idx[k] += 1
                pos += 1
                progressed = True
        if not progressed:
            break
    return out, cat


def split_into_regions(points: "np.ndarray") -> "np.ndarray":
    """
    Split 2D points into two spatial regions using HDBSCAN.

    The larger region (more points) is labelled ``0`` and the smaller region ``1``. HDBSCAN finds
    clusters of arbitrary shape and varying density without a fixed cluster count; noise and any
    extra clusters are assigned to the nearest of the two largest clusters. If fewer than two
    clusters are detected, all points are labelled ``0`` so the caller can fall back to a plain
    plot. Callers should additionally verify the two regions are well-separated before insetting.

    Parameters
    ----------
    points
        ``(n, 2)`` array of 2D embeddings. Not modified.

    Returns
    -------
    "np.ndarray"
        Integer labels of shape ``(n,)`` with values in ``{0, 1}``.
    """
    import numpy as np
    from sklearn.cluster import HDBSCAN

    n = points.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.int64)

    min_cluster_size = max(5, n // 100)
    raw = HDBSCAN(min_cluster_size=min_cluster_size, copy=True).fit_predict(points)
    present = [c for c in range(int(raw.max()) + 1) if (raw == c).any()]
    if len(present) < 2:
        return np.zeros(n, dtype=np.int64)

    counts = {c: int((raw == c).sum()) for c in present}
    main_label, inset_label = sorted(present, key=lambda c: counts[c], reverse=True)[:2]
    centroids = np.stack([points[raw == main_label].mean(axis=0), points[raw == inset_label].mean(axis=0)])
    out = np.ones(n, dtype=np.int64)
    out[raw == main_label] = 0
    other = (raw != main_label) & (raw != inset_label)
    if other.any():
        diff = points[other][:, None, :] - centroids[None, :, :]
        nearest = np.argmin(np.linalg.norm(diff, axis=2), axis=1)
        out[np.where(other)[0]] = nearest
    return out


def _region_bbox(points: "np.ndarray", margin_frac: float = 0.05) -> "np.ndarray":
    """
    Bounding box of ``points`` plus a symmetric fractional margin.

    Returns
    -------
    "np.ndarray"
        Array of shape ``(2, 2)`` as ``[[xmin, xmax], [ymin, ymax]]``.
    """
    import numpy as np

    if points.shape[0] == 0:
        return np.array([[0.0, 1.0], [0.0, 1.0]])
    extent = np.array([[points[:, 0].min(), points[:, 0].max()], [points[:, 1].min(), points[:, 1].max()]], dtype=float)
    margin = (extent[:, 1] - extent[:, 0]) * margin_frac
    extent[:, 0] -= margin[0]
    extent[:, 1] += margin[1]
    return extent


def _regions_well_separated(region0: "np.ndarray", region1: "np.ndarray", min_gap_frac: float = 0.05) -> bool:
    """
    Return True if the two regions' bounding boxes are separated by a meaningful gap.

    Used to avoid insetting when a single blob is over-segmented into adjacent pieces, since
    insetting only reduces whitespace when the two regions are genuinely disjoint.
    """
    import numpy as np

    if region0.shape[0] == 0 or region1.shape[0] == 0:
        return False
    b0 = _region_bbox(region0, margin_frac=0.0)
    b1 = _region_bbox(region1, margin_frac=0.0)
    gap_x = max(b0[0, 0], b1[0, 0]) - min(b0[0, 1], b1[0, 1])
    gap_y = max(b0[1, 0], b1[1, 0]) - min(b0[1, 1], b1[1, 1])
    overall = np.vstack([region0, region1])
    overall_extent = float(max(overall[:, 0].max() - overall[:, 0].min(), overall[:, 1].max() - overall[:, 1].min()))
    if overall_extent <= 0:
        return False
    return max(gap_x, gap_y) > min_gap_frac * overall_extent


def _sibling_inset_axes(
    fig: "Figure", main_ax, corner: str, inset_width: float, inset_height: float, pad: float = 0.02
):
    """
    Create a *sibling* inset axes (via ``fig.add_axes``) at ``corner`` of the main axes.

    A sibling (top-level) axes is used rather than ``inset_axes`` (which parents the inset under the
    main axes) for two reasons: (1) a child inset corrupts rasterization of the main axes' artists in
    the PDF backend, squishing rasterized points into a corner; (2) a sibling axes is drawn *after*
    the main axes, so its opaque white patch reliably covers the main axes' grid/points, giving the
    inset a clean white background. The position is computed directly (no locatable locator), so it
    is known before draw.
    """
    pos = main_ax.get_position()
    iw = inset_width * (pos.x1 - pos.x0)
    ih = inset_height * (pos.y1 - pos.y0)
    if corner == "upper right":
        left, bottom = pos.x1 - iw - pad, pos.y1 - ih - pad
    elif corner == "upper left":
        left, bottom = pos.x0 + pad, pos.y1 - ih - pad
    elif corner == "lower right":
        left, bottom = pos.x1 - iw - pad, pos.y0 + pad
    elif corner == "lower left":
        left, bottom = pos.x0 + pad, pos.y0 + pad
    else:  # center
        left = pos.x0 + (pos.x1 - pos.x0 - iw) / 2
        bottom = pos.y0 + (pos.y1 - pos.y0 - ih) / 2
    return fig.add_axes([left, bottom, iw, ih])


def _orient_inset_ticks(ax_inset, corner: str) -> None:
    """
    Put the inset's ticks on the sides that face the main axes interior.

    For a corner placement the interior-facing sides are those toward the main axes centre, e.g. an
    upper-right inset gets x ticks on the bottom and y ticks on the left. This keeps the inset's
    tick labels inside the main axes and away from the main axes' outer labels.
    """
    if corner == "upper right":
        ax_inset.xaxis.tick_bottom()
        ax_inset.yaxis.tick_left()
    elif corner == "upper left":
        ax_inset.xaxis.tick_bottom()
        ax_inset.yaxis.tick_right()
    elif corner == "lower right":
        ax_inset.xaxis.tick_top()
        ax_inset.yaxis.tick_left()
    elif corner == "lower left":
        ax_inset.xaxis.tick_top()
        ax_inset.yaxis.tick_right()
    else:  # center: fall back to the default bottom/left placement
        ax_inset.xaxis.tick_bottom()
        ax_inset.yaxis.tick_left()


def _best_inset_corner(main_points: "np.ndarray") -> tuple[str, dict[str, int]]:
    """
    Return the corner of the main axes containing the fewest points so the inset covers little data.

    Returns
    -------
    tuple[str, dict[str, int]]
        The chosen corner and a mapping from each corner name to the number of main-region points
        that fall in it.
    """

    counts: dict[str, int] = {loc: 0 for loc in _CORNER_LOCATIONS}
    if main_points.shape[0] == 0:
        return "upper right", counts
    xs = main_points[:, 0]
    ys = main_points[:, 1]
    x_mid = (xs.min() + xs.max()) / 2.0
    y_mid = (ys.min() + ys.max()) / 2.0
    counts = {
        "upper right": int(((xs >= x_mid) & (ys >= y_mid)).sum()),
        "upper left": int(((xs < x_mid) & (ys >= y_mid)).sum()),
        "lower right": int(((xs >= x_mid) & (ys < y_mid)).sum()),
        "lower left": int(((xs < x_mid) & (ys < y_mid)).sum()),
    }
    corner = min(_CORNER_LOCATIONS, key=lambda loc: (counts[loc], _CORNER_LOCATIONS.index(loc)))
    return corner, counts


def _scatter_interleaved(ax, points_per_category: list["np.ndarray"]) -> None:
    """Interleave the category points and scatter them on ``ax`` with cycled colours."""
    import numpy as np

    pts, cat = _interleave(points_per_category)
    colors = np.array(_CATEGORY_COLORS)[cat]
    ax.scatter(pts[:, 0], pts[:, 1], c=colors, s=3, edgecolor="none")


def _add_legend(ax, location: str) -> None:
    """Attach the shared real/original/improved legend to ``ax``."""
    handles = [
        ax.scatter([], [], c=_CATEGORY_COLORS[i], label=_CATEGORIES[i], s=3, edgecolor="none")
        for i in range(len(_CATEGORIES))
    ]
    ax.legend(handles=handles, loc=location, ncol=1, fontsize=8).set(zorder=5)


def plot_UMAP_inset(
    real_embedding: "np.ndarray",
    orig_fake_embedding: "np.ndarray",
    improved_fake_embedding: "np.ndarray",
    legend_location: str = "lower left",
    inset_loc: str = "best",
    inset_width: float = 0.35,
    inset_height: float = 0.35,
) -> "Figure":
    """
    Scatter the UMAP embeddings with the smaller of two disjoint regions shown as a zoomed inset.

    The main axes are cropped to the larger region's bounding box, eliminating the whitespace
    that the gap between the two regions would otherwise create. The smaller region is plotted on
    an inset axes zoomed to its own bounding box and placed in the least-dense corner of the main
    axes. If two disjoint regions cannot be detected, a plain full-extent UMAP scatter is produced
    instead.

    Parameters
    ----------
    real_embedding
        2D UMAP embedding of the real cells.
    orig_fake_embedding
        2D UMAP embedding of the original generated cells.
    improved_fake_embedding
        2D UMAP embedding of the improved generated cells.
    legend_location
        Location of the shared legend on the main axes. Defaults to ``"lower left"``.
    inset_loc
        Placement of the inset axes. ``"best"`` picks the corner with the fewest main points.
        Defaults to ``"best"``.
    inset_width
        Inset width as a fraction of the main axes width. Defaults to ``0.35``.
    inset_height
        Inset height as a fraction of the main axes height. Defaults to ``0.35``.

    Returns
    -------
    Figure
        A matplotlib Figure object for the scatter plot.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    all_per_category = [real_embedding, orig_fake_embedding, improved_fake_embedding]
    all_points = np.vstack(all_per_category)
    region = split_into_regions(all_points)
    has_inset = bool((region == 0).any() and (region == 1).any())
    if has_inset and not _regions_well_separated(all_points[region == 0], all_points[region == 1]):
        has_inset = False

    if has_inset:
        main_mask = region == 0
        inset_mask = region == 1
        logger.info(
            "Inset identified: main region %d points (midpoint %s), smaller region %d points (midpoint %s)",
            int(main_mask.sum()),
            all_points[main_mask].mean(axis=0).tolist(),
            int(inset_mask.sum()),
            all_points[inset_mask].mean(axis=0).tolist(),
        )
    else:
        logger.warning("Two well-separated disjoint regions were not detected; producing a plain UMAP plot.")

    plt.clf()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.subplots()

    if not has_inset:
        _scatter_interleaved(ax, all_per_category)
        extent = _region_bbox(all_points)
        ax.set_xlim(extent[0, 0], extent[0, 1])
        ax.set_ylim(extent[1, 0], extent[1, 1])
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_title("UMAP Projection of Real and Generated Cells")
        _add_legend(ax, legend_location)
        return fig

    n_real = real_embedding.shape[0]
    n_orig = orig_fake_embedding.shape[0]
    real_region = region[:n_real]
    orig_region = region[n_real : n_real + n_orig]
    imp_region = region[n_real + n_orig :]

    main_per_category = [
        real_embedding[real_region == 0],
        orig_fake_embedding[orig_region == 0],
        improved_fake_embedding[imp_region == 0],
    ]
    inset_per_category = [
        real_embedding[real_region == 1],
        orig_fake_embedding[orig_region == 1],
        improved_fake_embedding[imp_region == 1],
    ]
    main_points = np.vstack(main_per_category)
    inset_points = np.vstack(inset_per_category)
    main_extent = _region_bbox(main_points)
    inset_extent = _region_bbox(inset_points)

    ax.set_xlim(main_extent[0, 0], main_extent[0, 1])
    ax.set_ylim(main_extent[1, 0], main_extent[1, 1])
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    _scatter_interleaved(ax, main_per_category)
    ax.set_title("UMAP Projection of Real and Generated Cells")

    best_corner, corner_counts = _best_inset_corner(main_points)
    corner = best_corner if inset_loc == "best" else inset_loc
    logger.info("Inset corner point counts: %s; placing inset at '%s'", corner_counts, corner)
    # Sibling inset axes (drawn on top of the main axes) with an opaque white patch so the main
    # grid/points are hidden behind the inset.
    ax_inset = _sibling_inset_axes(fig, ax, corner, inset_width, inset_height)
    ax_inset.patch.set_facecolor("white")
    ax_inset.patch.set_alpha(1.0)
    _scatter_interleaved(ax_inset, inset_per_category)
    ax_inset.set_xlim(inset_extent[0, 0], inset_extent[0, 1])
    ax_inset.set_ylim(inset_extent[1, 0], inset_extent[1, 1])
    # Inset ticks on the sides facing the main axes interior (so labels stay inside and do not
    # overlap the main axes' outer labels).
    _orient_inset_ticks(ax_inset, corner)
    ax_inset.grid(True, linestyle="--", linewidth=0.5)
    ax_inset.set_axisbelow(True)
    ax_inset.tick_params(labelsize=6)

    _add_legend(ax, legend_location)
    return fig


def main(
    real_cells_path: Path,
    orig_fake_cells_path: Path,
    improved_fake_cells_path: Path,
    output_dir: Path,
    legend_location: str = "best",
    inset_loc: str = "best",
    inset_width: float = 0.35,
    inset_height: float = 0.35,
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

    logger.info("Plotting UMAP inset scatter plot")
    scatter_fig = plot_UMAP_inset(
        real_embedding,
        orig_fake_embedding,
        improved_fake_embedding,
        legend_location=legend_location,
        inset_loc=inset_loc,
        inset_width=inset_width,
        inset_height=inset_height,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    scatter_fig.savefig(output_dir / "UMAP Inset Scatter.png", transparent=False, facecolor=(1, 1, 1, 0))
    scatter_fig.savefig(output_dir / "UMAP Inset Scatter.pdf", transparent=False, facecolor=(1, 1, 1, 0))
    # Rasterize only the dense scatter collections (per-artist) for a compact PDF. Using
    # set_rasterization_zorder instead mis-renders the inset (child axes), squishing the
    # rasterized points into the lower-left corner.
    for ax in scatter_fig.axes:
        for coll in ax.collections:
            coll.set_rasterized(True)
    scatter_fig.savefig(output_dir / "UMAP Inset Scatter Rasterized.pdf", transparent=False, facecolor=(1, 1, 1, 0))
    logger.info(f"UMAP inset scatter plots saved to '{output_dir}'")
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
    help="Directory where the scatter plots will be saved.",
)
@click.option(
    "--legend-location",
    type=click.Choice(list(_LEGEND_LOCATIONS), case_sensitive=False),
    default="best",
    show_default=True,
    help="Location of the shared legend on the main axes.",
)
@click.option(
    "--inset-loc",
    type=click.Choice(list(_INSET_LOCATIONS), case_sensitive=False),
    default="best",
    show_default=True,
    help="Placement of the zoomed inset. 'best' picks the corner with the fewest main-region points.",
)
@click.option(
    "--inset-width",
    type=click.FloatRange(0.05, 1.0),
    default=0.35,
    show_default=True,
    help="Inset width as a fraction of the main axes width.",
)
@click.option(
    "--inset-height",
    type=click.FloatRange(0.05, 1.0),
    default=0.35,
    show_default=True,
    help="Inset height as a fraction of the main axes height.",
)
def cli(
    real: Path,
    orig: Path,
    improved: Path,
    out: Path,
    legend_location: str,
    inset_loc: str,
    inset_width: float,
    inset_height: float,
) -> None:
    """
    Plot UMAP embeddings of real test cells and both original and improved generated cells, showing the smaller of two disjoint regions as a zoomed inset to reduce whitespace.
    """
    main(real, orig, improved, out, legend_location, inset_loc, inset_width, inset_height)


if __name__ == "__main__":
    cli()
