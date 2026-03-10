from typing import TYPE_CHECKING

import pytest
from pytest_lazy_fixtures import lf

from . import pytestmark  # pyright: ignore[reportUnusedImport]  # noqa: F401

if TYPE_CHECKING:
    from pathlib import Path

    from gans import GAN

    from .conftest import MakeGanCheckpoint


@pytest.mark.parametrize(
    "_gan",
    [lf("gan"), lf("causalgan"), lf("conditional_cat_gan"), lf("conditional_proj_gan")],
    ids=["GAN", "CausalGAN", "ConditionalCatGAN", "ConditionalProjGAN"],
)
class TestGanMethods:
    def test_gans(
        self, _gan: "GAN", tmp_path: "Path", make_gan_checkpoint: "MakeGanCheckpoint", request: pytest.FixtureRequest
    ) -> None:
        gan = _gan

        noise = gan.generate_noise(gan.batch_size, gan.latent_dim, device=gan.device)
        assert noise.shape == (gan.batch_size, gan.latent_dim)

        cells, _ = gan.generate_cells(gan.batch_size)
        assert cells.shape == (gan.batch_size, gan.genes_no)

        checkpoint_path = make_gan_checkpoint(gan)
        assert checkpoint_path.exists()

        cells, _ = gan.generate_cells(gan.batch_size, checkpoint_path)
        assert cells.shape == (gan.batch_size, gan.genes_no)

        gan.log_tensorboard_graph(tmp_path)
        assert any(tmp_path.rglob("events.*")), "TensorBoard events file not found in output"


@pytest.mark.parametrize(
    "_gan",
    [lf("gan"), lf("causalgan"), lf("conditional_cat_gan"), lf("conditional_proj_gan")],
    ids=["GAN", "CausalGAN", "ConditionalCatGAN", "ConditionalProjGAN"],
)
class TestGenerateH5ad:
    @pytest.fixture(autouse=True)
    def __only_cpu(self, request: pytest.FixtureRequest) -> None:  # pyright: ignore[reportUnusedFunction]
        if request.node.get_closest_marker("gpu") is not None:
            pytest.skip("Always skipped.")

    def test_generate_h5ad_returns_anndata(self, _gan: "GAN") -> None:
        """Test that generate_h5ad returns AnnData object without save_path."""
        from scanpy import AnnData

        cells_no = 10
        result = _gan.generate_h5ad(cells_no)
        assert isinstance(result, AnnData)
        assert result.shape[0] == cells_no
        assert result.shape[1] == _gan.genes_no

    def test_generate_h5ad_with_gene_names(self, _gan: "GAN") -> None:
        """Test generate_h5ad with provided gene names."""
        cells_no = 10
        gene_names = [f"gene_{i}" for i in range(_gan.genes_no)]
        result = _gan.generate_h5ad(cells_no, gene_names=gene_names)
        assert result.var_names.tolist() == gene_names

    def test_generate_h5ad_saves_to_file(self, _gan: "GAN", tmp_path: "Path") -> None:
        """Test that generate_h5ad saves file and returns None."""
        cells_no = 10
        save_path = tmp_path / "generated.h5ad"
        result = _gan.generate_h5ad(cells_no, save_path)
        assert result is None
        assert save_path.exists()

    def test_generate_h5ad_with_reference_dataset(self, _gan: "GAN", tmp_path: "Path") -> None:
        """Test generate_h5ad with reference dataset for gene names."""
        import numpy as np
        from scanpy import AnnData

        cells_no = 10
        ref_path = tmp_path / "reference.h5ad"
        gene_names = [f"gene_{i}" for i in range(_gan.genes_no)]
        X = np.random.random_sample((cells_no, _gan.genes_no))
        ref_adata = AnnData(X)
        ref_adata.var_names = gene_names
        ref_adata.write(ref_path)

        result = _gan.generate_h5ad(cells_no, reference_dataset=ref_path)
        assert result.var_names.tolist() == gene_names

    def test_generate_h5ad_with_checkpoint(
        self, _gan: "GAN", tmp_path: "Path", make_gan_checkpoint: "MakeGanCheckpoint"
    ) -> None:
        """Test generate_h5ad with checkpoint loading."""
        from scanpy import AnnData

        cells_no = 10
        checkpoint_path = make_gan_checkpoint(_gan)
        result = _gan.generate_h5ad(cells_no, checkpoint=checkpoint_path)
        assert isinstance(result, AnnData)
        assert result.shape[0] == cells_no

    def test_generate_h5ad_obs_names_unique(self, _gan: "GAN") -> None:
        """Test that generated h5ad has unique observation names."""
        cells_no = 100
        result = _gan.generate_h5ad(cells_no)
        assert len(result.obs_names) == len(set(result.obs_names))
