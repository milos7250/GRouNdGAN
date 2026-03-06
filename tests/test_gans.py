
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
def test_gans(
    _gan: "GAN", tmp_path: "Path", make_gan_checkpoint: "MakeGanCheckpoint", request: pytest.FixtureRequest
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
