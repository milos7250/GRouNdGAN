import logging
from typing import TYPE_CHECKING

import pytest

from loggers import setup_logger

from . import pytestmark  # pyright: ignore[reportUnusedImport]  # noqa: F401
from .resources.constants import CAUSAL_GRAPH_FILE, TRAIN_FILE

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path
    from typing import TypeAlias

    from gans import GAN, CausalGAN, ConditionalCatGAN, ConditionalProjGAN


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # items.sort(key=lambda item: "keyword" in item.nodeid, reverse=True) puts tests with "keyword" in their nodeid at the beginning of the list, so they run first
    # items.sort(key=lambda item: "keyword" not in item.nodeid, reverse=True) puts tests with "keyword" in their nodeid at the end of the list, so they run last
    
    # Run loggers tests first to ensure errors in logging setup are caught early
    items.sort(key=lambda item: "test_loggers" in item.nodeid, reverse=True)
    # Run test resource availability tests early since they are critical for other tests to run successfully
    items.sort(key=lambda item: "test_resources" in item.nodeid, reverse=True)
    
    # Run dicts tests early since they are used by many other tests
    items.sort(key=lambda item: "test_dicts" not in item.nodeid, reverse=True)
    items.sort(key=lambda item: "test_gans" not in item.nodeid, reverse=True)
    items.sort(key=lambda item: "test_trainers" not in item.nodeid, reverse=True)
    # Run long tests last since they take a long time to run
    items.sort(key=lambda item: "long" not in item.nodeid, reverse=True)
    # Run DDP trainer tests very last
    items.sort(key=lambda item: "TestDDPTrainers" not in item.nodeid, reverse=True)


MakeGanCheckpoint: "TypeAlias" = "Callable[[GAN], Path]"


@pytest.fixture
def make_gan_checkpoint(tmp_path: "Path") -> MakeGanCheckpoint:
    """
    Provides a fixture callable that takes a GAN instance, saves its checkpoint to a temporary file, and returns the
    path to the checkpoint.
    """

    def _make_gan_checkpoint(gan: "GAN") -> "Path":
        """
        Saves a checkpoint of the given GAN instance to a temporary file and returns the path to the checkpoint.
        """
        checkpoint_path = tmp_path / f"{gan.__class__.__name__}_checkpoint.pth"
        gan.save(checkpoint_path)
        return checkpoint_path

    return _make_gan_checkpoint


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)], ids=["cpu", "cuda"])
def device(request: pytest.FixtureRequest) -> "Generator[str, None, None]":
    """
    Provides a device string for testing, either "cpu" or "cuda". The "cuda" option is marked with @pytest.mark.gpu
    to allow selective test runs.
    """
    device = request.param
    yield device
    if device == "cuda":
        from torch.cuda import empty_cache

        empty_cache()  # Clear GPU memory after tests that use CUDA


@pytest.fixture(params=[False, True], ids=["no_compile", "compile"])
def compile(request: pytest.FixtureRequest, caplog: pytest.LogCaptureFixture) -> "Generator[bool, None, None]":
    """
    Provides a boolean indicating whether to compile the model with torch.compile. If True, the fixture will reset
    torch._dynamo after the test to avoid side effects on other tests.
    """
    _compile: bool = request.param
    yield _compile
    if _compile:
        import torch

        with caplog.at_level(logging.INFO):
            setup_logger("compile_fixture").info("Resetting torch._dynamo after test")
            torch._dynamo.reset()


@pytest.fixture
def gan(device: str) -> "GAN":
    """Provides a simple GAN instance for testing. Uses the provided device fixture to set the device."""
    from gans import GAN

    return GAN(
        genes_no=991,
        batch_size=32,
        latent_dim=10,
        gen_layers=[64, 128],
        crit_layers=[128, 64],
        device=device,
    )


@pytest.fixture(params=[pytest.param("cuda", marks=pytest.mark.gpu)], ids=["cuda"], scope="session")
def gan_complex_factory() -> "Callable[[], GAN]":
    """Provides a more complex GAN instance for testing. This fixture is marked with @pytest.mark.gpu."""
    from gans import GAN

    def _get_gan_complex() -> "GAN":
        return GAN(
            genes_no=991,
            batch_size=128,
            latent_dim=128,
            gen_layers=[256, 512, 1024],
            crit_layers=[1024, 512, 256],
            device="cuda",
        )

    return _get_gan_complex


@pytest.fixture(params=[pytest.param("cuda", marks=pytest.mark.gpu)], ids=["cuda"])
def gan_complex(gan_complex_factory: "Callable[[], GAN]") -> "Generator[GAN, None, None]":
    """Provides a more complex GAN instance for testing. This fixture is marked with @pytest.mark.gpu."""
    from torch.cuda import empty_cache

    yield gan_complex_factory()

    empty_cache()  # Clear GPU memory after tests that use this complex GAN


@pytest.fixture
def causalgan(gan: "GAN", make_gan_checkpoint: MakeGanCheckpoint) -> "CausalGAN":
    """Provides a simple CausalGAN instance for testing, using a checkpoint from a simple GAN."""
    from gans import CausalGAN

    cc_checkpoint = make_gan_checkpoint(gan)

    with open(CAUSAL_GRAPH_FILE, "rb") as f:
        import pickle

        causal_graph = pickle.load(f)

    return CausalGAN(
        genes_no=991,
        batch_size=32,
        latent_dim=10,
        noise_per_gene=1,
        depth_per_gene=1,
        width_per_gene=1,
        cc_latent_dim=10,
        cc_layers=[64, 128],
        cc_pretrained_checkpoint=cc_checkpoint,
        crit_layers=[128, 64],
        causal_graph=causal_graph,
        labeler_layers=[64, 128],
        device=gan.device,
    )


@pytest.fixture(scope="session")
def conditional_labels_and_ratios() -> tuple[int, list[float]]:
    """
    Provides the number of classes and their ratios for conditional GAN tests, extracted from the training data.
    """
    # Get the number of classes and their ratios from the training data
    import anndata as ad

    adata = ad.read_h5ad(TRAIN_FILE, backed="r")
    num_classes = adata.uns["clusters_no"]
    # Need to sort the label ratios according to the cluster labels (assuming they are integers starting from 0)
    label_ratios: list[float] = [adata.uns["cluster_ratios"][str(i)] for i in range(num_classes)]

    return num_classes, label_ratios


@pytest.fixture
def conditional_cat_gan(device: str, conditional_labels_and_ratios: tuple[int, list[float]]) -> "ConditionalCatGAN":
    """Provides a simple ConditionalCatGAN instance for testing."""
    from gans import ConditionalCatGAN

    num_classes, label_ratios = conditional_labels_and_ratios
    return ConditionalCatGAN(
        genes_no=991,
        batch_size=32,
        latent_dim=10,
        gen_layers=[64, 128],
        crit_layers=[128, 64],
        num_classes=num_classes,
        label_ratios=label_ratios,
        device=device,
    )


@pytest.fixture(params=[pytest.param("cuda", marks=pytest.mark.gpu)], ids=["cuda"])
def conditional_cat_gan_complex(
    conditional_labels_and_ratios: tuple[int, list[float]],
) -> "Generator[ConditionalCatGAN, None, None]":
    """Provides a more complex ConditionalCatGAN instance for testing. This fixture is marked with @pytest.mark.gpu."""
    from torch.cuda import empty_cache

    from gans import ConditionalCatGAN

    num_classes, label_ratios = conditional_labels_and_ratios
    yield ConditionalCatGAN(
        genes_no=991,
        batch_size=128,
        latent_dim=128,
        gen_layers=[256, 512, 1024],
        crit_layers=[1024, 512, 256],
        num_classes=num_classes,
        label_ratios=label_ratios,
        device="cuda",
    )

    empty_cache()  # Clear GPU memory after tests that use this complex ConditionalCatGAN


@pytest.fixture
def conditional_proj_gan(device: str, conditional_labels_and_ratios: tuple[int, list[float]]) -> "ConditionalProjGAN":
    """Provides a simple ConditionalProjGAN instance for testing."""
    from gans import ConditionalProjGAN

    num_classes, label_ratios = conditional_labels_and_ratios

    return ConditionalProjGAN(
        genes_no=991,
        batch_size=32,
        latent_dim=10,
        gen_layers=[64, 128],
        crit_layers=[128, 64],
        num_classes=num_classes,
        label_ratios=label_ratios,
        device=device,
    )


@pytest.fixture(params=[pytest.param("cuda", marks=pytest.mark.gpu)], ids=["cuda"])
def conditional_proj_gan_complex(
    conditional_labels_and_ratios: tuple[int, list[float]],
) -> "Generator[ConditionalProjGAN, None, None]":
    """Provides a more complex ConditionalProjGAN instance for testing. This fixture is marked with @pytest.mark.gpu."""
    from torch.cuda import empty_cache

    from gans import ConditionalProjGAN

    num_classes, label_ratios = conditional_labels_and_ratios
    yield ConditionalProjGAN(
        genes_no=991,
        batch_size=128,
        latent_dim=128,
        gen_layers=[256, 512, 1024],
        crit_layers=[1024, 512, 256],
        num_classes=num_classes,
        label_ratios=label_ratios,
        device="cuda",
    )

    empty_cache()  # Clear GPU memory after tests that use this complex ConditionalProjGAN
