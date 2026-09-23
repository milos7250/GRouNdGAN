"""Opt-in numerical comparison with the pinned upstream implementation."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

UPSTREAM_URL = "https://github.com/Emad-COMBINE-lab/GRouNdGAN.git"
UPSTREAM_COMMIT = "2df087f"
UPSTREAM_IMAGE = "docker://yazdanz/groundgan:GRouNdGAN-toolkit"
CONFIG = Path("tests/resources/configs/causalgan_train.cfg")
GRAPH = Path("tests/resources/processed/causal_graph.pkl")
WORKER = Path(__file__).with_name("causal_gan_upstream_worker.py")


def _clone_upstream(pytestconfig: pytest.Config) -> Path:
    cache_root = Path(pytestconfig.cache.makedir("causal_gan_upstream"))
    checkout = cache_root / UPSTREAM_COMMIT

    if checkout.exists():
        current = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if current.startswith(UPSTREAM_COMMIT):
            return checkout

    if checkout.exists():
        pytest.fail(f"Cached upstream checkout exists but is not {UPSTREAM_COMMIT}: {checkout}")

    subprocess.run(
        ["git", "clone", "--no-checkout", UPSTREAM_URL, str(checkout)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "checkout", "--detach", UPSTREAM_COMMIT],
        check=True,
        capture_output=True,
        text=True,
    )
    return checkout


@pytest.fixture(scope="session")
def upstream_checkout(pytestconfig: pytest.Config) -> Path:
    return _clone_upstream(pytestconfig)


def _make_small_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    """Create a compact fixture for strict dense/sparse comparison."""
    with GRAPH.open("rb") as file:
        source_graph = pickle.load(file)

    selected_genes = list(source_graph)[: max(1, len(source_graph) // 2)]
    selected_regulators = sorted({regulator for gene in selected_genes for regulator in source_graph[gene]})
    node_mapping = {node: index for index, node in enumerate(selected_genes + selected_regulators)}
    small_graph = {
        node_mapping[gene]: {node_mapping[regulator] for regulator in source_graph[gene]}
        for gene in selected_genes
    }

    small_graph_path = tmp_path / "causal_graph.pkl"
    with small_graph_path.open("wb") as file:
        pickle.dump(small_graph, file)

    small_config_path = tmp_path / "causalgan_train.cfg"
    small_config_path.write_text(
        CONFIG.read_text().replace("number of genes = 991", f"number of genes = {len(node_mapping)}")
    )
    return small_config_path, small_graph_path


@pytest.mark.upstream
def test_sparse_causal_generator_matches_upstream(
    tmp_path: Path,
    upstream_checkout: Path,
    compile: bool,
    device: str,
) -> None:
    """Verify sparse forward calculations against the pinned dense implementation."""

    import numpy as np

    from loggers import setup_logger

    logger = setup_logger("causal_gan_upstream_comparison")
    config, graph = _make_small_fixtures(tmp_path)
    checkpoint = tmp_path / "canonical_upstream.pt"
    reference = tmp_path / "upstream_output.npy"
    reference_gradients = tmp_path / "upstream_gradients.npz"
    local_gradients = tmp_path / "local_gradients.npz"

    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    local_compile_args = ["--compile"] if compile else []
    device_args = ["--device", device]

    logger.info(f"Loading upstream code from {upstream_checkout}")
    subprocess.run(
        [
            "apptainer",
            "exec",
            "--nv",
            UPSTREAM_IMAGE,
            "python3.9",
            str(WORKER.resolve()),
            "export-upstream",
            "--upstream",
            str(upstream_checkout.resolve()),
            "--config",
            str(config.resolve()),
            "--graph",
            str(graph.resolve()),
            "--checkpoint",
            str(checkpoint.resolve()),
            "--output",
            str(reference.resolve()),
            "--gradients",
            str(reference_gradients.resolve()),
            *device_args,
        ],
        check=True,
        env=environment,
    )

    logger.info("Loading local sparse implementation")
    subprocess.run(
        [
            sys.executable,
            str(WORKER.resolve()),
            "run-local",
            "--config",
            str(config.resolve()),
            "--graph",
            str(graph.resolve()),
            "--checkpoint",
            str(checkpoint.resolve()),
            "--output",
            str((tmp_path / "local_output.npy").resolve()),
            "--gradients",
            str(local_gradients.resolve()),
            *local_compile_args,
            *device_args,
        ],
        check=True,
        env=environment,
    )

    expected = np.load(reference)
    actual = np.load(tmp_path / "local_output.npy")
    # The implementations run in different PyTorch environments, so allow small
    # floating-point differences while still detecting meaningful calculation changes.
    np.testing.assert_allclose(actual, expected, rtol=1e-7)

    expected_gradients = np.load(reference_gradients)
    actual_gradients = np.load(local_gradients)
    assert set(actual_gradients.files) == set(expected_gradients.files)
    for name in expected_gradients.files:
        np.testing.assert_allclose(actual_gradients[name], expected_gradients[name], rtol=1e-7)
