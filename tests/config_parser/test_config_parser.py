import os
from configparser import NoOptionError
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.resources.constants import CAUSAL_GRAPH_FILE, TRAIN_FILE, VALID_FILE

from .. import pytestmark  # pyright: ignore[reportUnusedImport]  # noqa: F401
from ..training.test_trainers import check_num_outputs, check_short_outputs

if TYPE_CHECKING:
    from ..conftest import MakeGanCheckpoint


class TestTrain:
    def update_env_vars(self, output_dir: Path) -> None:
        os.environ["PYTEST_OUTPUT_DIR"] = str(output_dir)
        os.environ["PYTEST_TRAIN_FILE"] = str(TRAIN_FILE)
        os.environ["PYTEST_VALID_FILE"] = str(VALID_FILE)
        os.environ["PYTEST_TEST_FILE"] = str(VALID_FILE)  # Using the same file for validation and test for simplicity
        os.environ["PYTEST_CAUSAL_GRAPH_FILE"] = str(CAUSAL_GRAPH_FILE)

    def test_config_parser_defaults(self, tmp_path: Path) -> None:
        self.update_env_vars(tmp_path)

        from custom_parser import get_configparser

        cfg_parser = get_configparser()
        cfg_parser.read("tests/resources/configs/gan_train.cfg")

        assert cfg_parser.get("EXPERIMENT", "output directory") == os.environ["PYTEST_OUTPUT_DIR"]
        assert cfg_parser.get("Data", "train") == os.environ["PYTEST_TRAIN_FILE"]
        assert cfg_parser.get("Data", "validation") == os.environ["PYTEST_VALID_FILE"]
        assert cfg_parser.get("Data", "test") == os.environ["PYTEST_TEST_FILE"]

        # Test handling of missing environment variable with fallback
        assert cfg_parser.get("EXPERIMENT", "nonexistent variable", fallback="default_value") == "default_value"
        with pytest.raises(NoOptionError):
            cfg_parser.get("EXPERIMENT", "nonexistent variable")
        assert (
            cfg_parser.get("EXPERIMENT", "checkpoint", fallback="test") is None
        )  # FALLBACK VALUE IS NOT USED FOR EMPTY VALUES
        assert cfg_parser.get("EXPERIMENT", "checkpoint") is None

        # Also test new getpath method
        assert cfg_parser.getpath("EXPERIMENT", "nonexistent path", fallback="default_path") == Path("default_path")
        with pytest.raises(NoOptionError):
            cfg_parser.getpath("EXPERIMENT", "nonexistent path")
        assert cfg_parser.getpath("EXPERIMENT", "checkpoint", fallback="default_path") is None
        assert cfg_parser.getpath("EXPERIMENT", "checkpoint") is None
        assert cfg_parser.getpath("EXPERIMENT", "output directory") == Path(os.environ["PYTEST_OUTPUT_DIR"])

    def test_config_parsing(self, tmp_path: Path) -> None:
        from main import main

        config_path = Path("tests/resources/configs/gan_train.cfg")
        output_dir = tmp_path / "results/GAN"

        self.update_env_vars(output_dir)

        main(config=config_path)

        assert (output_dir / config_path.name).exists()

    def test_gan_factory(self, tmp_path: Path) -> None:
        from main import main

        config_path = Path("tests/resources/configs/gan_train.cfg")
        output_dir = tmp_path / "results/GAN"

        self.update_env_vars(output_dir)

        main(config=config_path, train=True)

        assert (output_dir / config_path.name).exists()
        check_short_outputs(output_dir)

    def test_conditional_cat_gan_factory(self, tmp_path: Path) -> None:
        from main import main

        config_path = Path("tests/resources/configs/conditional_cat_gan_train.cfg")
        output_dir = tmp_path / "results/catGAN"

        self.update_env_vars(output_dir)

        main(config=config_path, train=True)

        assert (output_dir / config_path.name).exists()
        check_short_outputs(output_dir)

    def test_conditional_proj_gan_factory(self, tmp_path: Path) -> None:
        from main import main

        config_path = Path("tests/resources/configs/conditional_proj_gan_train.cfg")
        output_dir = tmp_path / "results/projGAN"

        self.update_env_vars(output_dir)

        main(config=config_path, train=True)

        assert (output_dir / config_path.name).exists()
        check_short_outputs(output_dir)

    def test_causal_gan_factory(self, tmp_path: Path) -> None:
        from main import main

        config_path = Path("tests/resources/configs/causalgan_train.cfg")
        output_dir = tmp_path / "results/CausalGAN"

        self.update_env_vars(output_dir)

        main(config=config_path, train=True)

        assert (output_dir / config_path.name).exists()
        check_short_outputs(output_dir / "CC")  # Check that CC also ran and produced outputs
        check_short_outputs(output_dir)

    def test_causal_gan_factory_with_checkpoint(self, tmp_path: Path, make_gan_checkpoint: "MakeGanCheckpoint") -> None:
        from gans import GAN
        from main import main

        config_path = Path("tests/resources/configs/causalgan_train_with_checkpoint.cfg")
        output_dir = tmp_path / "results/CausalGAN"

        gan = GAN(
            genes_no=991,
            batch_size=32,
            latent_dim=10,
            gen_layers=[64, 128],
            crit_layers=[128, 64],
            library_size=20_000,
        )

        cc_checkpoint = make_gan_checkpoint(gan)
        os.environ["PYTEST_CC_CHECKPOINT_FILE"] = str(cc_checkpoint)

        self.update_env_vars(output_dir)

        main(config=config_path, train=True)

        assert (output_dir / config_path.name).exists()
        check_short_outputs(output_dir)


class TestHyperopt:
    def update_env_vars(self, output_dir: Path) -> None:
        os.environ["PYTEST_OUTPUT_DIR"] = str(output_dir)
        os.environ["PYTEST_TRAIN_FILE"] = str(TRAIN_FILE)
        os.environ["PYTEST_VALID_FILE"] = str(VALID_FILE)
        os.environ["PYTEST_TEST_FILE"] = str(VALID_FILE)  # Using the same file for validation and test for simplicity
        os.environ["PYTEST_CAUSAL_GRAPH_FILE"] = str(CAUSAL_GRAPH_FILE)

    def test_config_resolving_for_hyperopt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import optuna

        from custom_parser import get_configparser
        from hyperparameter_optimization import resolve_hyperparameters

        config_path = Path("tests/resources/configs/gan_hyperopt.cfg")
        output_dir = tmp_path / "results/GAN_hyperopt"

        self.update_env_vars(output_dir)

        cfg_parser = get_configparser()
        cfg_parser.read(config_path)

        class MockTrial:
            def __init__(self) -> None:
                self.number = 0

            def suggest_int(self, name: str, low: int, high: int, log: bool = False, step: int = 1) -> int:
                return low + 1  # Always return the lower bound for simplicity

            def suggest_float(
                self, name: str, low: float, high: float, log: bool = False, step: float | None = None
            ) -> float:
                return low + 0.1  # Always return the lower bound for simplicity

        monkeypatch.setattr(optuna, "Trial", MockTrial)

        resolved_cfg_parser = get_configparser()
        resolved_cfg_parser.read(resolve_hyperparameters(cfg_parser, MockTrial()))  # pyright: ignore[reportArgumentType]

        assert resolved_cfg_parser.get("EXPERIMENT", "output directory") == str(output_dir / "0")
        assert resolved_cfg_parser.get("Model", "generator layers") == "17 33 49"
        assert resolved_cfg_parser.get("Model", "critic layers") == "49 33 17"
        assert resolved_cfg_parser.getint("Model", "latent dim") == 65
        assert resolved_cfg_parser.getint("Model", "lambda") == 2
        assert resolved_cfg_parser.getint("Training", "batch size") == 17
        assert resolved_cfg_parser.getint("Training", "critic iterations") == 2
        assert resolved_cfg_parser.getint("Training", "maximum steps") == 110
        assert resolved_cfg_parser.getfloat("Optimizer", "beta1") == pytest.approx(0.6)
        assert resolved_cfg_parser.getfloat("Optimizer", "beta2") == pytest.approx(1.0)
        assert resolved_cfg_parser.getfloat("Learning Rate", "generator initial") == pytest.approx(1e-5 + 0.1)
        assert resolved_cfg_parser.getfloat("Learning Rate", "generator final") == pytest.approx(1e-6)
        assert resolved_cfg_parser.getfloat("Learning Rate", "critic initial") == pytest.approx(1e-5 + 0.1)
        assert resolved_cfg_parser.getfloat("Learning Rate", "critic final") == pytest.approx(1e-6)

    @pytest.mark.parametrize(
        "config_name",
        ["gan_hyperopt.cfg", "causalgan_hyperopt.cfg", "causalgan_hyperopt_with_checkpoint.cfg"],
        ids=["GAN", "CausalGAN", "CausalGAN w/ckpt"],
    )
    def test_hyperopt(self, tmp_path: Path, config_name: str, make_gan_checkpoint: "MakeGanCheckpoint") -> None:
        from gans import GAN
        from main import main

        config_path = Path(f"tests/resources/configs/{config_name}")
        output_dir = tmp_path / "results/hyperopt"

        if config_name == "causalgan_hyperopt_with_checkpoint.cfg":
            gan = GAN(
                genes_no=991,
                batch_size=32,
                latent_dim=10,
                gen_layers=[64, 128],
                crit_layers=[128, 64],
                library_size=20_000,
            )

            cc_checkpoint = make_gan_checkpoint(gan)
            os.environ["PYTEST_CC_CHECKPOINT_FILE"] = str(cc_checkpoint)

        self.update_env_vars(output_dir)

        main(config=config_path, optimize_hyperparameters=True)

        assert (output_dir / config_path.name).exists()
        assert (output_dir / "optuna_study.db").exists()
        assert (output_dir / "optuna_stop.txt").exists()
        check_num_outputs(output_dir / "0", 3, 5, 15, 3)
        check_num_outputs(output_dir / "1", 3, 5, 15, 3)
        assert not (output_dir / "2").exists()  # Only 2 trials should have run
