import os
from pathlib import Path

from tests.resources.constants import CAUSAL_GRAPH_FILE, TRAIN_FILE, VALID_FILE


class TestConfigParser:
    def test_gan_train(self, tmp_path: Path) -> None:
        from main import main

        config_path = Path("tests/resources/configs/gan_train.cfg")
        output_dir = tmp_path / "results/GAN"

        os.environ["PYTEST_OUTPUT_DIR"] = str(output_dir)
        os.environ["PYTEST_TRAIN_FILE"] = str(TRAIN_FILE)
        os.environ["PYTEST_VALID_FILE"] = str(VALID_FILE)
        os.environ["PYTEST_TEST_FILE"] = str(VALID_FILE)  # Using the same file for validation and test for simplicity
        os.environ["PYTEST_CAUSAL_GRAPH_FILE"] = str(CAUSAL_GRAPH_FILE)

        main(config=config_path)

        assert (output_dir / "config_used.cfg").exists()
        