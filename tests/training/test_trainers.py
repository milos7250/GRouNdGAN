import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from loggers import setup_logger
from tests.resources.constants import CAUSAL_GRAPH_FILE, TRAIN_FILE, VALID_FILE

from .. import pytestmark  # pyright: ignore[reportUnusedImport]  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import Callable

    from gans import GAN, CausalGAN, ConditionalCatGAN, ConditionalProjGAN


@pytest.fixture(scope="session", params=[pytest.param("cuda", marks=pytest.mark.gpu)], ids=["cuda"])
def trained_gan(gan_complex_factory: "Callable[[], GAN]", tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest) -> Path:
    """
    Provides a fixture that returns the path to a checkpoint of a GAN that has been trained for a long time. The
    checkpoint is saved to a temporary directory and the path is cached in pytest's cache to avoid retraining on
    subsequent test runs.
    """

    from training.dicts import GANTrainingArgs, SummaryArgs
    from training.gan import GANTrainer

    tmp_path = tmp_path_factory.mktemp("trained_gan")
    logger = setup_logger("trained_gan_fixture")
    
    gan_complex = gan_complex_factory()

    # Look for existing checkpoint in pytest cache to avoid retraining if possible
    params = {
        "genes_no": gan_complex.genes_no,
        "batch_size": gan_complex.batch_size,
        "latent_dim": gan_complex.latent_dim,
        "gen_layers": gan_complex.gen_layers,
        "crit_layers": gan_complex.crit_layers,
    }
    cache = request.config.cache
    checkpoint_dir = Path(cache.mkdir("gan_checkpoints"))
    checkpoint_name = f"trained_gan_genes-no{params['genes_no']}_batch-size{params['batch_size']}_latent-dim{params['latent_dim']}_gen-layers{'-'.join(map(str, params['gen_layers']))}_crit-layers{'-'.join(map(str, params['crit_layers']))}.pth"  # pyright: ignore[reportArgumentType]
    if (checkpoint_path := (checkpoint_dir / checkpoint_name)).exists():
        try:
            gan_complex.load(checkpoint_path)
        except RuntimeError as e:
            logger.warning(
                f"Failed to load cached GAN checkpoint at {checkpoint_path} due to parameter mismatch: {e}. Will retrain."
            )
        else:
            logger.info(f"Using cached trained GAN checkpoint from {checkpoint_path}")
            return checkpoint_path

    gan_trainer = GANTrainer(
        gan=gan_complex,
        train_file=TRAIN_FILE,
        valid_file=VALID_FILE,
        training_args=GANTrainingArgs(
            gen_alpha_0=0.0001,
            gen_alpha_final=0.00001,
            crit_alpha_0=0.0001,
            crit_alpha_final=0.00001,
            crit_iter=5,
            max_steps=5000,
            beta1=0.5,
            beta2=0.9,
            c_lambda=10.0,
        ),
        summary_args=SummaryArgs(summary_freq=250, plt_freq=1000, save_freq=1000, rf_auroc_freq=1000),
        output_dir=tmp_path,
    )

    rf_auroc = gan_trainer.train(compile_modules=True)

    assert 0.499 <= rf_auroc <= 0.9, f"RF AUROC out of expected range: {rf_auroc}"

    gan_trainer.gan.save(checkpoint_path)
    setup_logger("trained_gan_fixture").info(f"Saved trained GAN checkpoint to {checkpoint_path}")
    return checkpoint_path


class TestGANTrainer:
    @pytest.mark.dependency()
    def test_short(self, compile: bool, gan: "GAN", tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        from training import GANTrainer
        from training.dicts import GANTrainingArgs, SummaryArgs

        gan_trainer = GANTrainer(
            gan=gan,
            train_file=TRAIN_FILE,
            valid_file=VALID_FILE,
            training_args=GANTrainingArgs(
                gen_alpha_0=0.0002,
                gen_alpha_final=0.00002,
                crit_alpha_0=0.0002,
                crit_alpha_final=0.00002,
                crit_iter=2,
                max_steps=8,
                beta1=0.5,
                beta2=0.999,
                c_lambda=10.0,
            ),
            summary_args=SummaryArgs(summary_freq=1, plt_freq=2, save_freq=3, rf_auroc_freq=5),
            output_dir=tmp_path,
        )

        rf_auroc = gan_trainer.train(compile_modules=compile)

        for record in caplog.records:
            assert record.levelno < logging.ERROR

        assert 0.499 <= rf_auroc <= 1.001, f"RF AUROC out of expected range: {rf_auroc}"

    @pytest.mark.long
    @pytest.mark.gpu
    def test_long(self, trained_gan: Path, caplog: pytest.LogCaptureFixture) -> None:
        assert trained_gan.exists(), f"Trained GAN checkpoint not found at {trained_gan}"

        for record in caplog.records:
            assert record.levelno < logging.ERROR, f"Error log found: {record.message}"


class TestConditionalCatGANTrainer:
    @pytest.mark.dependency()
    def test_short(
        self, compile: bool, conditional_cat_gan: "ConditionalCatGAN", tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from training import ConditionalCatGANTrainer
        from training.dicts import GANTrainingArgs, SummaryArgs

        gan_trainer = ConditionalCatGANTrainer(
            gan=conditional_cat_gan,
            train_file=TRAIN_FILE,
            valid_file=VALID_FILE,
            training_args=GANTrainingArgs(
                gen_alpha_0=0.0002,
                gen_alpha_final=0.00002,
                crit_alpha_0=0.0002,
                crit_alpha_final=0.00002,
                crit_iter=2,
                max_steps=8,
                beta1=0.5,
                beta2=0.999,
                c_lambda=10.0,
            ),
            summary_args=SummaryArgs(summary_freq=1, plt_freq=2, save_freq=3, rf_auroc_freq=5),
            output_dir=tmp_path,
        )

        rf_auroc = gan_trainer.train(compile_modules=compile)

        for record in caplog.records:
            assert record.levelno < logging.ERROR

        assert 0.499 <= rf_auroc <= 1.001, f"RF AUROC out of expected range: {rf_auroc}"

    @pytest.mark.long
    @pytest.mark.gpu
    def test_long(
        self, conditional_cat_gan_complex: "ConditionalCatGAN", tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from training import ConditionalCatGANTrainer
        from training.dicts import GANTrainingArgs, SummaryArgs

        gan_trainer = ConditionalCatGANTrainer(
            gan=conditional_cat_gan_complex,
            train_file=TRAIN_FILE,
            valid_file=VALID_FILE,
            training_args=GANTrainingArgs(
                gen_alpha_0=0.0001,
                gen_alpha_final=0.00001,
                crit_alpha_0=0.0001,
                crit_alpha_final=0.00001,
                crit_iter=5,
                max_steps=5000,
                beta1=0.5,
                beta2=0.9,
                c_lambda=10.0,
            ),
            summary_args=SummaryArgs(summary_freq=250, plt_freq=1000, save_freq=1000, rf_auroc_freq=1000),
            output_dir=tmp_path,
        )

        rf_auroc = gan_trainer.train(compile_modules=True)

        for record in caplog.records:
            assert record.levelno < logging.ERROR

        assert 0.499 <= rf_auroc <= 0.9, f"RF AUROC out of expected range: {rf_auroc}"
        
class TestConditionalProjGANTrainer:
    @pytest.mark.dependency()
    def test_short(
        self, compile: bool, conditional_proj_gan: "ConditionalProjGAN", tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from training import ConditionalProjGANTrainer
        from training.dicts import GANTrainingArgs, SummaryArgs

        gan_trainer = ConditionalProjGANTrainer(
            gan=conditional_proj_gan,
            train_file=TRAIN_FILE,
            valid_file=VALID_FILE,
            training_args=GANTrainingArgs(
                gen_alpha_0=0.0002,
                gen_alpha_final=0.00002,
                crit_alpha_0=0.0002,
                crit_alpha_final=0.00002,
                crit_iter=2,
                max_steps=8,
                beta1=0.5,
                beta2=0.999,
                c_lambda=10.0,
            ),
            summary_args=SummaryArgs(summary_freq=1, plt_freq=2, save_freq=3, rf_auroc_freq=5),
            output_dir=tmp_path,
        )

        rf_auroc = gan_trainer.train(compile_modules=compile)

        for record in caplog.records:
            assert record.levelno < logging.ERROR

        assert 0.499 <= rf_auroc <= 1.001, f"RF AUROC out of expected range: {rf_auroc}"

    @pytest.mark.long
    @pytest.mark.gpu
    def test_long(
        self, conditional_proj_gan_complex: "ConditionalProjGAN", tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from training import ConditionalProjGANTrainer
        from training.dicts import GANTrainingArgs, SummaryArgs

        gan_trainer = ConditionalProjGANTrainer(
            gan=conditional_proj_gan_complex,
            train_file=TRAIN_FILE,
            valid_file=VALID_FILE,
            training_args=GANTrainingArgs(
                gen_alpha_0=0.0001,
                gen_alpha_final=0.00001,
                crit_alpha_0=0.0001,
                crit_alpha_final=0.00001,
                crit_iter=5,
                max_steps=5000,
                beta1=0.5,
                beta2=0.9,
                c_lambda=10.0,
            ),
            summary_args=SummaryArgs(summary_freq=250, plt_freq=1000, save_freq=1000, rf_auroc_freq=1000),
            output_dir=tmp_path,
        )

        rf_auroc = gan_trainer.train(compile_modules=True)

        for record in caplog.records:
            assert record.levelno < logging.ERROR

        assert 0.499 <= rf_auroc <= 0.9, f"RF AUROC out of expected range: {rf_auroc}"


class TestCausalGANTrainer:
    @pytest.mark.dependency()
    def test_short(
        self,
        compile: bool,
        causalgan: "CausalGAN",
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from training.causal_gan import CausalGANTrainer
        from training.dicts import CausalGANTrainingArgs, SummaryArgs

        gan_trainer = CausalGANTrainer(
            gan=causalgan,
            train_file=TRAIN_FILE,
            valid_file=VALID_FILE,
            training_args=CausalGANTrainingArgs(
                gen_alpha_0=0.001,
                gen_alpha_final=0.0001,
                crit_alpha_0=0.001,
                crit_alpha_final=0.001,
                crit_iter=2,
                labeler_alpha=0.0001,
                antilabeler_alpha=0.0001,
                labeler_training_interval=2,
                max_steps=8,
                beta1=0.5,
                beta2=0.9,
                c_lambda=10.0,
            ),
            summary_args=SummaryArgs(summary_freq=1, plt_freq=2, save_freq=3, rf_auroc_freq=5),
            output_dir=tmp_path,
        )

        rf_auroc = gan_trainer.train(compile_modules=compile)

        for record in caplog.records:
            assert record.levelno < logging.ERROR

        assert 0.499 <= rf_auroc <= 1.001, f"RF AUROC out of expected range: {rf_auroc}"

    @pytest.mark.long
    @pytest.mark.gpu
    def test_long(self, trained_gan: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        from gans.causal_gan import CausalGAN
        from training.causal_gan import CausalGANTrainer
        from training.dicts import CausalGANTrainingArgs, SummaryArgs

        with open(CAUSAL_GRAPH_FILE, "rb") as f:
            import pickle

            causal_graph = pickle.load(f)

        gan = CausalGAN(
            genes_no=991,
            batch_size=1024,
            latent_dim=128,
            noise_per_gene=1,
            depth_per_gene=3,
            width_per_gene=2,
            cc_latent_dim=128,
            cc_layers=[256, 512, 1024],
            cc_pretrained_checkpoint=trained_gan,
            crit_layers=[1024, 512, 256],
            causal_graph=causal_graph,
            labeler_layers=[2000, 2000],
            device="cuda",
        )

        gan_trainer = CausalGANTrainer(
            gan=gan,
            train_file=TRAIN_FILE,
            valid_file=VALID_FILE,
            training_args=CausalGANTrainingArgs(
                gen_alpha_0=0.001,
                gen_alpha_final=0.0001,
                crit_alpha_0=0.001,
                crit_alpha_final=0.001,
                crit_iter=5,
                labeler_alpha=0.0001,
                antilabeler_alpha=0.0001,
                labeler_training_interval=2,
                max_steps=1000,
                beta1=0.5,
                beta2=0.999,
                c_lambda=10.0,
            ),
            summary_args=SummaryArgs(summary_freq=100, plt_freq=250, save_freq=300, rf_auroc_freq=500),
            output_dir=tmp_path,
        )

        rf_auroc = gan_trainer.train(compile_modules=True)

        for record in caplog.records:
            assert record.levelno < logging.ERROR, f"Error log found: {record.message}"

        assert 0.499 <= rf_auroc <= 0.9, f"RF AUROC out of expected range: {rf_auroc}"
