import logging
import os
from logging import LogRecord
from pathlib import Path

import pytest

from tests.resources.constants import CAUSAL_GRAPH_FILE, TRAIN_FILE, VALID_FILE

from ..conftest import MakeGanCheckpoint


class RaiseOnErrorHandler(logging.Handler):
    def emit(self, record: LogRecord) -> None:
        if record.levelno >= logging.ERROR:
            # You can raise custom exceptions here if desired
            raise RuntimeError(record.getMessage())


# Needs to be module-level to be picklable for multiprocessing spawn
def gan_process(rank: int, world_size: int, compile: bool, tmp_path: Path) -> None:
    from torch.distributed import destroy_process_group, init_process_group

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)

    from loggers import setup_logging

    setup_logging()
    root_logger = logging.getLogger()
    root_logger.addHandler(RaiseOnErrorHandler())

    from gans.gan import GAN
    from training.dicts import GANTrainingArgs, SummaryArgs
    from training.gan import GANTrainer

    init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=rank)

    gan = GAN(
        genes_no=991,
        batch_size=128,
        latent_dim=128,
        gen_layers=[256, 512, 1024],
        crit_layers=[1024, 512, 256],
        device=f"cuda:{rank}",
    )

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

    gan_trainer.train(compile_modules=compile)

    destroy_process_group()


# Needs to be module-level to be picklable for multiprocessing spawn
def causalgan_process(rank: int, world_size: int, compile: bool, tmp_path: Path, cc_gan_checkpoint: Path) -> None:
    from torch.distributed import destroy_process_group, init_process_group

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)

    from loggers import setup_logging

    setup_logging()
    root_logger = logging.getLogger()
    root_logger.addHandler(RaiseOnErrorHandler())

    init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=rank)

    from gans.causal_gan import CausalGAN
    from training.causal_gan import CausalGANTrainer
    from training.dicts import CausalGANTrainingArgs, SummaryArgs

    with open(CAUSAL_GRAPH_FILE, "rb") as f:
        import pickle

        causal_graph = pickle.load(f)

    causal_gan = CausalGAN(
        genes_no=991,
        batch_size=1024,
        latent_dim=128,
        noise_per_gene=1,
        depth_per_gene=3,
        width_per_gene=2,
        cc_latent_dim=128,
        cc_layers=[256, 512, 1024],
        cc_pretrained_checkpoint=cc_gan_checkpoint,
        crit_layers=[1024, 512, 256],
        causal_graph=causal_graph,
        labeler_layers=[2000, 2000],
        device=f"cuda:{rank}",
    )

    gan_trainer = CausalGANTrainer(
        gan=causal_gan,
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

    gan_trainer.train(compile_modules=compile)

    destroy_process_group()


@pytest.mark.gpu(multi=True)
class TestDDPTrainers:
    @pytest.mark.dependency(
        depends=[
            "tests/training/test_trainers.py::TestGANTrainer::test_short[no_compile-cuda]",
            "tests/training/test_trainers.py::TestGANTrainer::test_short[compile-cuda]",
        ],
        scope="session",
    )
    def test_gan(self, compile: bool, tmp_path: Path) -> None:
        from torch.cuda import device_count
        from torch.multiprocessing.spawn import spawn

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        world_size = device_count()
        spawn(gan_process, args=(world_size, compile, tmp_path), nprocs=world_size, join=True)

    @pytest.mark.dependency(
        depends=[
            "tests/training/test_trainers.py::TestCausalGANTrainer::test_short[no_compile-cuda]",
            "tests/training/test_trainers.py::TestCausalGANTrainer::test_short[compile-cuda]",
        ],
        scope="session",
    )
    def test_causalgan(
        self, compile: bool, tmp_path: Path, make_gan_checkpoint: MakeGanCheckpoint, caplog: pytest.LogCaptureFixture
    ) -> None:
        orig_loglevel = os.environ["GROUNDGAN_LOGLEVEL"]
        os.environ["GROUNDGAN_LOGLEVEL"] = "ERROR"
        from gans.gan import GAN

        cc_gan = GAN(
            genes_no=991,
            batch_size=128,
            latent_dim=128,
            gen_layers=[256, 512, 1024],
            crit_layers=[1024, 512, 256],
            device="cpu",
        )
        cc_gan_checkpoint = make_gan_checkpoint(cc_gan)
        os.environ["GROUNDGAN_LOGLEVEL"] = orig_loglevel

        from torch.cuda import device_count
        from torch.multiprocessing.spawn import spawn

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        world_size = device_count()
        # logging_queue = None
        spawn(causalgan_process, args=(world_size, compile, tmp_path, cc_gan_checkpoint), nprocs=world_size, join=True)
