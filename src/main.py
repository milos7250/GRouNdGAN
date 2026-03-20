#!/usr/bin/env python3
from pathlib import Path
from typing import TYPE_CHECKING, overload

import rich_click as click

from custom_parser import click_options, get_configparser
from init_ddp import with_ddp
from loggers import setup_logger

if TYPE_CHECKING:
    from typing import Literal

    from optuna import Trial

# Setup logger
logger = setup_logger("GRouNdGAN CLI")


@overload
def main(config: Path, *, train: "Literal[True]", trial: "Trial") -> float: ...
@overload
def main(
    config: Path,
    *,
    preprocess: bool = False,
    create_grn: bool = False,
    train: bool = False,
    optimize_hyperparameters: bool = False,
    generate: bool = False,
    evaluate: bool = False,
    benchmark_grn: bool = False,
    perturb: bool = False,
) -> None: ...
def main(
    config: Path,
    *,
    preprocess: bool = False,
    create_grn: bool = False,
    train: bool = False,
    optimize_hyperparameters: bool = False,
    generate: bool = False,
    evaluate: bool = False,
    benchmark_grn: bool = False,
    perturb: bool = False,
    trial: "Trial | None" = None,
) -> None | float:
    """Main script to process the data and/or start training or generate cells."""

    cfg_parser = get_configparser()
    cfg_parser.read(config)

    # copy the config file to the output dir
    output_dir = cfg_parser._get_conv("EXPERIMENT", "output directory", Path)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_parser.save_interpolated(output_dir / config.name)

    import randomness

    deterministic = cfg_parser.getboolean("EXPERIMENT", "deterministic mode", fallback=False)
    seed = cfg_parser.get("EXPERIMENT", "random seed", fallback=None)
    seed = int(seed) if seed is not None else None
    randomness.set_seeds(seed)

    if preprocess:
        from preprocessing import preprocess as preprocess_func

        preprocess_func(cfg_parser)

    if create_grn:
        from preprocessing import create_GRN

        create_GRN(cfg_parser)

    if train:
        logger.info("Initializing training libraries...")
        randomness.set_pytorch_seeds(seed, deterministic)

        from factory import get_factory

        fac = get_factory(cfg_parser)

        if cfg_parser.get("EXPERIMENT", "use DDP", fallback="False") == "True":
            with with_ddp(fac):
                result = fac.get_trainer()(trial)
        else:
            result = fac.get_trainer()(trial)

        if trial:
            return result

    if optimize_hyperparameters:
        logger.info("Initializing training libraries...")
        randomness.set_pytorch_seeds(seed, deterministic)

        from hyperparameter_optimization import optuna_trainer

        optuna_trainer(cfg_parser)()

    if generate:
        randomness.set_pytorch_seeds(seed, deterministic)

        from factory import get_factory

        fac = get_factory(cfg_parser)

        # Get generation path if defined, otherwise fallback
        num_cells = int(cfg_parser.get("Generation", "number of cells to generate"))
        generation_path = cfg_parser.getpath(
            "Generation",
            "generation path",
            fallback=cfg_parser.getpath("EXPERIMENT", "output directory") / "simulated.h5ad",
        )

        logger.info(f"Generating {num_cells} cells...")
        fac.get_gan().generate_h5ad(
            num_cells,
            generation_path,
            reference_dataset=cfg_parser.getpath("Data", "train"),
            checkpoint=cfg_parser.getpath("EXPERIMENT", "checkpoint", fallback=None),
        )
        logger.info(f"Simulated cells saved to {generation_path}")

    if evaluate:
        from evaluation import data_quality

        data_quality.evaluate(cfg_parser)

    if benchmark_grn:
        from evaluation import grn_inference

        grn_inference.evaluate(cfg_parser)

    if perturb:
        randomness.set_pytorch_seeds(seed, deterministic)
        from perturbation import perturbation

        perturbation.perturb(cfg_parser)

    logger.info("Finished")


if __name__ == "__main__":
    try:
        # Call Click command and allow exceptions to propagate for logging
        @click.command()
        @click_options
        def _main(
            config: Path,
            preprocess: bool,
            create_grn: bool,
            train: bool,
            optimize_hyperparameters: bool,
            generate: bool,
            evaluate: bool,
            benchmark_grn: bool,
            perturb: bool,
        ):
            if optimize_hyperparameters and any([generate, evaluate, benchmark_grn, perturb]):
                logger.error(
                    "Cannot generate, evaluate, benchmark GRN, or perturb while optimizing hyperparameters. Please run "
                    "these tasks separately after optimization is complete."
                )
                generate, evaluate, benchmark_grn, perturb = False, False, False, False
            main(
                config,
                preprocess=preprocess,
                create_grn=create_grn,
                train=train,
                optimize_hyperparameters=optimize_hyperparameters,
                generate=generate,
                evaluate=evaluate,
                benchmark_grn=benchmark_grn,
                perturb=perturb,
            )

        _main()
    except Exception:
        logger.exception("An error occurred during execution.")
