#!/usr/bin/python
import os
import shutil
from pathlib import Path

from custom_parser import get_argparser, get_configparser
from loggers import setup_logger

# Setup logger
logger = setup_logger("main")


def main():
    """
    Main script to process the data and/or start the training or
    generate cells from an existing model.
    """
    argparser = get_argparser()
    args = argparser.parse_args()

    cfg_parser = get_configparser()
    cfg_parser.read(args.config)

    # copy the config file to the output dir
    output_dir = cfg_parser.get("EXPERIMENT", "output directory")
    os.makedirs(output_dir, exist_ok=True)

    try:
        shutil.copy(args.config, output_dir)
    except shutil.SameFileError:
        pass

    if args.preprocess:
        from preprocessing import preprocess

        preprocess(cfg_parser)

    if args.create_grn:
        from preprocessing import create_GRN

        create_GRN(cfg_parser)

    if args.train:
        logger.info("Initializing training libraries...")
        import torch.distributed as dist

        from factory import get_factory

        fac = get_factory(cfg_parser)

        if cfg_parser.get("EXPERIMENT", "use DDP", fallback="False") == "True":
            try:
                local_rank = int(os.environ["LOCAL_RANK"])
            except KeyError:
                logger.error(
                    "LOCAL_RANK not found in environment variables. Make sure the program is launched with torch.distributed.launch or torchrun."
                )
                exit(1)
            group = dist.init_process_group("nccl")

            fac.parser.set("EXPERIMENT", "device", f"cuda:{local_rank}")
            logger.info("Starting DDP training...")
            try:
                fac.get_trainer()()
            except Exception as e:
                dist.destroy_process_group(group)
                raise e
            dist.destroy_process_group(group)
            logger.info("Finished training.")
            if args.generate or args.evaluate or args.benchmark_grn or args.perturb:
                raise ValueError(
                    "Cannot generate, evaluate, benchmark GRN, or perturb after training with DDP."
                    "Please run these tasks again without DDP."
                )
            exit(0)
        else:
            logger.info("Starting trainer...")
            fac.get_trainer()()
            logger.info("Finished training")

    if args.optimize_hyperparameters:
        logger.info("Initializing training libraries...")
        import torch.distributed as dist

        from factory import get_factory

        fac = get_factory(cfg_parser)

        if cfg_parser.get("EXPERIMENT", "use DDP", fallback="False") == "True":
            try:
                local_rank = int(os.environ["LOCAL_RANK"])
            except KeyError:
                logger.error(
                    "LOCAL_RANK not found in environment variables. Make sure the program is launched with "
                    "torch.distributed.launch or torchrun."
                )
                exit(1)
            group = dist.init_process_group("nccl")

            fac.parser.set("EXPERIMENT", "device", f"cuda:{local_rank}")
            logger.info("Starting optuna hyperparameter optimization using DDP for each trial...")
            fac.run_optuna_study()
            dist.destroy_process_group(group)
            logger.info("Finished training.")
            if args.generate or args.evaluate or args.benchmark_grn or args.perturb:
                raise RuntimeError(
                    "Cannot generate, evaluate, benchmark GRN, or perturb after training with DDP."
                    "Please run these tasks again without DDP."
                )
            exit(0)
        else:
            if "LOCAL_RANK" in os.environ:
                raise RuntimeError(
                    "LOCAL_RANK found in environment variables but DDP is not enabled. Either enable DDP in the "
                    "config or run the program without torchrun."
                )
            logger.info("Starting optuna hyperparameter optimization...")
            fac.run_optuna_study()
            logger.info("Finished training")

    if args.generate:
        import numpy as np
        import scanpy as sc  # type: ignore
        from scipy.sparse import csr_matrix

        from factory import get_factory

        fac = get_factory(cfg_parser)

        num_cells = int(cfg_parser.get("Generation", "number of cells to generate"))
        logger.info(f"Generating {num_cells} cells...")
        simulated_cells = fac.get_gan().generate_cells(
            num_cells,
            checkpoint=Path(cfg_parser.get("EXPERIMENT", "checkpoint")),
        )[0]
        simulated_cells = csr_matrix(simulated_cells)

        simulated_cells = sc.AnnData(simulated_cells)
        simulated_cells.obs_names = np.repeat("fake", simulated_cells.shape[0]).tolist()
        simulated_cells.obs_names_make_unique()

        # Add variable names
        train_var_names = sc.read_h5ad(cfg_parser.get("Data", "train"), backed="r").var_names
        simulated_cells.var_names = train_var_names.tolist()

        # Get generation path if defined, otherwise fallback
        generation_path = cfg_parser.get("Generation", "generation path", fallback="")
        if not generation_path:
            generation_path = cfg_parser.get("EXPERIMENT", "output directory") + "/simulated.h5ad"

        simulated_cells.write(generation_path)
        logger.info(f"Simulated cells saved to {generation_path}")

    if args.evaluate:
        from evaluation import data_quality

        data_quality.evaluate(cfg_parser)

    if args.benchmark_grn:
        from evaluation import grn_inference

        grn_inference.evaluate(cfg_parser)

    if args.perturb:
        from perturbation import perturbation

        perturbation.perturb(cfg_parser)


if __name__ == "__main__":
    try:
        main()
        logger.info("Finished")
    except Exception:
        logger.exception("Exception occurred")
