import os
from contextlib import contextmanager
from typing import TYPE_CHECKING

from loggers import setup_logger

if TYPE_CHECKING:
    from collections.abc import Generator

    from factory import IGANFactory


@contextmanager
def with_ddp(fac: "IGANFactory") -> "Generator[None, None, None]":
    logger = setup_logger("DDP")
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError:
        logger.error(
            "LOCAL_RANK not found in environment variables. Make sure the program is launched with torch.distributed.launch or torchrun."
        )
        raise

    import torch.distributed as dist

    # Initialize the process group for DDP
    dist.init_process_group(backend="nccl")
    old_device = fac.parser.get("EXPERIMENT", "device", fallback=None)
    fac.parser.set("EXPERIMENT", "device", f"cuda:{local_rank}")

    try:
        yield
    finally:
        fac.parser.set("EXPERIMENT", "device", old_device)  # Reset device to original value after training
        dist.destroy_process_group()  # Clean up the process group
