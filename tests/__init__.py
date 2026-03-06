import os

os.environ["GROUNDGAN_LOGLEVEL"] = os.environ.get("GROUNDGAN_LOGLEVEL", "WARNING")

import pytest

import loggers  # pyright: ignore[reportUnusedImport] # noqa: F401


def ignore_warning(action: str, message: str, category: str, module: str) -> pytest.MarkDecorator:
    return pytest.mark.filterwarnings(rf"{action}:{message}:{category}:{module}")

pytestmark = [
    ignore_warning("ignore", r".*pkg_resources is deprecated as an API.*", "", "louvain"),
    ignore_warning(
        "ignore", r".*This package has been superseded by the `leidenalg` package and will no longer be maintained.*", "DeprecationWarning", "scanpy"
    ),
    ignore_warning("ignore", r".*rich is experimental/alpha.*", "tqdm.TqdmExperimentalWarning", ""),
    ignore_warning("ignore", r".*The argument 'device' of Tensor\.pin_memory\(\) is deprecated.*", "DeprecationWarning", "torch"),
    ignore_warning("ignore", r".*The argument 'device' of Tensor\.is_pinned\(\) is deprecated.*", "DeprecationWarning", "torch"),
    ignore_warning(
        "ignore",
        r"Converting a tensor to a Python boolean might cause the trace to be incorrect\. We can't record the data flow of Python values, so this value will be treated as a constant in the future\. This means that the trace might not generalize to other inputs\!",
        "torch.jit._trace.TracerWarning",
        "torch_sparse",
    ),
]