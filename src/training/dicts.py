import warnings
from typing import TypedDict, TypeVar

import numpy as np


class SummaryArgs(TypedDict):
    summary_freq: int
    plt_freq: int
    save_freq: int
    rf_auroc_freq: int

class GANTrainingArgs(TypedDict):
    gen_alpha_0: float
    gen_alpha_final: float
    crit_alpha_0: float
    crit_alpha_final: float
    crit_iter: int
    max_steps: int
    beta1: float
    beta2: float
    c_lambda: float

class CausalGANTrainingArgs(GANTrainingArgs):
    labeler_alpha: float
    antilabeler_alpha: float
    labeler_training_interval: int

__GANGenLosses = TypedDict(
    "__GANGenLosses",
    {
        "Generator Loss": float,
        "Generator Total Loss": float,
    },
)

__GANCritLosses = TypedDict(
    "__GANCritLosses",
    {
        "Critic Fake Loss": float,
        "Critic Real Loss": float,
        "Critic Gradient Penalty Loss": float,
        "Critic Total Loss": float,
    },
)

__GANLosses = TypedDict(
    "__GANLosses",
    {
        "Total Loss": float,
    },
)

__CausalGANGenLosses = TypedDict(
    "__CausalGANGenLosses",
    {
        "Generator Labeler Loss": float,
        "Generator Antilabeler Loss": float,
    },
)

CausalGANLabelerLosses = TypedDict(
    "CausalGANLabelerLosses",
    {
        "Labeler Fake Loss": float,
        "Labeler Real Loss": float,
        "Antilabeler Loss": float,
    },
)


class GANGenLosses(__GANGenLosses):
    pass

class GANCritLosses(__GANCritLosses):
    pass

class GANLosses(GANGenLosses, GANCritLosses, __GANLosses):
    pass

class CausalGANGenLosses(GANGenLosses, __CausalGANGenLosses):
    pass

class CausalGANCritLosses(GANCritLosses):
    pass

class CausalGANLosses(CausalGANGenLosses, CausalGANCritLosses, CausalGANLabelerLosses, __GANLosses):
    pass

Losses = TypeVar("Losses", GANGenLosses, GANCritLosses, GANLosses, CausalGANGenLosses, CausalGANCritLosses, CausalGANLosses)

class LossList(list[Losses]):
    def avg(self, last_n: int | None = None) -> Losses:
        if not self:
            raise ValueError("Cannot average an empty list of losses.")
        last_n = last_n or len(self)
        avg_loss = {}
        for key in self[0]:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r"Mean of empty slice", category=RuntimeWarning)
                avg_loss[key] = np.nanmean([loss[key] for loss in self[-last_n:]]).item()
        return self[0].__class__(**avg_loss)
