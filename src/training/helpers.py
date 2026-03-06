from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import ExponentialLR, LinearLR, SequentialLR

if TYPE_CHECKING:
    from torch.optim import Optimizer
    
def set_exponential_lr(
    optimizer: "Optimizer",
    alpha_0: float,
    alpha_final: float,
    max_steps: int,
    warmup_percent: float = 0.05,
) -> ExponentialLR | SequentialLR:
    """
    Sets up exponentially decaying learning rate scheduler to be used
    with the optimizer.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer for which to create an exponential learning rate scheduler.
    alpha_0 : float
        Initial learning rate.
    alpha_final : float
        Final learning rate.
    max_steps : int
        Total number of training steps. When current_step=max_steps, alpha_final
        will be set as the learning rate.
    warmup_percent : float, optional
        Percentage of total steps to use for learning rate warmup (default: 0.0).

    Returns
    -------
    ExponentialLR | SequentialLR
        Learning rate scheduler. Call the step() function on this
        scheduler in the training loop.
    """
    warmup_steps = int(max_steps * warmup_percent)
    exponential_steps = max_steps - warmup_steps

    # Find the decay rate of the exponential learning rate
    decay_rate = (alpha_final / alpha_0) ** (1 / exponential_steps)
    exponential_sched = ExponentialLR(optimizer, gamma=decay_rate)

    if warmup_steps > 0:
        warmup_sched = LinearLR(
            optimizer=optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(optimizer, [warmup_sched, exponential_sched], milestones=[warmup_steps])
    else:
        return exponential_sched

class RunningAverage:
    """
    Class for computing a running average of a metric.
    """

    def __init__(self, ignore_first: int = 0) -> None:
        self.total = 0.0
        self.count = 0
        self.ignore_next = ignore_first

    def update(self, value: float, n: int = 1) -> None:
        """
        Updates the running average with a new value.

        Parameters
        ----------
        value : float
            New value to update the running average with.
        n : int, optional
            Number of samples that the value corresponds to (default: 1).
        """
        if self.ignore_next > 0:
            self.ignore_next -= n
            if self.ignore_next < 0:
                n = -self.ignore_next
                self.ignore_next = 0
            else:
                return
        self.total += value * n
        self.count += n

    @property
    def average(self) -> float:
        """
        Returns the current running average.

        Returns
        -------
        float
            Current running average. Returns nan if no values have been added yet.
        """
        return self.total / self.count if self.count > 0 else float("nan")