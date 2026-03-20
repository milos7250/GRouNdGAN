import os
import re
from copy import deepcopy
from typing import TYPE_CHECKING

from optuna import create_study
from optuna.pruners import HyperbandPruner
from optuna.trial import TrialState
from optunahub import load_module
from torch._dynamo import reset as dynamo_reset
from torch.cuda import empty_cache as empty_cuda_cache

from loggers import setup_logger
from main import main
from randomness import random_seed

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any, TypeVar

    from optuna.samplers import BaseSampler
    from optuna.study import Study
    from optuna.trial import FrozenTrial, Trial

    from .custom_parser import MyConfigParser

    _T = TypeVar("_T", int, float, str)

logger = setup_logger("optuna")


def suggest_from_tuple(
    str_tuple: str, type_: "type[_T]", suggest_fun: "Callable[..., _T]", var_name: str, **kwargs: "Any"
) -> "_T":
    """
    Used with optuna trials. Suggest a value from a tuple string representation using the provided suggest function
    (e.g., trial.suggest_int or trial.suggest_float). The _T needs to match the type of the suggest function.
    If both bounds are equal, the bound value is returned and the suggest function is not called.

    Parameters
    ----------
    str_tuple
        String representation of a tuple, e.g. "(0.1 1.0)".
    type_
        The type of the values inside the tuple.
    suggest_fun
        Function to suggest a value from the given range. The type of the suggested value needs to match _T.
    var_name
        Name of the variable to suggest. Used in the suggest function.
    **kwargs
        Additional keyword arguments to pass to the suggest function.

    Returns
    -------
    _T
        Suggested value of the specified type, e.g., 0.5.
    """
    bounds = str_tuple.strip("()").split()
    if len(bounds) != 2:
        raise ValueError(f"Invalid tuple string: {str_tuple}")
    bounds = (type_(bounds[0]), type_(bounds[1]))
    if bounds[0] == bounds[1]:
        return bounds[0]
    elif issubclass(type_, (float, int)) and float(bounds[0]) > float(bounds[1]):
        bounds = (bounds[1], bounds[0])
    return suggest_fun(var_name, bounds[0], bounds[1], **kwargs)


def suggest_list_from_tuples(
    str_list: str, type_: "type[_T]", suggest_fun: "Callable[..., _T]", var_name: str, **kwargs: "Any"
) -> "list[_T]":
    """
    Used with optuna trials. Suggests list of values from a string representation of tuples using the provided suggest
    function (e.g., trial.suggest_int or trial.suggest_float). The _T needs to match the type of the suggest
    function.

    Parameters
    ----------
    str_list
        String representation of a list of tuples, e.g. "(0.1 1.0) (0.2 1.1)". Note: No enclosing brackets.
    type_
        The type of the values inside the tuple.
    suggest_fun
        Function to suggest a value from the given range. The type of the suggested value needs to match _T.
    var_name
        Name of the variable to suggest. Used in the suggest function.
    **kwargs
        Additional keyword arguments to pass to the suggest function.

    Returns
    -------
    list[_T]
        Suggested list of values of the specified type, e.g., [0.5, 1.0].
    """
    tuples = re.findall(r"\((.+? .+?)\)", str_list)
    return [suggest_from_tuple(tup, type_, suggest_fun, f"{var_name}_{i}", **kwargs) for i, tup in enumerate(tuples)]


def max_trial_callback(max_trials: int) -> "Callable[[Study, FrozenTrial], None]":
    """
    Optuna callback to stop the study after reaching the maximum number of completed trials.

    Parameters
    ----------
    max_trials
        Maximum number of completed trials before stopping the study.

    Returns
    -------
    Callable
        Callback function to be used with Optuna.
    """

    def max_trial_callback(study: "Study", trial: "FrozenTrial") -> None:
        n_complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        if n_complete >= max_trials:
            logger.info("Optuna study reached required number of completed trials, stopping optimization.")
            study.stop()

    return max_trial_callback


def manual_off_switch_callback(control_file_path: "Path") -> "Callable[[Study, FrozenTrial], None]":
    """
    Optuna callback to stop the study if first line of the control file reads as 'stop'.

    Parameters
    ----------
    control_file_path
        Path to the control file that acts as an off switch.

    Returns
    -------
    Callable
        Callback function to be used with Optuna.
    """

    if not control_file_path.exists():
        with open(control_file_path, "w") as f:
            f.write("# stop\n# To stop the Optuna hyperparameter optimization study, uncomment the first line.")

    def manual_off_switch_callback(study: "Study", trial: "FrozenTrial") -> None:
        if control_file_path.exists():
            with open(control_file_path, "r") as f:
                first_line = f.readline().strip()
            if first_line == "stop":
                setup_logger("optuna").info(
                    f"Manual off switch file detected at {control_file_path}, stopping Optuna optimization."
                )
                study.stop()

    return manual_off_switch_callback


def resolve_hyperparameters(cfg_parser: "MyConfigParser", trial: "Trial"):
    def is_log(*, key: str, value: None = None) -> bool:
        return "learning rate" in key.lower()

    def get_step(*, key: str, value: None = None, type_: type) -> int | None:
        if any(sub in key.lower() for sub in ["layer", "dim"]):
            return 16
        return 1 if type_ == int else None

    def float_or_int(
        *, key: None = None, value: str, trial: "Trial"
    ) -> "tuple[type[float], Callable[..., float]] | tuple[type[int], Callable[..., int]]":
        return (float, trial.suggest_float) if any(c in value for c in ".eE") else (int, trial.suggest_int)

    resolved_parser = deepcopy(cfg_parser)
    for section in cfg_parser.sections():
        if section.startswith("HO "):
            for key, value in cfg_parser.items(section):
                if value.startswith("(") and value.endswith(")") and key not in cfg_parser.defaults():
                    if value.count("(") == 1:
                        # Single tuple, suggest a single value
                        type_, suggest_func = float_or_int(value=value, trial=trial)
                        resolved_value = suggest_from_tuple(
                            value,
                            type_,
                            suggest_func,
                            key,
                            log=is_log(key=key),
                            step=get_step(key=key, type_=type_),
                        )
                        resolved_parser.set(section[3:], key, str(resolved_value))
                    else:
                        # List of tuples, suggest a list of values
                        type_, suggest_func = float_or_int(value=value, trial=trial)
                        resolved_value = suggest_list_from_tuples(
                            value,
                            type_,
                            suggest_func,
                            key,
                            log=is_log(key=key),
                            step=get_step(key=key, type_=type_),
                        )
                        resolved_parser.set(section[3:], key, " ".join([str(v) for v in resolved_value]))

    resolved_parser.set(
        "EXPERIMENT",
        "output directory",
        str(cfg_parser.getpath("EXPERIMENT", "output directory") / f"{trial.number}"),
    )
    resolved_parser_path = cfg_parser.getpath("EXPERIMENT", "output directory") / f"{trial.number}" / "config.cfg"
    resolved_parser.save_interpolated(resolved_parser_path)
    return resolved_parser_path


def optuna_trainer(cfg_parser: "MyConfigParser") -> "Callable[[], None]":
    def objective(trial: "Trial") -> float:
        if worker_id_env := cfg_parser.get(
            "Hyperparameter Optimization", "worker id environment variable", fallback=None
        ):
            trial.set_user_attr("worker_id", os.getenv(worker_id_env, "0"))
        dynamo_reset()  # Reset Dynamo cache between trials to avoid reaching cache limits
        empty_cuda_cache()

        resolved_parser_path = resolve_hyperparameters(cfg_parser, trial=trial)
        return main(resolved_parser_path, train=True, trial=trial)

    AutoSampler: BaseSampler = load_module(package="samplers/auto_sampler").AutoSampler
    storage = cfg_parser.get("Hyperparameter Optimization", "storage", fallback=None)
    study = create_study(
        direction="minimize",
        storage=storage if storage else f"sqlite:///{cfg_parser.get('EXPERIMENT', 'output directory')}/optuna_study.db",
        study_name=cfg_parser.get("Hyperparameter Optimization", "study name", fallback="optuna_study"),
        sampler=AutoSampler(seed=random_seed),  # type: ignore
        # The max_resource should be equal to the maximum number of steps used in training. The min_resource refers
        # to minimum number of steps after which pruning can start. Here, we set it to maximum steps divided by 10
        # to allow pruning after 10% of training is done. The reduction_factor is set to 2 to create 5 brackets as
        # recommended in Optuna docs.
        # (https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html)
        pruner=HyperbandPruner(
            min_resource=cfg_parser.getint("Training", "maximum steps") // 100,
            max_resource=cfg_parser.getint("Training", "maximum steps"),
            reduction_factor=3,
        ),
        load_if_exists=True,
    )

    if len(study.get_trials(states=[TrialState.COMPLETE])) >= cfg_parser.getint(
        "Hyperparameter Optimization", "number of trials"
    ):
        logger.info("Optuna study already has required number of completed trials, skipping optimization.")
        return lambda: None

    return lambda: study.optimize(
        objective,
        gc_after_trial=True,
        callbacks=[
            max_trial_callback(max_trials=cfg_parser.getint("Hyperparameter Optimization", "number of trials")),
            manual_off_switch_callback(
                control_file_path=cfg_parser.getpath("EXPERIMENT", "output directory") / "optuna_stop.txt"
            ),
        ],
    )
