import os
import pickle
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from configparser import ConfigParser
from pathlib import Path
from typing import TYPE_CHECKING

from optuna import create_study
from optuna.pruners import HyperbandPruner
from optuna.study import Study
from optuna.trial import TrialState
from optunahub import load_module
from torch._dynamo import reset as dynamo_reset
from torch.cuda import empty_cache as empty_cuda_cache

from gans.causal_gan import CausalGAN
from gans.conditional_gan_cat import ConditionalCatGAN
from gans.conditional_gan_proj import ConditionalProjGAN
from gans.gan import GAN
from loggers import setup_logger

if TYPE_CHECKING:
    from typing import Any, TypeVar

    from optuna.samplers import BaseSampler
    from optuna.study import Study
    from optuna.trial import FrozenTrial, Trial

    _T = TypeVar("_T", int, float, str)


def parse_list(str_list: str, _type: "type[_T]") -> "list[_T]":
    """
    Parse a string representation of a list into a list of specified type.

    Parameters
    ----------
    str_list : str
        String representation of a list (e.g., "0.1 0.2 0.3"). Note: No enclosing brackets.
    _type : type[_T]
        The type of the values inside the tuple.

    Returns
    -------
    list[_T]
        List of values of the specified type (e.g., [0.1, 0.2, 0.3]).
    """
    return list(map(_type, str.split(str_list)))


def suggest_from_tuple(
    str_tuple: str, type_: "type[_T]", suggest_fun: "Callable[..., _T]", var_name: str, **kwargs: "Any"
) -> "_T":
    """
    Used with optuna trials. Suggest a value from a tuple string representation using the provided suggest function
    (e.g., trial.suggest_int or trial.suggest_float). The _T needs to match the type of the suggest function.
    If both bounds are equal, the bound value is returned and the suggest function is not called.

    Parameters
    ----------
    str_tuple : str
        String representation of a tuple, e.g. "(0.1 1.0)".
    type_ : type[_T]
        The type of the values inside the tuple.
    suggest_fun : Callable[..., _T]
        Function to suggest a value from the given range. The type of the suggested value needs to match _T.
    var_name : str
        Name of the variable to suggest. Used in the suggest function.
    **kwargs : Any
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
    str_list : str
        String representation of a list of tuples, e.g. "(0.1 1.0) (0.2 1.1)". Note: No enclosing brackets.
    type_ : type[_T]
        The type of the values inside the tuple.
    suggest_fun : Callable
        Function to suggest a value from the given range. The type of the suggested value needs to match _T.
    var_name : str
        Name of the variable to suggest. Used in the suggest function.
    **kwargs : Any
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
    max_trials : int
        Maximum number of completed trials before stopping the study.

    Returns
    -------
    Callable
        Callback function to be used with Optuna.
    """

    def max_trial_callback(study: "Study", trial: "FrozenTrial") -> None:
        n_complete = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        if n_complete >= max_trials:
            setup_logger("optuna").info(
                "Optuna study reached required number of completed trials, stopping optimization."
            )
            study.stop()

    return max_trial_callback


def manual_off_switch_callback(control_file_path: Path) -> "Callable[[Study, FrozenTrial], None]":
    """
    Optuna callback to stop the study if first line of the control file reads as 'stop'.

    Parameters
    ----------
    control_file_path : Path
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


class IGANFactory(ABC):
    """
    Factory that represents a GAN.
    This factory does not keep of created references.
    """

    def __init__(self, parser: ConfigParser) -> None:
        """
        Initialize the factory.

        Parameters
        ----------
        parser : ConfigParser
            Parser for config file containing GAN model and training params.
        """
        self.parser = parser

    @abstractmethod
    def get_gan(self) -> GAN:
        """
        Returns a GAN instance

        Returns
        -------
        GAN
            GAN instance.
        """
        pass

    @abstractmethod
    def get_trainer(self) -> Callable[[], float]:
        """
        Returns the GAN train function.

        Returns
        -------
        Callable[[], float]
            GAN train() function.
        """
        pass

    @abstractmethod
    def run_optuna_study(self) -> "Study":
        """
        Runs an Optuna hyperparameter optimization study for the GAN.

        Returns
        -------
        Study
            The completed Optuna study.
        """
        pass


class GANFactory(IGANFactory):
    def get_gan(self) -> GAN:
        return GAN(
            genes_no=self.parser.getint("Data", "number of genes"),
            batch_size=self.parser.getint("Training", "batch size"),
            latent_dim=self.parser.getint("Model", "latent dim"),
            gen_layers=parse_list(self.parser["Model"]["generator layers"], int),
            crit_layers=parse_list(self.parser["Model"]["critic layers"], int),
            device=self.parser.get("EXPERIMENT", "device", fallback=None),
            library_size=self.parser.getint("Preprocessing", "library size"),
        )

    def get_trainer(self) -> Callable[[], float]:
        gan = self.get_gan()
        return lambda: gan.train(
            train_files=Path(self.parser.get("Data", "train")),
            valid_files=Path(self.parser.get("Data", "validation")),
            critic_iter=self.parser.getint("Training", "critic iterations"),
            max_steps=self.parser.getint("Training", "maximum steps"),
            c_lambda=self.parser.getfloat("Model", "lambda"),
            beta1=self.parser.getfloat("Optimizer", "beta1"),
            beta2=self.parser.getfloat("Optimizer", "beta2"),
            gen_alpha_0=self.parser.getfloat("Learning Rate", "generator initial"),
            gen_alpha_final=self.parser.getfloat("Learning Rate", "generator final"),
            crit_alpha_0=self.parser.getfloat("Learning Rate", "critic initial"),
            crit_alpha_final=self.parser.getfloat("Learning Rate", "critic final"),
            checkpoint=Path(self.parser.get("EXPERIMENT", "checkpoint"))
            if self.parser.get("EXPERIMENT", "checkpoint", fallback=None)
            else None,
            summary_freq=self.parser.getint("Logging", "summary frequency"),
            plt_freq=self.parser.getint("Logging", "plot frequency"),
            save_freq=self.parser.getint("Logging", "save frequency"),
            rf_auroc_freq=self.parser.getint("Logging", "rf auroc frequency", fallback=0),
            output_dir=Path(self.parser.get("EXPERIMENT", "output directory")),
        )

    def run_optuna_study(self) -> "Study":
        def objective(trial: "Trial") -> float:
            if worker_id_env := self.parser.get(
                "Hyperparameter Optimization", "worker id environment variable", fallback=None
            ):
                trial.set_user_attr("worker_id", os.getenv(worker_id_env, "0"))
            dynamo_reset()  # Reset Dynamo cache between trials to avoid reaching cache limits
            empty_cuda_cache()

            # Create GAN with hyperparameters suggested by Optuna
            model = GAN(
                genes_no=self.parser.getint("Data", "number of genes"),
                batch_size=suggest_from_tuple(
                    self.parser["HO Training"]["batch size"], int, trial.suggest_int, "batch_size", step=16
                ),
                latent_dim=suggest_from_tuple(
                    self.parser["HO Model"]["latent dim"], int, trial.suggest_int, "latent_dim", step=16
                ),
                gen_layers=suggest_list_from_tuples(
                    self.parser["HO Model"]["generator layers"], int, trial.suggest_int, "gen_layer", step=16
                ),
                crit_layers=suggest_list_from_tuples(
                    self.parser["HO Model"]["critic layers"], int, trial.suggest_int, "crit_layer", step=16
                ),
                device=self.parser.get("EXPERIMENT", "device", fallback=None),
                library_size=self.parser.getint("Preprocessing", "library size"),
            )

            return model.train(
                train_files=Path(self.parser.get("Data", "train")),
                valid_files=Path(self.parser.get("Data", "validation")),
                critic_iter=suggest_from_tuple(
                    self.parser["HO Training"]["critic iterations"],
                    int,
                    trial.suggest_int,
                    "critic_iter",
                ),
                max_steps=self.parser.getint("HO Training", "maximum steps"),
                c_lambda=suggest_from_tuple(
                    self.parser["HO Model"]["lambda"],
                    float,
                    trial.suggest_float,
                    "c_lambda",
                ),
                beta1=suggest_from_tuple(
                    self.parser["HO Optimizer"]["beta1"],
                    float,
                    trial.suggest_float,
                    "beta1",
                ),
                beta2=suggest_from_tuple(
                    self.parser["HO Optimizer"]["beta2"],
                    float,
                    trial.suggest_float,
                    "beta2",
                ),
                gen_alpha_0=suggest_from_tuple(
                    self.parser["HO Learning Rate"]["generator initial"],
                    float,
                    trial.suggest_float,
                    "gen_alpha_0",
                    log=True,
                ),
                gen_alpha_final=self.parser.getfloat("HO Learning Rate", "generator final"),
                crit_alpha_0=suggest_from_tuple(
                    self.parser["HO Learning Rate"]["critic initial"],
                    float,
                    trial.suggest_float,
                    "crit_alpha_0",
                    log=True,
                ),
                crit_alpha_final=self.parser.getfloat("HO Learning Rate", "critic final"),
                checkpoint=Path(self.parser.get("EXPERIMENT", "checkpoint"))
                if self.parser.get("EXPERIMENT", "checkpoint", fallback=None)
                else None,
                summary_freq=self.parser.getint("Logging", "summary frequency"),
                plt_freq=self.parser.getint("Logging", "plot frequency"),
                save_freq=self.parser.getint("Logging", "save frequency"),
                rf_auroc_freq=self.parser.getint("Logging", "rf auroc frequency", fallback=0),
                output_dir=Path(self.parser.get("EXPERIMENT", "output directory"))
                / f"optuna_gan_trials/{trial.number}",
                trial=trial,
            )

        AutoSampler: "BaseSampler" = load_module(package="samplers/auto_sampler").AutoSampler
        study = create_study(
            direction="minimize",
            storage=self.parser.get(
                "Hyperparameter Optimization",
                "storage",
                fallback=f"sqlite:///{self.parser.get('EXPERIMENT', 'output directory')}/optuna_gan_study.db",
            ),
            study_name=self.parser.get("Hyperparameter Optimization", "study name", fallback="optuna_gan_study"),
            sampler=AutoSampler(),  # type: ignore
            # The max_resource should be equal to the maximum number of steps used in training. The min_resource refers
            # to minimum number of steps after which pruning can start. Here, we set it to maximum steps divided by 10
            # to allow pruning after 10% of training is done. The reduction_factor is set to 2 to create 5 brackets as
            # recommended in Optuna docs.
            # (https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html)
            pruner=HyperbandPruner(
                min_resource=self.parser.getint("HO Training", "maximum steps") // 10,
                max_resource=self.parser.getint("HO Training", "maximum steps"),
                reduction_factor=2,
            ),
            load_if_exists=True,
        )

        if len(study.get_trials(states=[TrialState.COMPLETE])) >= self.parser.getint(
            "Hyperparameter Optimization", "number of trials"
        ):
            setup_logger("optuna").info(
                "Optuna study already has required number of completed trials, skipping optimization."
            )
            return study

        study.optimize(
            objective,
            gc_after_trial=True,
            callbacks=[
                max_trial_callback(max_trials=self.parser.getint("Hyperparameter Optimization", "number of trials"))
            ],
        )

        return study


class ConditionalCatGANFactory(IGANFactory):
    def get_gan(self) -> ConditionalCatGAN:
        return ConditionalCatGAN(
            genes_no=self.parser.getint("Data", "number of genes"),
            batch_size=self.parser.getint("Training", "batch size"),
            latent_dim=self.parser.getint("Model", "latent dim"),
            gen_layers=parse_list(self.parser["Model"]["generator layers"], int),
            crit_layers=parse_list(self.parser["Model"]["critic layers"], int),
            num_classes=self.parser.getint("Data", "number of classes"),
            label_ratios=parse_list(self.parser["Data"]["label ratios"], float),
            device=self.parser.get("EXPERIMENT", "device", fallback=None),
            library_size=self.parser.getint("Preprocessing", "library size"),
        )

    def get_trainer(self) -> Callable[[], float]:
        gan = self.get_gan()
        return lambda: gan.train(
            train_files=Path(self.parser.get("Data", "train")),
            valid_files=Path(self.parser.get("Data", "validation")),
            critic_iter=self.parser.getint("Training", "critic iterations"),
            max_steps=self.parser.getint("Training", "maximum steps"),
            c_lambda=self.parser.getfloat("Model", "lambda"),
            beta1=self.parser.getfloat("Optimizer", "beta1"),
            beta2=self.parser.getfloat("Optimizer", "beta2"),
            gen_alpha_0=self.parser.getfloat("Learning Rate", "generator initial"),
            gen_alpha_final=self.parser.getfloat("Learning Rate", "generator final"),
            crit_alpha_0=self.parser.getfloat("Learning Rate", "critic initial"),
            crit_alpha_final=self.parser.getfloat("Learning Rate", "critic final"),
            checkpoint=Path(self.parser.get("EXPERIMENT", "checkpoint"))
            if self.parser.get("EXPERIMENT", "checkpoint", fallback=None)
            else None,
            summary_freq=self.parser.getint("Logging", "summary frequency"),
            plt_freq=self.parser.getint("Logging", "plot frequency"),
            save_freq=self.parser.getint("Logging", "save frequency"),
            output_dir=Path(self.parser.get("EXPERIMENT", "output directory")),
        )

    def run_optuna_study(self) -> "Study":
        raise NotImplementedError("Optuna study not implemented for ConditionalCatGANFactory.")


class ConditionalProjGANFactory(IGANFactory):
    def get_gan(self) -> ConditionalProjGAN:
        return ConditionalProjGAN(
            genes_no=self.parser.getint("Data", "number of genes"),
            batch_size=self.parser.getint("Training", "batch size"),
            latent_dim=self.parser.getint("Model", "latent dim"),
            gen_layers=parse_list(self.parser["Model"]["generator layers"], int),
            crit_layers=parse_list(self.parser["Model"]["critic layers"], int),
            num_classes=self.parser.getint("Data", "number of classes"),
            label_ratios=parse_list(self.parser["Data"]["label ratios"], float),
            device=self.parser.get("EXPERIMENT", "device", fallback=None),
            library_size=self.parser.getint("Preprocessing", "library size"),
        )

    def get_trainer(self) -> Callable[[], float]:
        gan = self.get_gan()
        return lambda: gan.train(
            train_files=Path(self.parser.get("Data", "train")),
            valid_files=Path(self.parser.get("Data", "validation")),
            critic_iter=self.parser.getint("Training", "critic iterations"),
            max_steps=self.parser.getint("Training", "maximum steps"),
            c_lambda=self.parser.getfloat("Model", "lambda"),
            beta1=self.parser.getfloat("Optimizer", "beta1"),
            beta2=self.parser.getfloat("Optimizer", "beta2"),
            gen_alpha_0=self.parser.getfloat("Learning Rate", "generator initial"),
            gen_alpha_final=self.parser.getfloat("Learning Rate", "generator final"),
            crit_alpha_0=self.parser.getfloat("Learning Rate", "critic initial"),
            crit_alpha_final=self.parser.getfloat("Learning Rate", "critic final"),
            checkpoint=Path(self.parser.get("EXPERIMENT", "checkpoint"))
            if self.parser.get("EXPERIMENT", "checkpoint", fallback=None)
            else None,
            summary_freq=self.parser.getint("Logging", "summary frequency"),
            plt_freq=self.parser.getint("Logging", "plot frequency"),
            save_freq=self.parser.getint("Logging", "save frequency"),
            output_dir=Path(self.parser.get("EXPERIMENT", "output directory")),
        )

    def run_optuna_study(self) -> "Study":
        raise NotImplementedError("Optuna study not implemented for ConditionalProjGANFactory.")


class CausalGANFactory(IGANFactory):
    def get_cc(self) -> GAN:
        return GAN(
            genes_no=self.parser.getint("Data", "number of genes"),
            batch_size=self.parser.getint("CC Training", "batch size"),
            latent_dim=self.parser.getint("CC Model", "latent dim"),
            gen_layers=parse_list(self.parser["CC Model"]["generator layers"], int),
            crit_layers=parse_list(self.parser["CC Model"]["critic layers"], int),
            device=self.parser.get("EXPERIMENT", "device", fallback=None),
            library_size=self.parser.getint("Preprocessing", "library size"),
        )

    def get_gan(self) -> CausalGAN:
        with open(self.parser.get("Data", "causal graph"), "rb") as fp:
            causal_graph = pickle.load(fp)

        return CausalGAN(
            genes_no=self.parser.getint("Data", "number of genes"),
            batch_size=self.parser.getint("Training", "batch size"),
            latent_dim=self.parser.getint("Model", "latent dim"),
            noise_per_gene=self.parser.getint("Model", "noise per gene"),
            depth_per_gene=self.parser.getint("Model", "depth per gene"),
            width_per_gene=self.parser.getint("Model", "width per gene"),
            cc_latent_dim=self.parser.getint("CC Model", "latent dim"),
            cc_layers=parse_list(self.parser["CC Model"]["generator layers"], int),
            cc_pretrained_checkpoint=Path(
                self.parser.get(
                    "CC Model",
                    "checkpoint",
                    fallback=Path(self.parser.get("EXPERIMENT", "output directory"))
                    / f"/CC/checkpoints/step_{self.parser.getint('CC Training', 'maximum steps', fallback=0)}.pth",
                )
            ),
            crit_layers=parse_list(self.parser["Model"]["critic layers"], int),
            causal_graph=causal_graph,
            labeler_layers=parse_list(self.parser["Model"]["labeler layers"], int),
            device=self.parser.get("EXPERIMENT", "device", fallback=None),
            library_size=self.parser.getint("Preprocessing", "library size"),
        )

    def get_trainer(self) -> Callable[[], float]:
        def gan_train() -> float:
            return self.get_gan().train(
                train_files=Path(self.parser.get("Data", "train")),
                valid_files=Path(self.parser.get("Data", "validation")),
                critic_iter=self.parser.getint("Training", "critic iterations"),
                max_steps=self.parser.getint("Training", "maximum steps"),
                c_lambda=self.parser.getfloat("Model", "lambda"),
                beta1=self.parser.getfloat("Optimizer", "beta1"),
                beta2=self.parser.getfloat("Optimizer", "beta2"),
                gen_alpha_0=self.parser.getfloat("Learning Rate", "generator initial"),
                gen_alpha_final=self.parser.getfloat("Learning Rate", "generator final"),
                crit_alpha_0=self.parser.getfloat("Learning Rate", "critic initial"),
                crit_alpha_final=self.parser.getfloat("Learning Rate", "critic final"),
                labeler_alpha=self.parser.getfloat("Learning Rate", "labeler"),
                antilabeler_alpha=self.parser.getfloat("Learning Rate", "antilabeler"),
                labeler_training_interval=self.parser.getint("Training", "labeler and antilabeler training intervals"),
                checkpoint=Path(self.parser.get("EXPERIMENT", "checkpoint"))
                if self.parser.get("EXPERIMENT", "checkpoint", fallback=None)
                else None,
                starting_checkpoint=Path(self.parser.get("EXPERIMENT", "starting checkpoint"))
                if self.parser.get("EXPERIMENT", "starting checkpoint", fallback=None)
                else None,
                summary_freq=self.parser.getint("Logging", "summary frequency"),
                plt_freq=self.parser.getint("Logging", "plot frequency"),
                save_freq=self.parser.getint("Logging", "save frequency"),
                rf_auroc_freq=self.parser.getint("Logging", "rf auroc frequency"),
                output_dir=Path(self.parser.get("EXPERIMENT", "output directory")),
            )

        if self.parser.has_option("CC Model", "checkpoint"):
            return gan_train
        else:
            cc = self.get_cc()

            # the following lambda will train the causal controller for maximum steps
            # specified in the CC Training section of the config file
            # after training the causal controller, the causal GAN will be instantiated
            # with the pretrained causal controller and training will start.
            return lambda: (
                cc.train(
                    train_files=Path(self.parser.get("Data", "train")),
                    valid_files=Path(self.parser.get("Data", "validation")),
                    critic_iter=self.parser.getint("CC Training", "critic iterations"),
                    max_steps=self.parser.getint("CC Training", "maximum steps"),
                    c_lambda=self.parser.getfloat("CC Model", "lambda"),
                    beta1=self.parser.getfloat("CC Optimizer", "beta1"),
                    beta2=self.parser.getfloat("CC Optimizer", "beta2"),
                    gen_alpha_0=self.parser.getfloat("CC Learning Rate", "generator initial"),
                    gen_alpha_final=self.parser.getfloat("CC Learning Rate", "generator final"),
                    crit_alpha_0=self.parser.getfloat("CC Learning Rate", "critic initial"),
                    crit_alpha_final=self.parser.getfloat("CC Learning Rate", "critic final"),
                    checkpoint=Path(
                        self.parser.get(
                            "CC Model",
                            "checkpoint",
                            fallback=Path(self.parser.get("EXPERIMENT", "output directory"))
                            / f"/CC/checkpoints/step_{self.parser.getint('CC Training', 'maximum steps')}.pth",
                        )
                    ),
                    summary_freq=self.parser.getint("CC Logging", "summary frequency"),
                    plt_freq=self.parser.getint("CC Logging", "plot frequency"),
                    save_freq=self.parser.getint("CC Logging", "save frequency"),
                    rf_auroc_freq=self.parser.getint("CC Logging", "rf auroc frequency"),
                    output_dir=Path(self.parser.get("EXPERIMENT", "output directory")) / "CC",
                ),
                gan_train(),
            )[1]

    def run_optuna_study(self) -> "Study":
        def objective(trial: "Trial") -> float:
            if worker_id_env := self.parser.get(
                "Hyperparameter Optimization", "worker id environment variable", fallback=None
            ):
                trial.set_user_attr("worker_id", os.getenv(worker_id_env, "0"))
            dynamo_reset()  # Reset Dynamo cache between trials to avoid reaching cache limits
            empty_cuda_cache()

            # Load causal graph
            with open(self.parser.get("Data", "causal graph"), "rb") as fp:
                causal_graph = pickle.load(fp)

            # Create CausalGAN with hyperparameters suggested by Optuna
            model = CausalGAN(
                genes_no=self.parser.getint("Data", "number of genes"),
                batch_size=suggest_from_tuple(
                    self.parser["HO Training"]["batch size"], int, trial.suggest_int, "batch_size", step=16
                ),
                latent_dim=suggest_from_tuple(
                    self.parser["HO Model"]["latent dim"], int, trial.suggest_int, "latent_dim", step=16
                ),
                noise_per_gene=suggest_from_tuple(
                    self.parser["HO Model"]["noise per gene"],
                    int,
                    trial.suggest_int,
                    "noise_per_gene",
                ),
                depth_per_gene=suggest_from_tuple(
                    self.parser["HO Model"]["depth per gene"],
                    int,
                    trial.suggest_int,
                    "depth_per_gene",
                ),
                width_per_gene=suggest_from_tuple(
                    self.parser["HO Model"]["width per gene"],
                    int,
                    trial.suggest_int,
                    "width_per_gene",
                ),
                cc_latent_dim=self.parser.getint("CC Model", "latent dim"),
                cc_layers=parse_list(self.parser["CC Model"]["generator layers"], int),
                cc_pretrained_checkpoint=Path(self.parser.get("CC Model", "checkpoint")),
                crit_layers=suggest_list_from_tuples(
                    self.parser["HO Model"]["critic layers"], int, trial.suggest_int, "crit_layer", step=16
                ),
                causal_graph=causal_graph,
                labeler_layers=suggest_list_from_tuples(
                    self.parser["HO Model"]["labeler layers"],
                    int,
                    trial.suggest_int,
                    "labeler_layer",
                    step=16,
                ),
                device=self.parser.get("EXPERIMENT", "device", fallback=None),
                library_size=self.parser.getint("Preprocessing", "library size"),
            )

            return model.train(
                train_files=Path(self.parser.get("Data", "train")),
                valid_files=Path(self.parser.get("Data", "validation")),
                critic_iter=suggest_from_tuple(
                    self.parser["HO Training"]["critic iterations"],
                    int,
                    trial.suggest_int,
                    "critic_iter",
                ),
                max_steps=self.parser.getint("HO Training", "maximum steps"),
                c_lambda=suggest_from_tuple(self.parser["HO Model"]["lambda"], float, trial.suggest_float, "c_lambda"),
                beta1=suggest_from_tuple(
                    self.parser["HO Optimizer"]["beta1"],
                    float,
                    trial.suggest_float,
                    "beta1",
                ),
                beta2=suggest_from_tuple(
                    self.parser["HO Optimizer"]["beta2"],
                    float,
                    trial.suggest_float,
                    "beta2",
                ),
                gen_alpha_0=suggest_from_tuple(
                    self.parser["HO Learning Rate"]["generator initial"],
                    float,
                    trial.suggest_float,
                    "gen_alpha_0",
                    log=True,
                ),
                gen_alpha_final=self.parser.getfloat("HO Learning Rate", "generator final"),
                crit_alpha_0=suggest_from_tuple(
                    self.parser["HO Learning Rate"]["critic initial"],
                    float,
                    trial.suggest_float,
                    "crit_alpha_0",
                    log=True,
                ),
                crit_alpha_final=self.parser.getfloat("HO Learning Rate", "critic final"),
                labeler_alpha=suggest_from_tuple(
                    self.parser["HO Learning Rate"]["labeler"],
                    float,
                    trial.suggest_float,
                    "labeler_alpha",
                    log=True,
                ),
                antilabeler_alpha=suggest_from_tuple(
                    self.parser["HO Learning Rate"]["antilabeler"],
                    float,
                    trial.suggest_float,
                    "antilabeler_alpha",
                    log=True,
                ),
                labeler_training_interval=suggest_from_tuple(
                    self.parser["HO Training"]["labeler and antilabeler training intervals"],
                    int,
                    trial.suggest_int,
                    "labeler_training_interval",
                ),
                checkpoint=Path(self.parser.get("EXPERIMENT", "checkpoint"))
                if self.parser.get("EXPERIMENT", "checkpoint", fallback=None)
                else None,
                starting_checkpoint=Path(self.parser.get("EXPERIMENT", "starting checkpoint"))
                if self.parser.get("EXPERIMENT", "starting checkpoint", fallback=None)
                else None,
                summary_freq=self.parser.getint("Logging", "summary frequency"),
                plt_freq=self.parser.getint("Logging", "plot frequency"),
                save_freq=self.parser.getint("Logging", "save frequency"),
                rf_auroc_freq=self.parser.getint("Logging", "rf auroc frequency"),
                output_dir=Path(self.parser.get("EXPERIMENT", "output directory"))
                / f"optuna_causalgan_trials/{trial.number}",
                trial=trial,
            )

        AutoSampler: "BaseSampler" = load_module(package="samplers/auto_sampler").AutoSampler
        study = create_study(
            direction="minimize",
            storage=self.parser.get(
                "Hyperparameter Optimization",
                "storage",
                fallback=f"sqlite:///{self.parser.get('EXPERIMENT', 'output directory')}/optuna_causalgan_study.db",
            ),
            study_name=self.parser.get("Hyperparameter Optimization", "study name", fallback="optuna_causalgan_study"),
            sampler=AutoSampler(),  # type: ignore
            # The max_resource should be equal to the maximum number of steps used in training. The min_resource refers
            # to minimum number of steps after which pruning can start. Here, we set it to maximum steps divided by 100
            # to allow pruning after 1% of training is done. The reduction_factor is set to 3 to create 5 brackets as
            # recommended in Optuna docs.
            # (https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html)
            pruner=HyperbandPruner(
                min_resource=self.parser.getint("HO Training", "maximum steps") // 100,
                max_resource=self.parser.getint("HO Training", "maximum steps"),
                reduction_factor=3,  # chosen to create 5 buckets (as recommended in Optuna docs)
            ),
            load_if_exists=True,
        )

        if len(study.get_trials(states=[TrialState.COMPLETE])) >= self.parser.getint(
            "Hyperparameter Optimization", "number of trials"
        ):
            setup_logger("optuna").info(
                "Optuna study already has required number of completed trials, skipping optimization."
            )
            return study

        study.optimize(
            objective,
            gc_after_trial=True,
            callbacks=[
                max_trial_callback(max_trials=self.parser.getint("Hyperparameter Optimization", "number of trials")),
                manual_off_switch_callback(
                    control_file_path=Path(self.parser.get("EXPERIMENT", "output directory")) / "optuna_stop.txt"
                ),
            ],
        )

        return study


def get_factory(cfg: ConfigParser) -> IGANFactory:
    """
    Return the factory for the GAN type based on 'model' key in the parser.

    Parameters
    ----------
    cfg : ConfigParser
        Parser for config file containing GAN model and training params.

    Returns
    -------
    IGANFactory
        Factory for the specified GAN.

    Raises
    ------
    ValueError
        If the model is unknown or not implemented.
    """
    # read the desired GAN
    model = cfg.get("Model", "type")
    factories: dict[str, IGANFactory] = {
        "GAN": GANFactory(cfg),
        "proj conditional GAN": ConditionalProjGANFactory(cfg),
        "cat conditional GAN": ConditionalCatGANFactory(cfg),
        "causal GAN": CausalGANFactory(cfg),
    }

    if model in factories:
        return factories[model]
    raise ValueError(f"model '{model}' type is invalid")
