import pickle
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

# from loggers import setup_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from optuna.trial import Trial

    from custom_parser import MyConfigParser
    from gans import GAN, CausalGAN, ConditionalCatGAN, ConditionalProjGAN
    from training.dicts import CausalGANTrainingArgs, GANTrainingArgs, SummaryArgs

_T = TypeVar("_T", int, float, str)
"""
Type variable for :py:func:`parse_list` function. Allowed types are :py:class:`int`, :py:class:`float`, and :py:class:`str`.
"""


def parse_list(str_list: str, _type: "type[_T]") -> "list[_T]":
    """
    Parse a string representation of a list into a list of specified type.

    Parameters
    ----------
    str_list
        String representation of a list (e.g., "0.1 0.2 0.3"). Note: No enclosing brackets.
    _type
        The type of the values inside the tuple.

    Returns
    -------
    list[_T]
        List of values of the specified type (e.g., [0.1, 0.2, 0.3]).
    """
    return list(map(_type, str.split(str_list)))


class IGANFactory(ABC):
    """
    Factory that represents a GAN.
    This factory does not keep of created references.
    """

    def __init__(self, parser: "MyConfigParser") -> None:
        """
        Initialize the factory.

        Parameters
        ----------
        parser
            Parser for config file containing GAN model and training params.
        """
        self.parser = parser

    def get_training_args(self) -> "GANTrainingArgs":
        """
        Get GAN training arguments from the config parser.
        """
        from training.dicts import GANTrainingArgs

        return GANTrainingArgs(
            gen_alpha_0=self.parser.getfloat("Learning Rate", "generator initial"),
            gen_alpha_final=self.parser.getfloat("Learning Rate", "generator final"),
            crit_alpha_0=self.parser.getfloat("Learning Rate", "critic initial"),
            crit_alpha_final=self.parser.getfloat("Learning Rate", "critic final"),
            crit_iter=self.parser.getint("Training", "critic iterations"),
            max_steps=self.parser.getint("Training", "maximum steps"),
            beta1=self.parser.getfloat("Optimizer", "beta1"),
            beta2=self.parser.getfloat("Optimizer", "beta2"),
            c_lambda=self.parser.getfloat("Model", "lambda"),
        )

    def get_summary_args(self) -> "SummaryArgs":
        """
        Get summary arguments from the config parser.
        """
        from training.dicts import SummaryArgs

        return SummaryArgs(
            summary_freq=self.parser.getint("Logging", "summary frequency"),
            plt_freq=self.parser.getint("Logging", "plot frequency"),
            save_freq=self.parser.getint("Logging", "save frequency"),
            rf_auroc_freq=self.parser.getint("Logging", "rf auroc frequency"),
        )

    @abstractmethod
    def get_gan(self) -> "GAN":
        """
        Returns a GAN instance

        Returns
        -------
        GAN
            GAN instance.
        """

    @abstractmethod
    def get_trainer(self) -> "Callable[[Trial | None], float]":
        """
        Returns the GAN train function.

        Returns
        -------
        Callable[[], float]
            GAN train() function.
        """


class GANFactory(IGANFactory):
    def get_gan(self) -> "GAN":
        from gans import GAN

        return GAN(
            genes_no=self.parser.getint("Data", "number of genes"),
            batch_size=self.parser.getint("Training", "batch size"),
            latent_dim=self.parser.getint("Model", "latent dim"),
            gen_layers=parse_list(self.parser["Model"]["generator layers"], int),
            crit_layers=parse_list(self.parser["Model"]["critic layers"], int),
            device=self.parser.get("EXPERIMENT", "device", fallback=None),
            library_size=self.parser.getint("Preprocessing", "library size"),
        )

    def get_trainer(self) -> "Callable[[Trial | None], float]":
        from training import GANTrainer

        gan = self.get_gan()

        gan_trainer = GANTrainer(
            gan=gan,
            train_file=self.parser.getpath("Data", "train"),
            valid_file=self.parser.getpath("Data", "validation"),
            training_args=self.get_training_args(),
            summary_args=self.get_summary_args(),
            output_dir=self.parser.getpath("EXPERIMENT", "output directory"),
        )

        return lambda trial: gan_trainer.train(
            checkpoint_path=self.parser.getpath("EXPERIMENT", "checkpoint", fallback=None),
            compile_modules=self.parser.getboolean("EXPERIMENT", "compile modules", fallback=True),
            trial=trial,
        )


class ConditionalCatGANFactory(IGANFactory):
    def get_gan(self) -> "ConditionalCatGAN":
        from gans import ConditionalCatGAN

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

    def get_trainer(self) -> "Callable[[Trial | None], float]":
        from training import ConditionalCatGANTrainer

        gan = self.get_gan()

        gan_trainer = ConditionalCatGANTrainer(
            gan=gan,
            train_file=self.parser.getpath("Data", "train"),
            valid_file=self.parser.getpath("Data", "validation"),
            training_args=self.get_training_args(),
            summary_args=self.get_summary_args(),
            output_dir=self.parser.getpath("EXPERIMENT", "output directory"),
        )

        return lambda trial: gan_trainer.train(
            checkpoint_path=self.parser.getpath("EXPERIMENT", "checkpoint", fallback=None),
            compile_modules=self.parser.getboolean("EXPERIMENT", "compile modules", fallback=True),
            trial=trial,
        )


class ConditionalProjGANFactory(IGANFactory):
    def get_gan(self) -> "ConditionalProjGAN":
        from gans import ConditionalProjGAN

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

    def get_trainer(self) -> "Callable[[Trial | None], float]":
        from training import ConditionalProjGANTrainer

        gan = self.get_gan()

        gan_trainer = ConditionalProjGANTrainer(
            gan=gan,
            train_file=self.parser.getpath("Data", "train"),
            valid_file=self.parser.getpath("Data", "validation"),
            training_args=self.get_training_args(),
            summary_args=self.get_summary_args(),
            output_dir=self.parser.getpath("EXPERIMENT", "output directory"),
        )

        return lambda trial: gan_trainer.train(
            checkpoint_path=self.parser.getpath("EXPERIMENT", "checkpoint", fallback=None),
            compile_modules=self.parser.getboolean("EXPERIMENT", "compile modules", fallback=True),
            trial=trial,
        )


class CausalGANFactory(IGANFactory):
    def get_gan(self) -> "CausalGAN":
        from gans import CausalGAN

        with open(self.parser.get("Data", "causal graph"), "rb") as fp:
            causal_graph = pickle.load(fp)

        cc_pretrained_checkpoint = self.parser.getpath("CC Model", "checkpoint", fallback=None)
        cc_pretrained_checkpoint = (
            cc_pretrained_checkpoint
            if cc_pretrained_checkpoint
            else self.parser.getpath("EXPERIMENT", "output directory")
            / f"CC/checkpoints/step_{self.parser.getint('CC Training', 'maximum steps', fallback=0)}.pth"
        )

        return CausalGAN(
            genes_no=self.parser.getint("Data", "number of genes"),
            batch_size=self.parser.getint("Training", "batch size"),
            latent_dim=self.parser.getint("Model", "latent dim"),
            noise_per_gene=self.parser.getint("Model", "noise per gene"),
            depth_per_gene=self.parser.getint("Model", "depth per gene"),
            width_per_gene=self.parser.getint("Model", "width per gene"),
            cc_latent_dim=self.parser.getint("CC Model", "latent dim"),
            cc_layers=parse_list(self.parser["CC Model"]["generator layers"], int),
            cc_pretrained_checkpoint=cc_pretrained_checkpoint,
            crit_layers=parse_list(self.parser["Model"]["critic layers"], int),
            causal_graph=causal_graph,
            labeler_layers=parse_list(self.parser["Model"]["labeler layers"], int),
            device=self.parser.get("EXPERIMENT", "device", fallback=None),
            library_size=self.parser.getint("Preprocessing", "library size"),
        )

    def get_training_args(self) -> "CausalGANTrainingArgs":
        from training.dicts import CausalGANTrainingArgs

        base_args = super().get_training_args()
        causal_gan_args = {
            "labeler_alpha": self.parser.getfloat("Learning Rate", "labeler"),
            "antilabeler_alpha": self.parser.getfloat("Learning Rate", "antilabeler"),
            "labeler_training_interval": self.parser.getint("Training", "labeler and antilabeler training intervals"),
        }
        return CausalGANTrainingArgs(base_args | causal_gan_args)  # pyright: ignore[reportArgumentType]

    def get_trainer(self) -> "Callable[[Trial | None], float]":
        def get_cc() -> "GAN":
            from gans import GAN

            return GAN(
                genes_no=self.parser.getint("Data", "number of genes"),
                batch_size=self.parser.getint("CC Training", "batch size"),
                latent_dim=self.parser.getint("CC Model", "latent dim"),
                gen_layers=parse_list(self.parser["CC Model"]["generator layers"], int),
                crit_layers=parse_list(self.parser["CC Model"]["critic layers"], int),
                device=self.parser.get("EXPERIMENT", "device", fallback=None),
                library_size=self.parser.getint("Preprocessing", "library size"),
            )

        def get_cc_training_args() -> "GANTrainingArgs":
            from training.dicts import GANTrainingArgs

            return GANTrainingArgs(
                gen_alpha_0=self.parser.getfloat("CC Learning Rate", "generator initial"),
                gen_alpha_final=self.parser.getfloat("CC Learning Rate", "generator final"),
                crit_alpha_0=self.parser.getfloat("CC Learning Rate", "critic initial"),
                crit_alpha_final=self.parser.getfloat("CC Learning Rate", "critic final"),
                crit_iter=self.parser.getint("CC Training", "critic iterations"),
                max_steps=self.parser.getint("CC Training", "maximum steps"),
                beta1=self.parser.getfloat("CC Optimizer", "beta1"),
                beta2=self.parser.getfloat("CC Optimizer", "beta2"),
                c_lambda=self.parser.getfloat("CC Model", "lambda"),
            )

        def get_cc_summary_args() -> "SummaryArgs":
            from training.dicts import SummaryArgs

            return SummaryArgs(
                summary_freq=self.parser.getint("CC Logging", "summary frequency"),
                plt_freq=self.parser.getint("CC Logging", "plot frequency"),
                save_freq=self.parser.getint("CC Logging", "save frequency"),
                rf_auroc_freq=self.parser.getint("CC Logging", "rf auroc frequency"),
            )

        def get_cc_trainer() -> "Callable[[Trial | None], float]":
            from training import GANTrainer

            cc = get_cc()

            cc_trainer = GANTrainer(
                gan=cc,
                train_file=self.parser.getpath("Data", "train"),
                valid_file=self.parser.getpath("Data", "validation"),
                training_args=get_cc_training_args(),
                summary_args=get_cc_summary_args(),
                output_dir=self.parser.getpath("EXPERIMENT", "output directory") / "CC",
            )

            return lambda trial: cc_trainer.train(
                checkpoint_path=None,  # CC training does not use checkpoints
                compile_modules=self.parser.getboolean("EXPERIMENT", "compile modules", fallback=True),
                trial=trial,
            )

        def _get_trainer() -> "Callable[[Trial | None], float]":
            from training import CausalGANTrainer

            gan = self.get_gan()

            gan_trainer = CausalGANTrainer(
                gan=gan,
                train_file=self.parser.getpath("Data", "train"),
                valid_file=self.parser.getpath("Data", "validation"),
                training_args=self.get_training_args(),
                summary_args=self.get_summary_args(),
                output_dir=self.parser.getpath("EXPERIMENT", "output directory"),
            )

            return lambda trial: gan_trainer.train(
                checkpoint_path=self.parser.getpath("EXPERIMENT", "checkpoint", fallback=None),
                compile_modules=self.parser.getboolean("EXPERIMENT", "compile modules", fallback=True),
                trial=trial,
            )

        if self.parser.get("CC Model", "checkpoint", fallback=None) is not None:
            return _get_trainer()
        else:
            from torch.cuda import empty_cache as empty_cuda_cache

            return lambda trial: (
                get_cc_trainer()(None),
                empty_cuda_cache(),
                _get_trainer()(trial),
            )[2]


def get_factory(cfg: "MyConfigParser") -> IGANFactory:
    """
    Return the factory for the GAN type based on 'model' key in the parser.

    Parameters
    ----------
    cfg
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
    raise ValueError(f"model '{model}' type is invalid, expected one of {list(factories.keys())}")
