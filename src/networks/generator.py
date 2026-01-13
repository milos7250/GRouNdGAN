from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn.modules.activation import ReLU

from layers.cbn import ConditionalBatchNorm
from layers.lsn import LSN

if TYPE_CHECKING:
    from typing import Any

    from torch import Tensor


class Generator(nn.Module):
    def __init__(
        self,
        z_input: int,
        output_cells_dim: int,
        gen_layers: list[int],
        library_size: int | None = None,
    ) -> None:
        """
        Non-conditional Generator's constructor.

        Parameters
        ----------
        z_input : int
            The dimension of the noise tensor.
        output_cells_dim : int
            The dimension of the output cells (number of genes).
        gen_layers : list[int]
            List of integers corresponding to the number of neurons
            at each hidden layer of the generator.
        library_size : int | None
            Total number of counts per generated cell. If None, no LSN layer is added. Default is None.
        """
        super(Generator, self).__init__()

        self.z_input = z_input
        self.output_cells_dim = output_cells_dim
        self.gen_layers = gen_layers
        self.library_size = library_size

        self._create_generator()

    def forward(self, noise: "Tensor", *args: "Any", **kwargs: "Any") -> "Tensor":
        """
        Function for completing a forward pass of the generator.

        Parameters
        ----------
        noise : Tensor
            The noise used as input by the generator.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        Tensor
            The output of the generator (genes of the generated cell).
        """
        return self._generator(noise)

    def _create_generator(self) -> None:
        """Method for creating the Generator's network."""
        layers = []
        input_size = self.z_input
        for output_size in self.gen_layers:
            layers.append(nn.Sequential(*self._create_generator_block(input_size, output_size)))
            input_size = output_size  # update input size for the next layer

        # outermost layer
        layers.append(
            nn.Sequential(
                *self._create_generator_block(
                    input_size, self.output_cells_dim, library_size=self.library_size, final_layer=True
                )
            )
        )

        self._generator = nn.Sequential(*layers)

    @staticmethod
    def _create_generator_block(
        input_dim: int,
        output_dim: int,
        *,
        library_size: int | None = None,
        final_layer: bool | None = False,
        **kwargs: "Any",
    ) -> tuple[nn.Module, ...]:
        """
        Function for creating a sequence of operations corresponding to
        a Generator block; a linear layer, a batchnorm (except in the final block),
        a ReLU, and LSN in the final layer.

        Parameters
        ----------
        input_dim : int
            The block's input dimensions.
        output_dim : int
            The block's output dimensions.
        library_size : int | None, optional
            Total number of counts per generated cell, by default None.
        final_layer : bool | None, optional
            Indicates if the block contains the final layer, by default False.
        **kwargs: "Any"
            Arbitrary keyword arguments.

        Returns
        -------
        tuple[nn.Module, ...]
             Tuple containing the modules.
        """

        linear_layer = nn.Linear(input_dim, output_dim)

        if not final_layer:
            nn.init.xavier_uniform_(linear_layer.weight)
            return (
                linear_layer,
                nn.BatchNorm1d(output_dim),
                ReLU(inplace=True),
            )
        else:
            # * Unable to find variance_scaling_initializer() with FAN_AVG mode
            nn.init.kaiming_normal_(linear_layer.weight, mode="fan_in", nonlinearity="relu")
            torch.nn.init.zeros_(linear_layer.bias)

            # library_size = None
            if library_size is not None:
                return (linear_layer, ReLU(inplace=True), LSN(library_size))
            else:
                return (linear_layer, ReLU(inplace=True))


class ConditionalGenerator(Generator):
    def __init__(
        self,
        z_input: int,
        output_cells_dim: int,
        num_classes: int,
        gen_layers: list[int],
        library_size: int | None = None,
    ) -> None:
        """
        Conditional Generator's constructor.

        Parameters
        ----------
        z_input : int
            The dimension of the noise tensor.
        output_cells_dim : int
            The dimension of the output cells (number of genes).
        num_classes : int
            Number of clusters.
        gen_layers : list[int]
            List of integers corresponding to the number of neurons
            at each hidden layer of the generator.
        library_size : int | None, optional
            Total number of counts per generated cell, by default None.
        """
        self.num_classes = num_classes

        super(ConditionalGenerator, self).__init__(z_input, output_cells_dim, gen_layers, library_size)

    def forward(self, noise: "Tensor", labels: "Tensor | None" = None, *args: "Any", **kwargs: "Any") -> "Tensor":
        """
        Function for completing a forward pass of the generator.

        Parameters
        ----------
        noise : Tensor
            The noise used as input by the generator.
        labels : Tensor
            Tensor containing labels corresponding to cells to generate.
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        Tensor
            The output of the generator (genes of the generated cell).
        """
        y = noise
        for layer in self._generator:
            if isinstance(layer, ConditionalBatchNorm):
                y = layer(y, labels)
            else:
                y = layer(y)
        return y

    def _create_generator(self) -> None:
        """Method for creating the Generator's network."""
        self._generator = nn.ModuleList()
        input_size = self.z_input
        for output_size in self.gen_layers:
            layers = self._create_generator_block(input_size, output_size, num_classes=self.num_classes)
            for layer in layers:
                self._generator.append(layer)
            input_size = output_size  # update input size for the next layer

        # outermost layer
        self._generator.append(
            nn.Sequential(
                *self._create_generator_block(
                    input_size,
                    self.output_cells_dim,
                    library_size=self.library_size,
                    final_layer=True,
                    num_classes=self.num_classes,
                )
            )
        )

    @staticmethod
    def _create_generator_block(
        input_dim: int,
        output_dim: int,
        *,
        num_classes: int,
        library_size: int | None = None,
        final_layer: bool | None = False,
        **kwargs: "Any",
    ) -> tuple[nn.Module, ...]:
        """
        Function for creating a sequence of operations corresponding to
        a Conditional Generator block; a linear layer, a conditional
        batchnorm (except in the final block), a ReLU, and LSN in the final layer.

        Parameters
        ----------
        input_dim : int
            The block's input dimensions.
        output_dim : int
            The block's output dimensions.
        num_classes : int
            Number of clusters.
        library_size : int | None, optional
            Total number of counts per generated cell, by default None.
        final_layer : bool | None, optional
            Indicates if the block contains the final layer, by default False.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        tuple[nn.Module, ...]
             Tuple containing the modules.
        """

        linear_layer = nn.Linear(input_dim, output_dim)

        if not final_layer:
            nn.init.xavier_uniform_(linear_layer.weight)
            return (linear_layer, ConditionalBatchNorm(output_dim, num_classes), ReLU(inplace=True))
        else:
            nn.init.kaiming_normal_(linear_layer.weight, mode="fan_in", nonlinearity="relu")
            torch.nn.init.zeros_(linear_layer.bias)

            if library_size is not None:
                return (linear_layer, ReLU(inplace=True), LSN(library_size))
            else:
                return (linear_layer, ReLU(inplace=True))
