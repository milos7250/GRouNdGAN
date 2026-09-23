"""Subprocess worker for the upstream causal GAN comparison test."""

from __future__ import annotations

import argparse
import importlib
import pickle
import sys
from configparser import ConfigParser
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("export-upstream", "run-local"))
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gradients", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def _config(path: Path, graph_path: Path) -> dict[str, object]:
    parser = ConfigParser(interpolation=None, allow_no_value=True, inline_comment_prefixes=(";",))
    # The project config format indents nested sections for readability.
    parser.read_string("\n".join(line.lstrip() for line in path.read_text().splitlines()))
    with graph_path.open("rb") as file:
        causal_graph = pickle.load(file)
    return {
        "genes_no": parser.getint("Data", "number of genes"),
        "latent_dim": parser.getint("Model", "latent dim"),
        "noise_per_gene": parser.getint("Model", "noise per gene"),
        "depth_per_gene": parser.getint("Model", "depth per gene"),
        "width_per_gene": parser.getint("Model", "width per gene"),
        "cc_layers": [int(value) for value in parser["CC Model"]["generator layers"].split()],
        "cc_latent_dim": parser.getint("CC Model", "latent dim"),
        "library_size": parser.getint("Preprocessing", "library size"),
        "causal_graph": causal_graph,
    }


def _build_controller(config: dict[str, object], device: str):
    from networks.generator import Generator

    return Generator(
        z_input=config["cc_latent_dim"],
        output_cells_dim=config["genes_no"],
        gen_layers=config["cc_layers"],
        library_size=None,
    ).to(device)


def _build_causal_generator(config: dict[str, object], controller, device: str):
    from networks.masked_causal_generator import CausalGenerator

    generator = CausalGenerator(
        z_input=config["latent_dim"],
        noise_per_gene=config["noise_per_gene"],
        depth_per_gene=config["depth_per_gene"],
        width_scale_per_gene=config["width_per_gene"],
        causal_controller=controller,
        causal_graph=config["causal_graph"],
        library_size=config["library_size"],
        device=device,
    ).to(device)
    # The upstream LSN stores its device as a plain attribute, so Module.to() cannot update it.
    if hasattr(generator, "_lsn"):
        generator._lsn.device = device
    return generator


def _run(
    generator,
    controller_noise: torch.Tensor,
    target_noise: torch.Tensor,
    requires_grad: bool = False,
) -> torch.Tensor:
    import torch
    
    generator._generate_noise = lambda batch_size, latent_dim, device: target_noise  # type: ignore[method-assign]
    gan_module = importlib.import_module("gans.gan")
    gan_class = gan_module.GAN
    original_generate_noise = gan_class.__dict__.get("_generate_noise")
    if original_generate_noise is not None:
        gan_class._generate_noise = staticmethod(  # type: ignore[attr-defined]
            lambda batch_size, latent_dim, device: target_noise
        )
    generator.eval()
    try:
        context = torch.enable_grad() if requires_grad else torch.inference_mode()
        with context:
            return generator(controller_noise)
    finally:
        if original_generate_noise is not None:
            gan_class._generate_noise = original_generate_noise


def _compile_generator(generator, compile: bool):
    if not compile:
        return generator

    import torch

    return torch.compile(generator, fullgraph=True, mode="max-autotune-no-cudagraphs")


def _backward(
    generator,
    controller_noise: torch.Tensor,
    target_noise: torch.Tensor,
    loss_weights: torch.Tensor,
) -> dict[str, np.ndarray]:
    import torch

    gradient_generator = getattr(generator, "_orig_mod", generator)
    gradient_generator.zero_grad(set_to_none=True)
    target_noise = target_noise.detach().clone().requires_grad_(True)
    output = _run(generator, controller_noise, target_noise, requires_grad=True)
    (output * loss_weights).sum().backward()

    gradients: dict[str, np.ndarray] = {}
    modules = dict(gradient_generator.named_modules())
    for name, module in modules.items():
        if not hasattr(module, "weights"):
            continue
        dense_gradient = torch.zeros(
            module.out_features,
            module.in_features,
            dtype=module.weights.grad.dtype,
        )
        indices = module.indices.cpu()
        dense_gradient[indices[0], indices[1]] = module.weights.grad.detach().cpu()
        gradients[f"{name}.weight"] = dense_gradient.numpy()
        gradients[f"{name}.bias"] = module.bias.grad.detach().cpu().numpy()

    for name, parameter in gradient_generator.named_parameters():
        module_name = name.rsplit(".", 1)[0]
        module = modules.get(module_name)
        if name.endswith(".weights") or (name.endswith(".bias") and module is not None and hasattr(module, "weights")):
            continue
        if parameter.grad is not None:
            gradients[name] = parameter.grad.detach().cpu().numpy()

    gradients["target_noise"] = target_noise.grad.detach().cpu().numpy()
    return gradients


def _export_upstream(args: argparse.Namespace) -> None:
    import torch
    
    sys.path.insert(0, str(args.upstream / "src"))
    config = _config(args.config, args.graph)
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    torch.manual_seed(1729)
    controller = _build_controller(config, args.device)
    generator = _build_causal_generator(config, controller, args.device)
    controller.double()
    generator.double()
    controller_noise = torch.randn(4, config["cc_latent_dim"], dtype=torch.float64, device=device)
    target_noise = torch.randn(4, generator.num_noises, dtype=torch.float64, device=device)
    loss_weights = torch.randn(4, config["genes_no"], dtype=torch.float64, device=device)
    compiled_generator = _compile_generator(generator, args.compile)
    output = _run(compiled_generator, controller_noise, target_noise)
    torch.save(
        {
            "controller_state_dict": controller.state_dict(),
            "generator_state_dict": generator.state_dict(),
            "controller_noise": controller_noise,
            "target_noise": target_noise,
            "loss_weights": loss_weights,
        },
        args.checkpoint,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    reloaded_controller = _build_controller(config, args.device)
    reloaded_controller.double()
    reloaded_controller.load_state_dict(checkpoint["controller_state_dict"])
    reloaded_generator = _build_causal_generator(config, reloaded_controller, args.device)
    reloaded_generator.double()
    reloaded_generator.load_state_dict(checkpoint["generator_state_dict"])
    reloaded_compiled_generator = _compile_generator(reloaded_generator, args.compile)
    reloaded_output = _run(
        reloaded_compiled_generator, checkpoint["controller_noise"], checkpoint["target_noise"]
    )
    torch.testing.assert_close(output, reloaded_output)
    np.save(args.output, reloaded_output.detach().cpu().numpy())
    np.savez(
        args.gradients,
        **_backward(
            reloaded_compiled_generator,
            checkpoint["controller_noise"],
            checkpoint["target_noise"],
            checkpoint["loss_weights"],
        ),
    )


def _copy_dense_to_sparse(sparse_generator, dense_state: dict[str, torch.Tensor]) -> None:
    sparse_state = sparse_generator.state_dict()
    for name, module in sparse_generator.named_modules():
        if not hasattr(module, "weights"):
            continue
        dense_weight = dense_state[f"{name}.weight"]
        indices = module.indices.cpu()
        sparse_state[f"{name}.weights"] = dense_weight[indices[0], indices[1]]
        sparse_state[f"{name}.bias"] = dense_state[f"{name}.bias"]

    for name, value in dense_state.items():
        if name.endswith((".weight", ".bias")):
            module_name = name.rsplit(".", 1)[0]
            module = dict(sparse_generator.named_modules()).get(module_name)
            if module is not None and hasattr(module, "weights"):
                continue
        if name in sparse_state:
            sparse_state[name] = value
    sparse_generator.load_state_dict(sparse_state)


def _run_local(args: argparse.Namespace) -> None:
    import torch
    
    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
    config = _config(args.config, args.graph)
    torch.set_default_dtype(torch.float64)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    controller = _build_controller(config, args.device)
    controller.double()
    controller.load_state_dict(checkpoint["controller_state_dict"])
    generator = _build_causal_generator(config, controller, args.device)
    generator.double()
    _copy_dense_to_sparse(generator, checkpoint["generator_state_dict"])
    compiled_generator = _compile_generator(generator, args.compile)
    output = _run(compiled_generator, checkpoint["controller_noise"], checkpoint["target_noise"])
    np.save(args.output, output.detach().cpu().numpy())
    np.savez(
        args.gradients,
        **_backward(
            compiled_generator,
            checkpoint["controller_noise"],
            checkpoint["target_noise"],
            checkpoint["loss_weights"],
        ),
    )


if __name__ == "__main__":
    args = _arguments()
    if args.action == "export-upstream":
        _export_upstream(args)
    else:
        _run_local(args)
