# GRouNdGAN Docker Image

GRouNdGAN simulates single-cell RNA-seq data with gene regulatory network
(GRN)-guided causal generative adversarial networks. This image provides a
ready-to-run GRouNdGAN environment with PyTorch, CUDA, and the project
dependencies installed.

- **Docker Hub:** [milos7250/groundgan](https://hub.docker.com/r/milos7250/groundgan)
- **Documentation:** [GRouNdGAN documentation](https://emad-combine-lab.github.io/GRouNdGAN/)
- **Source code:** [github.com/milos7250/GRouNdGAN](https://github.com/milos7250/GRouNdGAN)

## Image Details

- PyTorch 2.14.0
- CUDA 13.2 with cuDNN 9
- Python 3.12
- GRouNdGAN CLI entrypoint: `/groundgan/src/main.py`

The image does not include datasets, configuration files, model checkpoints,
or generated outputs. Mount these files from the host when running a
container.

## Requirements

Install Docker. For GPU execution, also install the NVIDIA Container Toolkit
and verify that Docker can access the host GPU.

## Pull And Verify

Pull the latest image:

```bash
docker pull milos7250/groundgan:latest
```

The container displays the CLI help when no arguments are supplied:

```bash
docker run --rm milos7250/groundgan:latest
```

## Run GRouNdGAN

GRouNdGAN requires a configuration file. The simplest approach is to run from
a project directory containing `configs/`, `data/`, and `results/`:

```bash
docker run --rm --gpus all \
  -v "$PWD:/data" \
  milos7250/groundgan:latest \
  --config /data/configs/causal_gan.cfg \
  --preprocess --create-grn --train --generate --evaluate
```

The `--gpus all` option enables CUDA. Omit it for CPU execution, and set the
configuration's `device` to `cpu` when running training on a machine without a
GPU:

```bash
docker run --rm \
  -v "$PWD:/data" \
  milos7250/groundgan:latest \
  --config /data/configs/gan.cfg --generate
```

The CLI flags can be run separately or combined when the configuration file
contains the required settings:

```bash
docker run --rm --gpus all -v "$PWD:/data" \
  milos7250/groundgan:latest \
  --config /data/configs/causal_gan.cfg --preprocess

docker run --rm --gpus all -v "$PWD:/data" \
  milos7250/groundgan:latest \
  --config /data/configs/causal_gan.cfg --train

docker run --rm --gpus all -v "$PWD:/data" \
  milos7250/groundgan:latest \
  --config /data/configs/causal_gan.cfg --generate --evaluate
```

Available operations include `--preprocess`, `--create-grn`, `--train`,
`--optimize-hyperparameters`, `--generate`, `--evaluate`, `--benchmark-grn`,
and `--perturb`. Every invocation requires `--config PATH`.

## Interactive Shell

The image has a GRouNdGAN entrypoint. Override it when an interactive shell is
needed:

```bash
docker run --rm -it --gpus all \
  --entrypoint /bin/bash \
  -v "$PWD:/data" \
  milos7250/groundgan:latest
```

## Build Locally

Clone the repository and build the `groundgan` image from the repository root:

```bash
git clone https://github.com/milos7250/GRouNdGAN.git
cd GRouNdGAN
./docker/build-docker.sh
```

Run the locally built image by replacing the Docker Hub image name with
`groundgan`:

```bash
docker run --rm --gpus all \
  -v "$PWD:/data" \
  groundgan --config /data/configs/causal_gan.cfg --help
```

## Data And Configuration

The sample configuration files are in the repository's `configs/` directory:

- `gan.cfg` for a non-conditional GAN
- `conditional_gan.cfg` for conditional GANs
- `causal_gan.cfg` for GRouNdGAN

The demo datasets are not bundled in the image. Download them as described in
the [documentation](https://emad-combine-lab.github.io/GRouNdGAN/tutorial.html)
and place them in the paths referenced by your configuration. Preprocessing
must be completed before training, and causal GAN training additionally
requires the configured causal graph and controller inputs.

## Citation

If you use GRouNdGAN, please cite:

```bibtex
@article{zinati2024groundgan,
  title={GRouNdGAN: GRN-guided simulation of single-cell RNA-seq data using causal generative adversarial networks},
  author={Zinati, Yazdan and Takiddeen, Abdulrahman and Emad, Amin},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={1--18},
  year={2024},
  publisher={Nature Publishing Group}
}
```

## License

GRouNdGAN is distributed under the [GNU Affero General Public License v3](https://www.gnu.org/licenses/agpl-3.0.en.html).
