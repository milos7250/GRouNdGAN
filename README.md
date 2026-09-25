# <img src="https://github.com/milos7250/GRouNdScale/blob/master/docs/_static/logo.svg" width="250">
 

_**GR**N-guided in silic**o** sim**u**lation of single-cell R**N**A-seq **d**ata using Causal **G**enerative **A**dversarial **N**etworks_

[![Website](https://img.shields.io/website?url=https%3A%2F%2Fmilos7250.github.io%2FGRouNdScale%2F)](https://milos7250.github.io/GRouNdScale/)
[![CI](https://github.com/milos7250/GRouNdScale/actions/workflows/documentation.yaml/badge.svg?branch=master)](https://github.com/milos7250/GRouNdScale/actions)
[![Docker build status](https://img.shields.io/github/actions/workflow/status/milos7250/GRouNdScale/docker-build.yml?logo=docker&label=docker%20build)](https://github.com/milos7250/GRouNdScale/actions/workflows/docker-build.yml)
[![Docker Image Size with architecture (latest by date/latest semver)](https://img.shields.io/docker/image-size/milos7250/groundscale?logo=docker)](https://hub.docker.com/r/milos7250/groundscale)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11068246.svg)](https://doi.org/10.5281/zenodo.11068246) -->

---
This repository holds the updated and improved version of single-cell RNA-seq data simulation software [GRouNdScale](https://github.com/Emad-COMBINE-lab/GRouNdScale). This implementation has multiple improvements over the original, including but not limited to significantly improved speed and memory efficiency and hyperparameter optimisation procedures.

The original implementation of GRouNdScale is described in:
> Zinati, Y., Takiddeen, A. & Emad, A. GRouNdScale: GRN-guided simulation of single-cell RNA-seq data using causal generative adversarial networks. Nat Commun 15, 4055 (2024). https://doi.org/10.1038/s41467-024-48516-6


## Simulated Datasets and Ground Truth GRNs
Simulated dataset and their underlying ground truth GRNs are available for download on [our website](https://milos7250.github.io/GRouNdScale/benchmarking.html#bonemarrow-paul-et-al-2015).


## Tutorials and Documentation
For a detailed tutorial and comprehensive API references, please visit our project's documentation [here](https://milos7250.github.io/GRouNdScale/).

<!--
## BibTex Citation
```
@article{zinati2024groundscale,
  title={GRouNdScale: GRN-guided simulation of single-cell RNA-seq data using causal generative adversarial networks},
  author={Zinati, Yazdan and Takiddeen, Abdulrahman and Emad, Amin},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={1--18},
  year={2024},
  publisher={Nature Publishing Group}
}
```
-->

## License 
Copyright (C) 2026 Milos Micik. 

GRouNdScale is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

GRouNdScale is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with GRouNdScale. If not, see <https://www.gnu.org/licenses/>.
