# UMOSLS-DL

**Uncertainty modelling of static terrestrial laser scanning using deep learning**

UMOSLS-DL is a research codebase for estimating point-wise range residuals in
static terrestrial laser scanning (TLS). The proposed Regression PointNet
(RePN) combines multi-scale local geometric features with physically derived
scanner features to predict measurement uncertainty.

The project was developed at the Geodetic Institute, Leibniz University
Hannover, using measurements from a Z+F IMAGER 5016.

## Highlights

- RePN: a PointNet-inspired multi-scale regression network.
- Joint use of local XYZ geometry and scanner-derived features.
- Five-fold cross-validation with checkpointing and early stopping.
- Comparison against a fully connected baseline and XGBoost.
- Calibration of TLS range residuals at point level.

## Reported results

The original experiment used 2,534,160 measurements from a controlled
laboratory setup.

| Metric | Reported outcome |
| --- | ---: |
| Mean residual before calibration | 0.387 mm |
| Mean residual after calibration | 0.009 mm |
| Standard-deviation reduction | 49% |

These values are results reported by the original study. Reproducing them
requires the research dataset and the precomputed point-neighbourhood files,
which are not distributed in this repository.

## Repository layout

```text
UMOSLS-DL/
├── configs/                 Example experiment configuration
├── docs/                    Data and reproducibility documentation
├── notebooks/legacy/       Original exploratory training notebook
├── src/umosls_dl/           Reusable model, data, and training code
├── tests/                   Lightweight unit tests
├── CITATION.cff             Citation metadata
└── pyproject.toml           Package and dependency metadata
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone git@github.com:TingdeLiu/UMOSLS-DL.git
cd UMOSLS-DL

python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[notebook]"
```

PyTorch installation varies by operating system and CUDA version. If the
default package is not suitable for your GPU, install the appropriate PyTorch
build first and then install this project.

## Data preparation

The original pipeline expects:

1. A PLY file containing XYZ coordinates, scan identifiers, physical features,
   and the target range residual.
2. Precomputed `Group_<scan_id>.npy` files containing local XYZ
   neighbourhoods aligned with the filtered PLY records.

See [docs/DATA.md](docs/DATA.md) for the schema and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the current
reproducibility status. Do not commit raw measurements, generated
neighbourhoods, model checkpoints, or prediction files.

## Python API

```python
import torch

from umosls_dl.model import RePN

model = RePN(
    samples_per_scale=(16, 32, 128),
    mlp_channels=((32, 32, 64), (64, 64, 128), (64, 96, 128)),
    physical_feature_count=4,
)

xyz_neighbourhoods = torch.randn(8, 128, 3)
physical_features = torch.randn(8, 4)
predicted_residuals = model(xyz_neighbourhoods, physical_features)
print(predicted_residuals.shape)  # torch.Size([8, 1])
```

The reusable package removes notebook-global state and validates tensor shapes.
The original notebook is retained as a historical experiment record in
`notebooks/legacy/`.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

## Citation

If this work supports your research, please cite:

> Liu, T. (2023). *Uncertainty modelling of static laser scanning using deep
> learning*. Studienarbeit, Leibniz University Hannover.

Machine-readable metadata is provided in [CITATION.cff](CITATION.cff).

## Acknowledgements

- Supervision: Jan Hartmann and PD Dr.-Ing. Hamza Alkhatib
- Institute: Geodetic Institute, Leibniz University Hannover
- Dataset and laboratory support: HiTec Lab, IKG

## Contact and license

For questions or collaboration, contact
[Tingde Liu](mailto:tingde.liu.luh@gmail.com).

No software license has been declared yet. Until a license is added, the source
code remains protected by copyright and should not be redistributed or reused
without permission from the author.
