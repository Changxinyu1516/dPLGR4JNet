# dPLGR4JNet

`dPLGR4JNet` is a standalone Python implementation of a differentiable GR4J workflow for rainfall-runoff modeling on the `SXdata` dataset.

## Key Features

- Uses an LSTM to generate GR4J parameters.
- Runs differentiable GR4J routing inside PyTorch.
- Applies an MLP calibrator on top of the GR4J simulation.
- Trains directly on the `SXdata` Excel files after local configuration.
- Exports predictions, metrics, and GR4J parameters after evaluation.

## License And Notices

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for license and attribution details.

## Repository Layout

```text
dPLGR4JNet/
  configs/
    default.yaml
  docs/
    PUBLIC_RELEASE_CHECKLIST.md
    software-availability.md
  dplgr4jnet/
    __init__.py
    __main__.py
    cli.py
    config.py
    data.py
    losses.py
    metrics.py
    model.py
    postprocess.py
    trainer.py
  LICENSE
  THIRD_PARTY_NOTICES.md
  CITATION.cff
  pyproject.toml
  requirements.txt
  README.md
```

## Requirements

- Python 3.10+
- `torch`
- `pandas`
- `numpy`
- `openpyxl`
- `PyYAML`

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Data Setup

This repository does **not** redistribute the `SXdata` dataset.

Before running the software, obtain authorized access to the dataset and point the project to your local copy in one of these ways:

1. Set the `DPLGR4JNET_DATA_DIR` environment variable.
2. Edit `data.data_dir` in `configs/default.yaml`.
3. Pass `--data-dir` on the command line.

Example:

```powershell
$env:DPLGR4JNET_DATA_DIR = "D:/path/to/SXdata"
python -m dplgr4jnet --config configs/default.yaml --dry-run
```

Or:

```powershell
python -m dplgr4jnet --config configs/default.yaml --data-dir "D:/path/to/SXdata" --dry-run
```

## Run

Smoke test:

```powershell
python -m dplgr4jnet --config configs/default.yaml --dry-run
```

Training and evaluation:

```powershell
python -m dplgr4jnet --config configs/default.yaml
```

Useful overrides:

```powershell
python -m dplgr4jnet --config configs/default.yaml --data-dir "D:/path/to/SXdata" --output-dir outputs/my_run --epochs 20
```

## Outputs

By default, outputs are written to `outputs/dplgr4jnet_sx`.

Typical artifacts include:

- `best_model.pt`
- `resolved_config.yaml`
- `feature_scaler.json`
- `station_meta.json`
- `history.csv`
- `metrics_summary.csv`
- `train_predictions.csv`
- `valid_predictions.csv`
- `test_predictions.csv`
- `test_parameters.csv`
- `test_metrics.json`
- `test_predictions.xlsx`

## Reproducibility Notes

- The default config is intended as a starting point for the `SXdata` workflow.
- Dataset files are expected to be available locally and are not bundled with this repository.
- If you publish this repository, replace placeholder contact and repository metadata in `docs/software-availability.md`.

## Citation And Software Availability

- Citation metadata for GitHub is provided in `CITATION.cff`.
- A manuscript-ready software availability template is provided in `docs/software-availability.md`.
