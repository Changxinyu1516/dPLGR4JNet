"""Verify the two-period train-only selection and post-training evaluation flow."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dplgr4jnet.config import DEFAULT_CONFIG, validate_config
from dplgr4jnet.trainer import run_experiment


def main() -> None:
    config = deepcopy(DEFAULT_CONFIG)
    config["data"].update(
        data_dir="data",
        station_name="01013500",
        file=None,
        train_period=["1980-01-01", "1980-12-31"],
        valid_period=["1981-01-01", "1981-12-31"],
        test_period=None,
        sequence_length=40,
        warmup_length=5,
        window_stride=100,
    )
    config["model"]["name"] = "dPLGR4J"
    config["training"].update(epochs=1, batch_size=4, device="cpu")
    config["evaluation"]["save_excel"] = False
    with TemporaryDirectory() as directory:
        config["experiment"]["output_dir"] = directory
        validate_config(config)
        summary = run_experiment(config)
        history = pd.read_csv(Path(directory) / "history.csv")
        metric_summary = pd.read_csv(Path(directory) / "metrics_summary.csv")
        assert summary["valid_used_during_training"] is False
        assert {"train_NSE", "train_RMSE", "train_KGE"}.issubset(history.columns)
        assert not any("valid" in column.lower() for column in history.columns)
        assert metric_summary["split"].tolist() == ["train", "valid"]
        print("two-period train-only selection and post-training validation=ok")


if __name__ == "__main__":
    main()
