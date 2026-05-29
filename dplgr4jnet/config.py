from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "dplgr4jnet_sx",
        "seed": 42,
        "output_dir": "outputs/dplgr4jnet_sx",
    },
    "data": {
        "data_dir": "${DPLGR4JNET_DATA_DIR}",
        "station_name": "兴山",
        "basin_name": "兴山以上",
        "runoff_name": "兴山以上",
        "train_period": ["2010-01-01", "2020-12-31"],
        "valid_period": ["2021-01-01", "2022-12-31"],
        "test_period": ["2023-01-01", "2024-12-31"],
        "feature_columns": [
            "precipitation",
            "evapotranspiration",
            "temperature_mean",
            "relative_humidity",
            "solar_radiation",
        ],
        "target_column": "streamflow",
        "sequence_length": 180,
        "warmup_length": 0,
        "window_stride": 1,
    },
    "model": {
        "hidden_size": 16,
        "dropout": 0.2,
        "param_limit_func": "clamp",
        "calibration_hidden_size": 64,
        "calibration_dropout": 0.4,
    },
    "training": {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "grad_clip_norm": 5.0,
        "lr_factor": 0.5,
        "lr_patience": 8,
        "early_stopping_patience": 8,
        "num_workers": 0,
        "device": "auto",
    },
    "evaluation": {
        "save_excel": True,
    },
}


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        user_cfg = yaml.safe_load(file) or {}
    cfg = merge_dict(DEFAULT_CONFIG, user_cfg)
    cfg = _resolve_env_vars(cfg)
    _validate_config(cfg)
    return cfg


def save_config(config: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


def _validate_period(period: list[str], name: str) -> None:
    if not isinstance(period, list) or len(period) != 2:
        raise ValueError(f"{name} must be a two-element date list.")


def _validate_config(cfg: dict[str, Any]) -> None:
    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]

    for key in ("train_period", "valid_period", "test_period"):
        _validate_period(data_cfg[key], key)

    if data_cfg["sequence_length"] <= 1:
        raise ValueError("data.sequence_length must be greater than 1.")

    if data_cfg["warmup_length"] < 0:
        raise ValueError("data.warmup_length must be non-negative.")

    data_dir = str(data_cfg.get("data_dir", "")).strip()
    if not data_dir or data_dir.startswith("${"):
        raise ValueError(
            "data.data_dir is not configured. Set DPLGR4JNET_DATA_DIR, edit configs/default.yaml, "
            "or pass --data-dir."
        )

    if training_cfg["epochs"] < 1:
        raise ValueError("training.epochs must be at least 1.")

    if training_cfg["batch_size"] < 1:
        raise ValueError("training.batch_size must be at least 1.")

    if model_cfg["param_limit_func"] not in {"clamp", "sigmoid"}:
        raise ValueError("model.param_limit_func must be 'clamp' or 'sigmoid'.")

    if model_cfg["calibration_hidden_size"] < 1:
        raise ValueError("model.calibration_hidden_size must be at least 1.")

    if not 0.0 <= float(model_cfg["calibration_dropout"]) < 1.0:
        raise ValueError("model.calibration_dropout must be in [0, 1).")


def _resolve_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1].strip()
        return os.environ.get(env_name, value)
    return value
