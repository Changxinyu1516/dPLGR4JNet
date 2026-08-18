from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {"name": "dplgr4j_sample", "seed": 42, "output_dir": "outputs/dplgr4j_sample"},
    "data": {
        "data_dir": "data",
        "file": None,
        "station_name": "01013500",
        "date_column": "Time",
        "target_column": "runoff",
        "runoff_unit": "m3/s",
        "topo_file": "camels_topo.txt",
        "topo_gauge_column": "gauge_id",
        "area_column": "area_gages2",
        "area_km2": None,
        "train_period": ["1980-01-01", "2006-12-31"],
        "valid_period": ["2007-01-01", "2014-12-31"],
        "test_period": None,
        "feature_columns": ["PRECIP", "PET_Pristley-Taylor(mm)", "srad(W/m2)", "tmax(C)", "tmin(C)", "vp(Pa)"],
        "sequence_length": 365,
        "warmup_length": 30,
        "window_stride": 365,
    },
    "model": {
        "name": "dPLGR4J",
        "hidden_size": 16,
        "dropout": 0.0,
        "param_limit_func": "sigmoid",
        "param_test_way": "mean_time",
        "param_var_index": [0],
        "param_smoothness_weight": 0.01,
        "n_dynamic_features": None,
        "calibration_hidden_dim": [64, 32],
        "calibration_dropout": 0.05,
        "calibration_history_length": 3,
        "enforce_nonnegative": True,
        "zero_init_mlp": True,
        "transform_physical_features": True,
        "pretrained_dpl_path": None,
        "preserve_pretrained_baseline": True,
        "dpl_anchor_weight": 0.0,
        "correction_penalty_weight": 0.0,
    },
    "training": {
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 0.001,
        "dpl_learning_rate": None,
        "mlp_learning_rate": None,
        "weight_decay": 0.0,
        "grad_clip_norm": 5.0,
        "lr_factor": 0.5,
        "lr_patience": 4,
        "early_stopping_patience": 8,
        "num_workers": 0,
        "device": "auto",
    },
    "evaluation": {"save_excel": True},
}


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        merged[key] = merge_dict(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = merge_dict(DEFAULT_CONFIG, yaml.safe_load(file) or {})
    config = _resolve_env_vars(config)
    validate_config(config)
    return config


def save_config(config: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


def validate_config(cfg: dict[str, Any]) -> None:
    data, model, training = cfg["data"], cfg["model"], cfg["training"]
    for key in ("train_period", "valid_period"):
        if not isinstance(data[key], list) or len(data[key]) != 2:
            raise ValueError(f"data.{key} must be a two-element date list")
    if data.get("test_period") is not None and (
        not isinstance(data["test_period"], list) or len(data["test_period"]) != 2
    ):
        raise ValueError("data.test_period must be null or a two-element date list")
    sequence_length, warmup = int(data["sequence_length"]), int(data["warmup_length"])
    if not 1 < sequence_length or not 0 <= warmup < sequence_length:
        raise ValueError("Require sequence_length > 1 and 0 <= warmup_length < sequence_length")
    if len(data["feature_columns"]) < 2:
        raise ValueError("At least precipitation and PET feature columns are required")
    if model["name"].lower() not in {"dplgr4j", "dplgr4jnet", "dplgr4jd", "dplgr4jnetd"}:
        raise ValueError("model.name must be dPLGR4J, dPLGR4JNet, dPLGR4Jd, or dPLGR4JNetd")
    if model["param_limit_func"] not in {"sigmoid", "clamp"}:
        raise ValueError("model.param_limit_func must be sigmoid or clamp")
    if model["param_test_way"] not in {"final", "mean_time", "mean_basin"}:
        raise ValueError("model.param_test_way must be final, mean_time, or mean_basin")
    if model["name"].lower().endswith("d") and model["param_test_way"] == "mean_basin":
        raise ValueError("Dynamic models do not support param_test_way=mean_basin")
    if "net" in model["name"].lower() and not model.get("pretrained_dpl_path"):
        dependency = "dPLGR4Jd" if model["name"].lower().endswith("d") else "dPLGR4J"
        raise ValueError(
            f"{model['name']} requires model.pretrained_dpl_path from a trained {dependency} run"
        )
    if int(training["epochs"]) < 1 or int(training["batch_size"]) < 1:
        raise ValueError("training.epochs and training.batch_size must be positive")


_validate_config = validate_config


def _resolve_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env_vars(item) for item in value]
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1].strip(), value)
    return value
