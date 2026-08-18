"""Model-aware experiment entry point for the four dPL-GR4J variants.

The two Net models are second-stage models and must warm-start their dPL part:

* dPLGR4JNet  <- dPLGR4J/best_model.pt
* dPLGR4JNetd <- dPLGR4Jd/best_model.pt

Edit the settings below and run ``python main.py``.  Set
``AUTO_TRAIN_PREREQUISITE=True`` to run the required base model automatically
before a Net model when its compatible checkpoint is missing.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from dplgr4jnet.config import load_config, save_config, validate_config
from dplgr4jnet.trainer import run_experiment


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_NAMES = ("dPLGR4J", "dPLGR4JNet", "dPLGR4Jd", "dPLGR4JNetd")
MODEL_DEPENDENCIES = {
    "dPLGR4JNet": "dPLGR4J",
    "dPLGR4JNetd": "dPLGR4Jd",
}

# ---------------------------------------------------------------------------
# 1. Select exactly one model
# ---------------------------------------------------------------------------
MODEL_NAME = "dPLGR4J"
STATION_NAME = "01411300"
DEVICE = "auto"  # auto, cpu, cuda, cuda:0, ...
SEED = 42
DRY_RUN = False

# If a Net model is selected and its base checkpoint is missing, True trains
# the prerequisite first; False stops with an explicit instruction.
AUTO_TRAIN_PREREQUISITE = False

# Optional external checkpoint overrides. None uses the standard output path:
# outputs/{station}/{base_model}/best_model.pt
STATIC_DPL_CHECKPOINT = None
DYNAMIC_DPL_CHECKPOINT = None

# ---------------------------------------------------------------------------
# 2. Data settings shared by all four models
# ---------------------------------------------------------------------------
TRAIN_PERIOD = ["1980-01-01", "2006-12-31"]
# VALID_PERIOD is never used for gradient updates, learning-rate scheduling,
# early stopping, or best-model selection. It is evaluated only after training.
VALID_PERIOD = ["2007-01-01", "2014-12-31"]
TEST_PERIOD = None  # optional third period; keep None for this two-period workflow
SEQUENCE_LENGTH = 180
WARMUP_LENGTH = 30
WINDOW_STRIDE = 1

# The first two variables must be precipitation and potential evaporation.
FEATURE_COLUMNS = [
    "PRECIP",
    "PET_Pristley-Taylor(mm)",
    "srad(W/m2)",
    "tmax(C)",
    "tmin(C)",
    "vp(Pa)",
]
AREA_COLUMN = "area_gages2"  # alternatively: area_geospa_fabric
AREA_KM2 = None  # positive number overrides data/camels_topo.txt

# ---------------------------------------------------------------------------
# 3. dPLGR4J: static base model parameters
# ---------------------------------------------------------------------------
SHARED_TRAINING_PARAMS = {
    "weight_decay": 0.0,
    "grad_clip_norm": 5.0,
    "lr_factor": 0.5,  # plateau 时学习率乘以该系数
    "lr_patience": 4,  # 训练损失连续多少个 epoch 未改善后降低学习率
    "num_workers": 0,
}

DPLGR4J_PARAMS = {
    "hidden_size": 32,
    "dropout": 0,
    "param_limit_func": "sigmoid",
    "param_test_way": "mean_time",  # final, mean_time, mean_basin
}
DPLGR4J_TRAINING = {
    **SHARED_TRAINING_PARAMS,
    "epochs": 30,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "dpl_learning_rate": None,
    "mlp_learning_rate": None,
    "early_stopping_patience": 5,
}

# ---------------------------------------------------------------------------
# 4. dPLGR4JNet: static base parameters + residual MLP parameters
#    The four base settings MUST match the dPLGR4J checkpoint above.
# ---------------------------------------------------------------------------
DPLGR4JNET_PARAMS = {
    **DPLGR4J_PARAMS,
    "n_dynamic_features": len(FEATURE_COLUMNS),
    "calibration_hidden_dim": [64, 32],
    "calibration_dropout": 0.15,
    "calibration_history_length": 5,
    "enforce_nonnegative": True,
    "zero_init_mlp": True,
    "transform_physical_features": True,
    "preserve_pretrained_baseline": True,
    "dpl_anchor_weight": 0.05,
    "correction_penalty_weight": 0.002,
}
DPLGR4JNET_TRAINING = {
    **SHARED_TRAINING_PARAMS,
    "epochs": 15,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "dpl_learning_rate": 5e-7,
    "mlp_learning_rate": 1e-4,
    "early_stopping_patience": 10,
}

# ---------------------------------------------------------------------------
# 5. dPLGR4Jd: dynamic base model parameters
# ---------------------------------------------------------------------------
DPLGR4JD_PARAMS = {
    "hidden_size": 32,
    "dropout": 0.15,
    "param_limit_func": "sigmoid",
    "param_test_way": "final",  # final or mean_time
    "param_var_index": [0],  # X1=0, X2=1, X3=2; X4 cannot be dynamic
    "param_smoothness_weight": 0.01,
}
DPLGR4JD_TRAINING = {
    **SHARED_TRAINING_PARAMS,
    "epochs": 30,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "dpl_learning_rate": None,
    "mlp_learning_rate": None,
    "early_stopping_patience": 8,
}

# ---------------------------------------------------------------------------
# 6. dPLGR4JNetd: dynamic base parameters + residual MLP parameters
#    The six base settings MUST match the dPLGR4Jd checkpoint above.
# ---------------------------------------------------------------------------
DPLGR4JNETD_PARAMS = {
    **DPLGR4JD_PARAMS,
    "n_dynamic_features": len(FEATURE_COLUMNS),
    "calibration_hidden_dim": [64, 32],
    "calibration_dropout": 0.15,
    "calibration_history_length": 5,
    "enforce_nonnegative": True,
    "zero_init_mlp": True,
    "transform_physical_features": True,
    "preserve_pretrained_baseline": True,
    "dpl_anchor_weight": 0.05,
    "correction_penalty_weight": 0.002,
}
DPLGR4JNETD_TRAINING = {
    **SHARED_TRAINING_PARAMS,
    "epochs": 30,
    "batch_size": 8,
    "learning_rate": 1e-4,
    "dpl_learning_rate": 5e-7,
    "mlp_learning_rate": 1e-4,
    "early_stopping_patience": 8,
}

MODEL_PARAMETERS = {
    "dPLGR4J": DPLGR4J_PARAMS,
    "dPLGR4JNet": DPLGR4JNET_PARAMS,
    "dPLGR4Jd": DPLGR4JD_PARAMS,
    "dPLGR4JNetd": DPLGR4JNETD_PARAMS,
}
MODEL_TRAINING = {
    "dPLGR4J": DPLGR4J_TRAINING,
    "dPLGR4JNet": DPLGR4JNET_TRAINING,
    "dPLGR4Jd": DPLGR4JD_TRAINING,
    "dPLGR4JNetd": DPLGR4JNETD_TRAINING,
}

def model_output_dir(model_name: str) -> Path:
    return PROJECT_DIR / "outputs" / STATION_NAME / model_name

def default_base_checkpoint(net_model_name: str) -> Path:
    base_model = MODEL_DEPENDENCIES[net_model_name]
    override = STATIC_DPL_CHECKPOINT if base_model == "dPLGR4J" else DYNAMIC_DPL_CHECKPOINT
    return Path(override) if override is not None else model_output_dir(base_model) / "best_model.pt"

def build_config(
    model_name: str | None = None,
    pretrained_dpl_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a configuration containing only the selected model's parameters."""
    selected = MODEL_NAME if model_name is None else model_name
    if selected not in MODEL_NAMES:
        raise ValueError(f"MODEL_NAME must be one of {MODEL_NAMES}; got {selected!r}")
    config = load_config(PROJECT_DIR / "configs" / "default.yaml")
    config["experiment"].update(
        name=f"{STATION_NAME}_{selected}",
        seed=SEED,
        output_dir=str(model_output_dir(selected)),
    )
    config["data"].update(
        data_dir=str(PROJECT_DIR / "data"),
        file=None,
        station_name=STATION_NAME,
        train_period=TRAIN_PERIOD,
        valid_period=VALID_PERIOD,
        test_period=TEST_PERIOD,
        feature_columns=FEATURE_COLUMNS,
        sequence_length=SEQUENCE_LENGTH,
        warmup_length=WARMUP_LENGTH,
        window_stride=WINDOW_STRIDE,
        topo_file="camels_topo.txt",
        area_column=AREA_COLUMN,
        area_km2=AREA_KM2,
    )
    config["model"] = {"name": selected, **deepcopy(MODEL_PARAMETERS[selected])}
    if selected in MODEL_DEPENDENCIES:
        checkpoint = default_base_checkpoint(selected) if pretrained_dpl_path is None else Path(pretrained_dpl_path)
        config["model"]["pretrained_dpl_path"] = str(checkpoint.resolve())
    config["training"].update(deepcopy(MODEL_TRAINING[selected]), device=DEVICE)
    validate_config(config)
    return config

def _checkpoint_value(config: dict[str, Any], section: str, key: str):
    return config.get(section, {}).get(key)

def validate_base_checkpoint(net_config: dict[str, Any]) -> Path:
    """Check checkpoint type and all settings required to reproduce its baseline."""
    net_model = net_config["model"]["name"]
    expected_base = MODEL_DEPENDENCIES[net_model]
    path = Path(net_config["model"]["pretrained_dpl_path"])
    if not path.is_file():
        raise FileNotFoundError(
            f"{net_model} requires a trained {expected_base} checkpoint, but it was not found:\n"
            f"  {path}\n"
            f"First set MODEL_NAME = {expected_base!r} and run python main.py, or set "
            "AUTO_TRAIN_PREREQUISITE = True."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    actual_base = checkpoint.get("model_name")
    if actual_base != expected_base:
        raise ValueError(
            f"{net_model} requires {expected_base}, but checkpoint {path} contains {actual_base!r}"
        )
    base_config = checkpoint.get("config")
    if not isinstance(base_config, dict):
        raise ValueError(f"Checkpoint does not contain its resolved training config: {path}")

    common_checks = [
        ("model", "hidden_size"),
        ("model", "dropout"),
        ("model", "param_limit_func"),
        ("model", "param_test_way"),
        ("data", "station_name"),
        ("data", "feature_columns"),
        ("data", "warmup_length"),
        ("data", "sequence_length"),
        ("data", "window_stride"),
        ("data", "train_period"),
    ]
    if expected_base == "dPLGR4Jd":
        common_checks.extend(
            [("model", "param_var_index"), ("model", "param_smoothness_weight")]
        )
    mismatches = []
    for section, key in common_checks:
        baseline_value = _checkpoint_value(base_config, section, key)
        net_value = _checkpoint_value(net_config, section, key)
        if baseline_value != net_value:
            mismatches.append(f"{section}.{key}: checkpoint={baseline_value!r}, net={net_value!r}")
    if mismatches:
        raise ValueError(
            f"{net_model} is incompatible with its {expected_base} checkpoint {path}:\n  "
            + "\n  ".join(mismatches)
        )
    return path


def run_config(config: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "resolved_config.yaml")
    print("Experiment configuration:")
    print(
        json.dumps(
            {
                "model": config["model"],
                "training": config["training"],
                "station": config["data"]["station_name"],
                "output_dir": config["experiment"]["output_dir"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    summary = run_experiment(config, dry_run=dry_run)
    print("Run summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> dict[str, Any]:
    """Run the selected model, respecting the two required training stages."""
    selected = MODEL_NAME
    if selected in MODEL_DEPENDENCIES:
        net_config = build_config(selected)
        try:
            validate_base_checkpoint(net_config)
        except (FileNotFoundError, ValueError) as error:
            if not AUTO_TRAIN_PREREQUISITE:
                raise
            base_name = MODEL_DEPENDENCIES[selected]
            print(f"Prerequisite checkpoint is missing or incompatible: {error}")
            print(f"Training prerequisite {base_name} before {selected}...")
            run_config(build_config(base_name), dry_run=False)
            validate_base_checkpoint(net_config)
        return run_config(net_config, dry_run=DRY_RUN)
    return run_config(build_config(selected), dry_run=DRY_RUN)


if __name__ == "__main__":
    main()
