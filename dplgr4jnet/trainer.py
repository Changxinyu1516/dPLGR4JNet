from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dplgr4jnet.data import build_dataset, build_full_sequence_tensors, load_data_bundle
from dplgr4jnet.losses import NSELoss
from dplgr4jnet.metrics import summarize_metrics
from dplgr4jnet.model import PARAMETER_NAMES, build_model
from dplgr4jnet.postprocess import save_history, save_json, save_prediction_artifacts


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device_cfg: str) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_cfg == "auto" else torch.device(device_cfg)


def create_model(cfg: dict[str, Any], feature_count: int, warmup_length: int) -> nn.Module:
    model_cfg = cfg["model"]
    name = str(model_cfg["name"])
    kwargs: dict[str, Any] = {
        "n_input_features": feature_count,
        "n_output_features": 4,
        "n_hidden_states": int(model_cfg["hidden_size"]),
        "warmup_length": warmup_length,
        "param_limit_func": model_cfg["param_limit_func"],
        "param_test_way": model_cfg["param_test_way"],
        "dropout": float(model_cfg["dropout"]),
    }
    if name.lower().endswith("d"):
        kwargs.update(param_var_index=model_cfg.get("param_var_index"), param_smoothness_weight=float(model_cfg.get("param_smoothness_weight", 0.0)))
    if "net" in name.lower():
        kwargs.update(
            n_dynamic_features=model_cfg.get("n_dynamic_features"),
            calibration_hidden_dim=model_cfg["calibration_hidden_dim"],
            calibration_dropout=float(model_cfg["calibration_dropout"]),
            calibration_history_length=int(model_cfg["calibration_history_length"]),
            enforce_nonnegative=bool(model_cfg["enforce_nonnegative"]),
            zero_init_mlp=bool(model_cfg["zero_init_mlp"]),
            transform_physical_features=bool(model_cfg["transform_physical_features"]),
            pretrained_dpl_path=model_cfg.get("pretrained_dpl_path"),
            preserve_pretrained_baseline=bool(model_cfg["preserve_pretrained_baseline"]),
            dpl_anchor_weight=float(model_cfg["dpl_anchor_weight"]),
            correction_penalty_weight=float(model_cfg["correction_penalty_weight"]),
        )
    return build_model(name, **kwargs)


def run_experiment(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg["experiment"]["seed"]))
    bundle = load_data_bundle(cfg)
    save_json(output_dir / "feature_scaler.json", bundle.scaler.to_dict())
    save_json(output_dir / "station_meta.json", {
        "station_name": bundle.meta.station_name,
        "source_file": bundle.meta.source_file,
        "area_km2": bundle.meta.area_km2,
        "area_source": bundle.meta.area_source,
        "runoff_input_unit": bundle.meta.runoff_input_unit,
        "model_internal_unit": "mm/day",
    })
    train_loader = DataLoader(build_dataset(bundle.train_frame, bundle), batch_size=int(cfg["training"]["batch_size"]), shuffle=True, num_workers=int(cfg["training"]["num_workers"]))
    device = select_device(cfg["training"]["device"])
    model = create_model(cfg, len(bundle.feature_columns), bundle.warmup_length).to(device)

    if dry_run:
        x_raw, z_norm, y = move_batch_to_sequence_first(next(iter(train_loader)), device)
        model.eval()
        with torch.no_grad():
            prediction, extras = model(x_raw, z_norm, return_extras=True)
        return {
            "model": cfg["model"]["name"], "device": str(device),
            "x_raw_shape": list(x_raw.shape), "z_norm_shape": list(z_norm.shape),
            "target_shape": list(y[bundle.warmup_length:].shape), "prediction_shape": list(prediction.shape),
            "parameter_shape": list(extras["parameters_physical"].shape),
        }

    learning_rate = float(cfg["training"]["learning_rate"])
    if hasattr(model, "optimizer_parameter_groups"):
        groups = model.optimizer_parameter_groups(learning_rate, cfg["training"].get("dpl_learning_rate"), cfg["training"].get("mlp_learning_rate"))
        optimizer = torch.optim.Adam(groups, weight_decay=float(cfg["training"]["weight_decay"]))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=float(cfg["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=float(cfg["training"]["lr_factor"]), patience=int(cfg["training"]["lr_patience"]))
    criterion = NSELoss()
    best_loss, bad_epochs, best_epoch = float("inf"), 0, 0
    history: list[dict[str, Any]] = []
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            bundle.warmup_length,
            float(cfg["training"]["grad_clip_norm"]),
            True,
        )
        # VALID_PERIOD is intentionally not touched here. Scheduler, early
        # stopping, and checkpoint selection use training information only.
        scheduler.step(train_loss)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_NSE": train_metrics["NSE"],
            "train_RMSE": train_metrics["RMSE"],
            "train_KGE": train_metrics["KGE"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"NSE={train_metrics['NSE']:.6f} "
            f"RMSE={train_metrics['RMSE']:.6f} "
            f"KGE={train_metrics['KGE']:.6f}"
        )
        if np.isfinite(train_loss) and train_loss < best_loss:
            best_loss, best_epoch, bad_epochs = train_loss, epoch, 0
            torch.save(
                {
                    "model_name": cfg["model"]["name"],
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "selection_split": "train",
                    "train_loss": train_loss,
                    "train_metrics": train_metrics,
                    "config": cfg,
                },
                best_model_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["training"]["early_stopping_patience"]):
                break
    save_history(output_dir / "history.csv", history)
    if not best_model_path.exists():
        raise RuntimeError("Training did not produce a finite training loss")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    summaries = []
    evaluation_periods = [("train", bundle.train_frame), ("valid", bundle.valid_frame)]
    if bundle.test_frame is not None:
        evaluation_periods.append(("test", bundle.test_frame))
    for split_name, frame in evaluation_periods:
        summaries.append(evaluate_split(model, frame, bundle, split_name, output_dir, device, bool(cfg["evaluation"]["save_excel"])))
    pd.DataFrame(summaries).to_csv(output_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    return {
        "model": cfg["model"]["name"],
        "device": str(device),
        "best_model": str(best_model_path),
        "output_dir": str(output_dir),
        "best_epoch": best_epoch,
        "best_train_loss": best_loss,
        "valid_used_during_training": False,
    }


def run_epoch(model: nn.Module, loader: DataLoader, criterion: NSELoss, optimizer: torch.optim.Optimizer | None, device: torch.device, warmup_length: int, grad_clip_norm: float | None, train_mode: bool) -> tuple[float, dict[str, float]]:
    model.train(train_mode)
    losses = []
    observations = []
    simulations = []
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in loader:
            x_raw, z_norm, y = move_batch_to_sequence_first(batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(x_raw, z_norm)
            loss = criterion(prediction, y[warmup_length:])
            if hasattr(model, "regularization_loss"):
                loss = loss + model.regularization_loss()
            if optimizer is not None:
                loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            observations.append(y[warmup_length:].detach().cpu().numpy().reshape(-1))
            simulations.append(prediction.detach().cpu().numpy().reshape(-1))
    metrics = summarize_metrics(np.concatenate(observations), np.concatenate(simulations))
    return float(np.mean(losses)), metrics


def move_batch_to_sequence_first(batch, device: torch.device):
    return tuple(value.to(device).transpose(0, 1) for value in batch)


def _parameter_frame(extras: dict[str, Any], dates: pd.DatetimeIndex, warmup: int) -> pd.DataFrame:
    aligned_dates = dates[warmup:]
    frame = pd.DataFrame({"date": aligned_dates})
    for key, prefix in (("parameters_physical", ""), ("parameters_normalized", "normalized_"), ("parameters_generated_normalized", "generated_normalized_")):
        values = extras.get(key)
        if values is None:
            continue
        array = values.detach().cpu().numpy()
        if array.ndim == 2:
            array = np.repeat(array[0:1], len(aligned_dates), axis=0)
        elif array.ndim == 3:
            array = array[warmup:, 0, :]
        else:
            raise ValueError(f"Unexpected parameter array shape: {array.shape}")
        for index, name in enumerate(PARAMETER_NAMES):
            frame[f"{prefix}{name}"] = array[:, index]
    return frame


def evaluate_split(model: nn.Module, frame: pd.DataFrame, bundle, split_name: str, output_dir: Path, device: torch.device, save_excel: bool) -> dict[str, Any]:
    model.eval()
    x_raw, z_norm, target, dates = build_full_sequence_tensors(frame, bundle)
    with torch.no_grad():
        prediction, extras = model(x_raw.to(device), z_norm.to(device), return_extras=True)
    observed = target[bundle.warmup_length :, 0, 0].numpy()
    simulated = prediction[:, 0, 0].detach().cpu().numpy()
    metrics = summarize_metrics(observed, simulated)
    parameters = _parameter_frame(extras, dates, bundle.warmup_length)
    save_prediction_artifacts(output_dir, split_name, dates[bundle.warmup_length :], observed, simulated, bundle.meta.area_km2, metrics, parameters, save_excel)
    return {"split": split_name, **metrics}
