from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dplgr4jnet.data import (
    build_dataset,
    build_full_sequence_tensors,
    load_data_bundle,
)
from dplgr4jnet.losses import NSELoss
from dplgr4jnet.metrics import summarize_metrics
from dplgr4jnet.model import DPLGR4JNet
from dplgr4jnet.postprocess import save_history, save_json, save_prediction_artifacts


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def run_experiment(cfg: dict[str, Any], dry_run: bool = False) -> dict[str, Any]:
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(cfg["experiment"]["seed"]))
    bundle = load_data_bundle(cfg)
    save_json(output_dir / "feature_scaler.json", bundle.scaler.to_dict())
    save_json(
        output_dir / "station_meta.json",
        {
            "station_name": bundle.meta.station_name,
            "basin_name": bundle.meta.basin_name,
            "runoff_name": bundle.meta.runoff_name,
            "area_km2": bundle.meta.area_km2,
            "attributes": bundle.meta.attributes,
        },
    )

    train_dataset = build_dataset(bundle.train_frame, bundle)
    valid_dataset = build_dataset(bundle.valid_frame, bundle)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"]["num_workers"]),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["training"]["num_workers"]),
    )

    device = select_device(cfg["training"]["device"])
    model = DPLGR4JNet(
        n_input_features=len(bundle.feature_columns),
        n_hidden_states=int(cfg["model"]["hidden_size"]),
        warmup_length=bundle.warmup_length,
        param_limit_func=cfg["model"]["param_limit_func"],
        dropout=float(cfg["model"]["dropout"]),
        calibration_hidden_size=int(cfg["model"]["calibration_hidden_size"]),
        calibration_dropout=float(cfg["model"]["calibration_dropout"]),
    ).to(device)

    if dry_run:
        batch = next(iter(train_loader))
        x_raw, z_norm, y = move_batch_to_sequence_first(batch, device)
        with torch.no_grad():
            pred, extras = model(x_raw, z_norm, return_extras=True)
        return {
            "device": str(device),
            "x_raw_shape": list(x_raw.shape),
            "z_norm_shape": list(z_norm.shape),
            "y_shape": list(y.shape),
            "pred_shape": list(pred.shape),
            "parameter_shape": list(extras["parameters_physical"].shape),
            "base_streamflow_shape": list(extras["base_streamflow"].shape),
        }

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=float(cfg["training"]["lr_factor"]),
        patience=int(cfg["training"]["lr_patience"]),
    )
    criterion = NSELoss()

    best_val_loss = float("inf")
    bad_epochs = 0
    history: list[dict[str, float]] = []
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            warmup_length=bundle.warmup_length,
            grad_clip_norm=float(cfg["training"]["grad_clip_norm"]),
            train_mode=True,
        )
        valid_loss = run_epoch(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            warmup_length=bundle.warmup_length,
            grad_clip_norm=None,
            train_mode=False,
        )
        scheduler.step(valid_loss)

        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "valid_loss": float(valid_loss),
                "learning_rate": current_lr,
            }
        )

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            bad_epochs = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["training"]["early_stopping_patience"]):
                break

    save_history(output_dir / "history.csv", history)

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    summary_rows = []
    for split_name, frame in (
        ("train", bundle.train_frame),
        ("valid", bundle.valid_frame),
        ("test", bundle.test_frame),
    ):
        split_summary = evaluate_split(
            model=model,
            frame=frame,
            bundle=bundle,
            split_name=split_name,
            output_dir=output_dir,
            device=device,
            save_excel=bool(cfg["evaluation"]["save_excel"]),
        )
        summary_rows.append(split_summary)

    pd.DataFrame(summary_rows).to_csv(
        output_dir / "metrics_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "device": str(device),
        "best_model": str(best_model_path),
        "output_dir": str(output_dir),
        "best_valid_loss": best_val_loss,
    }


def run_epoch(
    model: DPLGR4JNet,
    loader: DataLoader,
    criterion: NSELoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    warmup_length: int,
    grad_clip_norm: float | None,
    train_mode: bool,
) -> float:
    model.train(mode=train_mode)
    losses = []

    for batch in loader:
        x_raw, z_norm, y = move_batch_to_sequence_first(batch, device)
        target = y[warmup_length:]

        if train_mode and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x_raw, z_norm)
        loss = criterion(pred, target)

        if train_mode and optimizer is not None:
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses))


def move_batch_to_sequence_first(
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_raw, z_norm, y = batch
    x_raw = x_raw.to(device).transpose(0, 1)
    z_norm = z_norm.to(device).transpose(0, 1)
    y = y.to(device).transpose(0, 1)
    return x_raw, z_norm, y


def evaluate_split(
    model: DPLGR4JNet,
    frame: pd.DataFrame,
    bundle,
    split_name: str,
    output_dir: Path,
    device: torch.device,
    save_excel: bool,
) -> dict[str, Any]:
    model.eval()
    x_raw, z_norm, y, dates = build_full_sequence_tensors(frame, bundle)
    x_raw = x_raw.to(device)
    z_norm = z_norm.to(device)

    with torch.no_grad():
        pred, extras = model(x_raw, z_norm, return_extras=True)

    warmup = bundle.warmup_length
    aligned_dates = dates[warmup:]
    observed = y[warmup:, 0, 0].cpu().numpy()
    predicted = pred[:, 0, 0].detach().cpu().numpy()

    metrics = summarize_metrics(observed, predicted)
    parameter_values = extras["parameters_physical"][0].detach().cpu().numpy()
    parameter_frame = pd.DataFrame(
        {
            "x1": [parameter_values[0]],
            "x2": [parameter_values[1]],
            "x3": [parameter_values[2]],
            "x4": [parameter_values[3]],
        }
    )

    save_prediction_artifacts(
        output_dir=output_dir,
        split_name=split_name,
        dates=aligned_dates,
        observed_mmday=observed,
        predicted_mmday=predicted,
        area_km2=bundle.meta.area_km2,
        metrics=metrics,
        parameters=parameter_frame,
        save_excel=save_excel,
    )

    return {"split": split_name, **metrics}
