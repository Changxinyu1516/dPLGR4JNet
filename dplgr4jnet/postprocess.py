from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def mmday_to_m3s(values, area_km2: float):
    return values * area_km2 * 1000.0 / 86400.0


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_history(path: str | Path, history: list[dict[str, Any]]) -> None:
    pd.DataFrame(history).to_csv(path, index=False)


def save_prediction_artifacts(
    output_dir: str | Path,
    split_name: str,
    dates,
    observed_mmday,
    predicted_mmday,
    area_km2: float,
    metrics: dict[str, float],
    parameters: pd.DataFrame | None = None,
    save_excel: bool = True,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "observed_mm_day": observed_mmday,
            "predicted_mm_day": predicted_mmday,
            "observed_m3_s": mmday_to_m3s(observed_mmday, area_km2),
            "predicted_m3_s": mmday_to_m3s(predicted_mmday, area_km2),
        }
    )
    frame.to_csv(output_path / f"{split_name}_predictions.csv", index=False, encoding="utf-8-sig")

    if parameters is not None:
        parameters.to_csv(
            output_path / f"{split_name}_parameters.csv",
            index=False,
            encoding="utf-8-sig",
        )

    save_json(output_path / f"{split_name}_metrics.json", metrics)

    if save_excel:
        excel_path = output_path / f"{split_name}_predictions.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            frame.to_excel(writer, sheet_name="predictions", index=False)
            pd.DataFrame([metrics]).to_excel(writer, sheet_name="metrics", index=False)
            if parameters is not None:
                parameters.to_excel(writer, sheet_name="parameters", index=False)
