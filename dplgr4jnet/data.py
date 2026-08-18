from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class StationMeta:
    station_name: str
    source_file: str
    area_km2: float
    area_source: str
    runoff_input_unit: str


@dataclass
class FeatureScaler:
    mean: np.ndarray
    std: np.ndarray
    feature_names: list[str]

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_names: list[str]) -> "FeatureScaler":
        values = frame[feature_names].to_numpy(dtype=np.float32)
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        return cls(mean, np.where(std < 1e-6, 1.0, std), feature_names)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def to_dict(self) -> dict[str, Any]:
        return {"feature_names": self.feature_names, "mean": self.mean.tolist(), "std": self.std.tolist()}


@dataclass
class DataBundle:
    full_frame: pd.DataFrame
    train_frame: pd.DataFrame
    valid_frame: pd.DataFrame
    test_frame: pd.DataFrame | None
    scaler: FeatureScaler
    feature_columns: list[str]
    target_column: str
    sequence_length: int
    warmup_length: int
    window_stride: int
    meta: StationMeta


class WindowedSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, frame: pd.DataFrame, scaler: FeatureScaler, feature_columns: list[str], target_column: str, sequence_length: int, stride: int):
        if len(frame) < sequence_length:
            raise ValueError(f"Sequence length {sequence_length} is longer than split size {len(frame)}")
        self.sequence_length = int(sequence_length)
        self.values_raw = frame[feature_columns].to_numpy(dtype=np.float32)
        self.values_norm = scaler.transform(self.values_raw).astype(np.float32)
        self.targets = frame[[target_column]].to_numpy(dtype=np.float32)
        self.starts = list(range(0, len(frame) - sequence_length + 1, stride))
        if not self.starts:
            raise ValueError("No training windows were created; check sequence_length and window_stride")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int):
        start = self.starts[index]
        end = start + self.sequence_length
        return (
            torch.from_numpy(self.values_raw[start:end]),
            torch.from_numpy(self.values_norm[start:end]),
            torch.from_numpy(self.targets[start:end]),
        )


def _resolve_csv(data_cfg: dict[str, Any]) -> Path:
    data_dir = Path(data_cfg["data_dir"])
    configured = data_cfg.get("file")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else data_dir / path
    station = str(data_cfg["station_name"])
    matches = sorted(data_dir.glob(f"{station}*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one CSV for station {station!r} in {data_dir}, found {len(matches)}")
    return matches[0]


def _resolve_area_km2(data_cfg: dict[str, Any]) -> tuple[float, str]:
    """Resolve basin area from an explicit value or the CAMELS topology table."""
    explicit_area = data_cfg.get("area_km2")
    if explicit_area is not None:
        area = float(explicit_area)
        if area <= 0:
            raise ValueError("data.area_km2 must be positive")
        return area, "data.area_km2"

    data_dir = Path(data_cfg["data_dir"])
    topo_file = Path(data_cfg.get("topo_file", "camels_topo.txt"))
    topo_path = topo_file if topo_file.is_absolute() else data_dir / topo_file
    if not topo_path.is_file():
        raise FileNotFoundError(
            f"Basin topology file was not found: {topo_path}. "
            "Set data.topo_file or provide data.area_km2 explicitly."
        )
    gauge_column = str(data_cfg.get("topo_gauge_column", "gauge_id"))
    area_column = str(data_cfg.get("area_column", "area_gages2"))
    topology = pd.read_csv(topo_path, sep=None, engine="python", dtype={gauge_column: str})
    missing = [column for column in (gauge_column, area_column) if column not in topology.columns]
    if missing:
        raise ValueError(
            f"Missing topology columns {missing} in {topo_path}; "
            f"available columns: {list(topology.columns)}"
        )
    station_name = str(data_cfg["station_name"]).strip()
    gauge_ids = topology[gauge_column].astype(str).str.strip()
    matched = topology.loc[gauge_ids == station_name]
    if len(matched) != 1:
        raise ValueError(
            f"Expected exactly one topology row for station {station_name!r}, found {len(matched)}"
        )
    area = float(matched.iloc[0][area_column])
    if not np.isfinite(area) or area <= 0:
        raise ValueError(
            f"Invalid basin area {area!r} for station {station_name!r} in column {area_column!r}"
        )
    return area, f"{topo_path}:{area_column}"


def _to_mm_day(runoff: pd.Series, unit: str, area_km2: float) -> pd.Series:
    canonical = unit.lower().replace(" ", "")
    if canonical in {"mm/day", "mm/d", "mmday"}:
        return runoff.astype(float)
    if area_km2 <= 0:
        raise ValueError("data.area_km2 must be positive when runoff is a discharge unit")
    if canonical in {"m3/s", "m^3/s", "cms"}:
        return runoff.astype(float) * 86400.0 / (area_km2 * 1000.0)
    if canonical in {"ft3/s", "ft^3/s", "cfs"}:
        return runoff.astype(float) * 0.028316846592 * 86400.0 / (area_km2 * 1000.0)
    raise ValueError("data.runoff_unit must be mm/day, m3/s, or ft3/s")


def load_data_bundle(cfg: dict[str, Any]) -> DataBundle:
    data_cfg = cfg["data"]
    path = _resolve_csv(data_cfg)
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV was not found: {path}")
    date_column = data_cfg["date_column"]
    feature_columns = list(data_cfg["feature_columns"])
    source_target = data_cfg["target_column"]
    required = [date_column, *feature_columns, source_target]
    frame = pd.read_csv(path)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}; available columns: {list(frame.columns)}")
    frame = frame[required].copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    for column in [*feature_columns, source_target]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().drop_duplicates(date_column).set_index(date_column).sort_index()
    area_km2, area_source = _resolve_area_km2(data_cfg)
    runoff_unit = str(data_cfg.get("runoff_unit", "mm/day"))
    internal_target = "__runoff_mm_day__"
    frame[internal_target] = _to_mm_day(frame[source_target], runoff_unit, area_km2)
    frame = frame.drop(columns=[source_target]) if source_target not in feature_columns else frame

    train_frame = slice_period(frame, data_cfg["train_period"])
    valid_frame = slice_period(frame, data_cfg["valid_period"])
    test_period = data_cfg.get("test_period")
    test_frame = slice_period(frame, test_period) if test_period is not None else None
    if train_frame.empty or valid_frame.empty or (test_frame is not None and test_frame.empty):
        raise ValueError("At least one configured period is empty after slicing")
    scaler = FeatureScaler.fit(train_frame, feature_columns)
    return DataBundle(
        full_frame=frame,
        train_frame=train_frame,
        valid_frame=valid_frame,
        test_frame=test_frame,
        scaler=scaler,
        feature_columns=feature_columns,
        target_column=internal_target,
        sequence_length=int(data_cfg["sequence_length"]),
        warmup_length=int(data_cfg["warmup_length"]),
        window_stride=int(data_cfg["window_stride"]),
        meta=StationMeta(
            str(data_cfg["station_name"]),
            str(path),
            area_km2,
            area_source,
            runoff_unit,
        ),
    )


def build_dataset(frame: pd.DataFrame, bundle: DataBundle) -> WindowedSequenceDataset:
    return WindowedSequenceDataset(frame, bundle.scaler, bundle.feature_columns, bundle.target_column, bundle.sequence_length, bundle.window_stride)


def build_full_sequence_tensors(frame: pd.DataFrame, bundle: DataBundle):
    raw = frame[bundle.feature_columns].to_numpy(dtype=np.float32)
    normalized = bundle.scaler.transform(raw).astype(np.float32)
    target = frame[[bundle.target_column]].to_numpy(dtype=np.float32)
    return torch.from_numpy(raw).unsqueeze(1), torch.from_numpy(normalized).unsqueeze(1), torch.from_numpy(target).unsqueeze(1), frame.index


def slice_period(frame: pd.DataFrame, period: list[str]) -> pd.DataFrame:
    return frame.loc[str(period[0]) : str(period[1])].copy()
