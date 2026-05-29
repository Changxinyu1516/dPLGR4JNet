from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


FORCING_SPECS = {
    "precipitation": ("pre_evap_Grid_Data.xlsx", "MeanPrecip1"),
    "evapotranspiration": ("pre_evap_Grid_Data.xlsx", "MeanEvapor1"),
    "relative_humidity": ("rhum_Grid_SX_Data.xlsx", "Mean_rhum1"),
    "solar_radiation": ("srad_Grid_SX_Data.xlsx", "Mean_srad1"),
    "temperature_mean": ("temp_Grid_SX_Data.xlsx", "Mean_temp1"),
}

RUNOFF_FILE = "三峡库区子流域日尺度流量数据.xlsx"
RUNOFF_SHEET = "流量数据"
ATTR_FILE = "三峡库区子流域静态属性.xlsx"
ATTR_SHEET = "原始数据"
AREA_COLUMN = "流域面积"
SUFFIXES = ("以上", "以下")


@dataclass
class StationMeta:
    station_name: str
    basin_name: str
    runoff_name: str
    area_km2: float
    attributes: dict[str, float]


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
        std = np.where(std < 1e-6, 1.0, std)
        return cls(mean=mean, std=std, feature_names=feature_names)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }


@dataclass
class DataBundle:
    full_frame: pd.DataFrame
    train_frame: pd.DataFrame
    valid_frame: pd.DataFrame
    test_frame: pd.DataFrame
    scaler: FeatureScaler
    feature_columns: list[str]
    target_column: str
    sequence_length: int
    warmup_length: int
    window_stride: int
    meta: StationMeta


class WindowedSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        frame: pd.DataFrame,
        scaler: FeatureScaler,
        feature_columns: list[str],
        target_column: str,
        sequence_length: int,
        stride: int,
    ) -> None:
        if len(frame) < sequence_length:
            raise ValueError(
                f"Sequence length {sequence_length} is longer than the split size {len(frame)}."
            )

        self.values_raw = frame[feature_columns].to_numpy(dtype=np.float32)
        self.values_norm = scaler.transform(self.values_raw).astype(np.float32)
        self.targets = frame[[target_column]].to_numpy(dtype=np.float32)
        self.starts = list(range(0, len(frame) - sequence_length + 1, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        end = start + self.values_raw.shape[0] * 0 + 0
        end = start + self.sequence_length
        x_raw = torch.from_numpy(self.values_raw[start:end])
        z_norm = torch.from_numpy(self.values_norm[start:end])
        y = torch.from_numpy(self.targets[start:end])
        return x_raw, z_norm, y

    @property
    def sequence_length(self) -> int:
        if not hasattr(self, "_sequence_length"):
            raise AttributeError("sequence_length has not been initialized.")
        return self._sequence_length

    @sequence_length.setter
    def sequence_length(self, value: int) -> None:
        self._sequence_length = value


def load_data_bundle(cfg: dict[str, Any]) -> DataBundle:
    data_cfg = cfg["data"]
    data_dir = Path(data_cfg["data_dir"])
    feature_columns = list(data_cfg["feature_columns"])
    target_column = data_cfg["target_column"]

    attr_df = pd.read_excel(data_dir / ATTR_FILE, sheet_name=ATTR_SHEET, index_col=0)
    runoff_df = pd.read_excel(data_dir / RUNOFF_FILE, sheet_name=RUNOFF_SHEET)

    forcing_columns = pd.read_excel(
        data_dir / FORCING_SPECS["precipitation"][0],
        sheet_name=FORCING_SPECS["precipitation"][1],
        nrows=0,
    ).columns[1:]
    runoff_columns = runoff_df.columns[1:]

    basin_name = resolve_name(
        preferred=data_cfg.get("basin_name"),
        fallback=data_cfg["station_name"],
        available=forcing_columns,
        label="forcing/attribute basin",
    )
    runoff_name = resolve_name(
        preferred=data_cfg.get("runoff_name"),
        fallback=data_cfg["station_name"],
        available=runoff_columns,
        label="runoff station",
    )

    attr_row = attr_df.loc[basin_name]
    area_km2 = float(attr_row[AREA_COLUMN])

    forcing_series = []
    for feature in feature_columns:
        file_name, sheet_name = FORCING_SPECS[feature]
        frame = pd.read_excel(
            data_dir / file_name,
            sheet_name=sheet_name,
            index_col=0,
            parse_dates=True,
        )
        series = frame[basin_name].rename(feature)
        forcing_series.append(series)

    runoff = runoff_df.iloc[:, [0, runoff_df.columns.get_loc(runoff_name)]].copy()
    runoff.columns = ["date", target_column]
    runoff["date"] = pd.to_datetime(runoff["date"])
    runoff = runoff.dropna().set_index("date")
    runoff[target_column] = runoff[target_column].astype(float) * 86400.0 / (area_km2 * 1000.0)

    full_frame = pd.concat(forcing_series + [runoff[target_column]], axis=1).sort_index()
    full_frame = full_frame.dropna()

    train_frame = slice_period(full_frame, data_cfg["train_period"])
    valid_frame = slice_period(full_frame, data_cfg["valid_period"])
    test_frame = slice_period(full_frame, data_cfg["test_period"])

    if len(train_frame) == 0 or len(valid_frame) == 0 or len(test_frame) == 0:
        raise ValueError("At least one split is empty after slicing by date.")

    scaler = FeatureScaler.fit(train_frame, feature_columns)
    meta = StationMeta(
        station_name=data_cfg["station_name"],
        basin_name=basin_name,
        runoff_name=runoff_name,
        area_km2=area_km2,
        attributes={
            str(key): float(value)
            for key, value in attr_row.to_dict().items()
            if pd.notna(value)
        },
    )

    return DataBundle(
        full_frame=full_frame,
        train_frame=train_frame,
        valid_frame=valid_frame,
        test_frame=test_frame,
        scaler=scaler,
        feature_columns=feature_columns,
        target_column=target_column,
        sequence_length=int(data_cfg["sequence_length"]),
        warmup_length=int(data_cfg["warmup_length"]),
        window_stride=int(data_cfg["window_stride"]),
        meta=meta,
    )


def build_dataset(
    frame: pd.DataFrame,
    bundle: DataBundle,
) -> WindowedSequenceDataset:
    dataset = WindowedSequenceDataset(
        frame=frame,
        scaler=bundle.scaler,
        feature_columns=bundle.feature_columns,
        target_column=bundle.target_column,
        sequence_length=bundle.sequence_length,
        stride=bundle.window_stride,
    )
    dataset.sequence_length = bundle.sequence_length
    return dataset


def build_full_sequence_tensors(
    frame: pd.DataFrame,
    bundle: DataBundle,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DatetimeIndex]:
    x_raw = frame[bundle.feature_columns].to_numpy(dtype=np.float32)
    z_norm = bundle.scaler.transform(x_raw).astype(np.float32)
    y = frame[[bundle.target_column]].to_numpy(dtype=np.float32)
    dates = frame.index
    return (
        torch.from_numpy(x_raw).unsqueeze(1),
        torch.from_numpy(z_norm).unsqueeze(1),
        torch.from_numpy(y).unsqueeze(1),
        dates,
    )


def slice_period(frame: pd.DataFrame, period: list[str]) -> pd.DataFrame:
    start, end = period
    return frame.loc[start:end].copy()


def resolve_name(
    preferred: str | None,
    fallback: str,
    available: pd.Index | list[str],
    label: str,
) -> str:
    available_list = [str(item) for item in available]
    for candidate in candidate_names(preferred, fallback):
        if candidate in available_list:
            return candidate

    base = strip_suffix(preferred or fallback)
    fuzzy = [item for item in available_list if base and base in item]
    if len(fuzzy) == 1:
        return fuzzy[0]

    raise ValueError(
        f"Could not resolve {label} name from '{preferred or fallback}'. "
        f"Available examples: {available_list[:12]}"
    )


def candidate_names(preferred: str | None, fallback: str) -> list[str]:
    ordered: list[str] = []
    for raw_name in (preferred, fallback):
        if raw_name is None:
            continue
        ordered.extend(expand_name(raw_name))
    seen = set()
    result = []
    for name in ordered:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def expand_name(name: str) -> list[str]:
    name = str(name).strip()
    base = strip_suffix(name)
    options = [name]
    if base != name:
        options.append(base)
    options.extend([f"{base}{suffix}" for suffix in SUFFIXES])
    return options


def strip_suffix(name: str) -> str:
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name
